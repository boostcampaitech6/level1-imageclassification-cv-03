import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from dataset import MaskBaseDataset, BaseAugmentation_for_prefetch, fast_collate, PrefetchLoader
from loss import create_criterion
from lion_pytorch import Lion
from adamp import AdamP
import wandb
from sklearn.metrics import f1_score

def wandb_init(args):
        
    wandb.init(
        # set the wandb project where this run will be logged
        project="competition_1",
        
        # track hyperparameters and run metadata
        config=args
    )
    wandb.run.name = f"TY_{args.model}_{args.dataset}_{args.optimizer}_{args.augmentation}_{args.criterion}_{args.epochs}_{args.lr}_{args.batch_size}"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def hard_voting(models, data):
    outputs = []
    for model in models:
        mask_outputs, gender_outputs, age_outputs = model(data)
        predicted_class = mask_outputs*6 + gender_outputs*3 + age_outputs
        outputs.append(predicted_class)

    return torch.stack(outputs, dim=0).mean(0)

def hard_voting(models, data):
    outputs = []
    for model in models:
        mask_outputs, gender_outputs, age_outputs = model(data)
        predicted_class = mask_outputs * 6 + gender_outputs * 3 + age_outputs
        outputs.append(predicted_class)

    return torch.stack(outputs, dim=0).mean(0)

def soft_voting(mask_preds, gender_preds, age_preds):
    outputs = mask_preds * 6 + gender_preds * 3 + age_preds
    probabilities = nn.Softmax(dim=1)(torch.stack(outputs, dim=0).mean(0))
    predicted_class = probabilities[1].max(1)
    return predicted_class


def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    torch.cuda.empty_cache()

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device : {device}")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    # transform_module = getattr(
    #     import_module("dataset"), args.augmentation
    # )  # default: BaseAugmentation
    # transform = transform_module(
    #     resize=args.resize,
    #     mean=dataset.mean,
    #     std=dataset.std,
    # )
    transform = BaseAugmentation_for_prefetch(
        resize=args.resize
    ) # for prefetch, using ToNumpy, no normalization on cpu
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
        collate_fn=fast_collate
    )
    train_loader = PrefetchLoader(
        train_loader,
        mean=dataset.mean,
        std=dataset.std
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
        collate_fn=fast_collate
    )
    val_loader = PrefetchLoader(
        val_loader,
        mean=dataset.mean,
        std=dataset.std
    )

    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=args.batch_size,
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     shuffle=True,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )

    # val_loader = DataLoader(
    #     val_set,
    #     batch_size=args.valid_batch_size,
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     shuffle=False,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    scaler = torch.cuda.amp.GradScaler()

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    if args.wandb:
        wandb_init(args)


    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            # inputs, labels = train_batch
            inputs, mask_labels, gender_labels, age_labels = train_batch
            inputs = inputs.to(device)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)

            # model 1
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                mask_outs, gender_outs, age_outs = model(inputs)
                mask_preds = torch.argmax(mask_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                age_preds = torch.argmax(age_outs, dim=-1)
                
                mask_loss = criterion(mask_outs, mask_labels)
                gender_loss = criterion(gender_outs, gender_labels)
                age_loss = criterion(age_outs, age_labels)

                loss = mask_loss + gender_loss + args.gamma * age_loss

                labels = mask_labels * 6 + gender_labels * 3 + age_labels
                preds = mask_preds * 6 + gender_preds * 3 + age_preds

            # with torch.cuda.amp.autocast():
            #     outs = model(inputs)
                
            #     loss = criterion(outs,labels)
            #     preds = torch.argmax(outs, dim=-1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4}|| "
                    f"training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                if args.wandb:
                    wandb.log({
                        "train acc": train_acc,
                        "train loss": train_loss
                    })
                # logger.add_scalar(
                #     "Train/loss", train_loss, epoch * len(train_loader) + idx
                # )
                # logger.add_scalar(
                #     "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                # )

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                # inputs, labels = val_batch
                inputs, mask_labels, gender_labels, age_labels = val_batch
                inputs = inputs.to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    mask_outs, gender_outs, age_outs = model(inputs)
                    mask_preds = torch.argmax(mask_outs, dim=-1)
                    gender_preds = torch.argmax(gender_outs, dim=-1)
                    age_preds = torch.argmax(age_outs, dim=-1)
                    
                    mask_loss = criterion(mask_outs, mask_labels)
                    gender_loss = criterion(gender_outs, gender_labels)
                    age_loss = criterion(age_outs, age_labels)

                    loss = mask_loss + gender_loss + args.gamma * age_loss

                    labels = mask_labels * 6 + gender_labels * 3 + age_labels
                    preds = mask_preds * 6 + gender_preds * 3 + age_preds
                
                # with torch.cuda.amp.autocast():
                #     outs = model(inputs)
                #     loss = criterion(outs, labels)
                #     preds = torch.argmax(outs, dim=-1)

                loss_item = loss.item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    figure = grid_image(
                        inputs_np,
                        labels,
                        preds,
                        n=16,
                        shuffle=args.dataset != "MaskSplitByProfileDataset",
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} || "
                f"f1 score : {f1_score(preds.cpu(),labels.cpu(),average='macro')}"
            )
            # logger.add_scalar("Val/loss", val_loss, epoch)
            # logger.add_scalar("Val/accuracy", val_acc, epoch)
            # logger.add_figure("results", figure, epoch)
            if args.wandb:
                wandb.log({
                    "val acc": val_acc, 
                    "val loss": val_loss,
                    "best val acc": best_val_acc,
                    "best val loss": best_val_loss
                })
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MultiLabelDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[128, 96],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=128,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="Resnet50", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", help="optimizer type (default: SGD)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="focal",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )
    parser.add_argument(
        "--wandb", type=bool, default=True, help="logging hyperparameters on wandb"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.5, help="multi label loss ratio"
    )
    torch.cuda.empty_cache()
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
