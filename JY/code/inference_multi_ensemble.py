import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model_1(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model1)
    model = model_cls(num_classes=num_classes)
    
    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_model_2(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model2)
    model = model_cls(num_classes=num_classes)
    
    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_model_3(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model3)
    model = model_cls(num_classes=num_classes)
    
    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    모델 추론을 수행하는 함수

    Args:
        data_dir (str): 테스트 데이터가 있는 디렉토리 경로
        model_dir (str): 모델 가중치가 저장된 디렉토리 경로
        output_dir (str): 결과 CSV를 저장할 디렉토리 경로
        args (argparse.Namespace): 커맨드 라인 인자

    Returns:
        None
    """

    # CUDA를 사용할 수 있는지 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 클래스의 개수를 설정한다. (마스크, 성별, 나이의 조합으로 18)
    num_classes = MaskBaseDataset.num_classes  # 18
    models = []
    for i in range(3):
        if i == 0:
            model_ens_dir = os.path.join(model_dir, 'ConvNeXt')
            model = load_model_1(model_ens_dir, num_classes, device).to(device)
        elif i == 1:
            model_ens_dir = os.path.join(model_dir, 'Swin')
            model = load_model_2(model_ens_dir, num_classes, device).to(device)
        elif i == 2:
            model_ens_dir = os.path.join(model_dir, 'ViT')
            model = load_model_3(model_ens_dir, num_classes, device).to(device)
        
        models.append(model)
    #model = load_model(model_dir, num_classes, device).to(device)
    #model.eval()

    for model in models:
        model.eval()
    
    # 이미지 파일 경로와 정보 파일을 읽어온다.
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    # 이미지 경로를 리스트로 생성한다.
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        # num_workers=multiprocessing.cpu_count() // 2,
        num_workers = 0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    #preds_mask = []
    #preds_gender = []
    #preds_age = []

    all_preds = [[] for _ in range(len(models))]

    for model_idx, model in enumerate(models):
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                mask, gender, age = model(images)
                mask_out = mask.argmax(dim=-1)
                gender_out = gender.argmax(dim=-1)
                age_out = age.argmax(dim=-1)
                pred = mask_out * 6 + gender_out * 3 + age_out
                preds.extend(pred.cpu().numpy())
        all_preds[model_idx] = preds

    final_preds = []
    for idx in range(len(all_preds[0])):
        voting = np.bincount([all_preds[model_idx][idx] for model_idx in range(len(models))])
        final_pred = voting.argmax()
        final_preds.append(final_pred)

    info['ans'] = final_preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


def ensemble(mask_out, gender_out, age_out, output_dir):
    info = pd.read_csv('/data/ephemeral/home/jero/eval/info.csv')
    preds = []
    weights = []

    mask = np.average(mask_out, axis=0)
    #mask = np.argmax(mask)
    gender = np.average(gender_out, axis=0)
    #gender = np.argmax(gender)
    age = np.average(age_out, axis=0)
    #age = np.argmax(age)
    
    pred = mask * 6 + gender * 3 + age
    pred
    preds.extend([pred])

    info["ans"] = preds
    save_path = os.path.join(output_dir, f"output.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    # 커맨드 라인 인자를 파싱한다.
    parser = argparse.ArgumentParser()

    # 데이터와 모델 체크포인트 디렉터리 관련 인자
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=(224, 224),
        help="resize size for image when you trained (default: (96, 128))",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--model1", type=str, default="ConvNeXt", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--model2", type=str, default="Swin", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--model3", type=str, default="ViT", help="model type (default: BaseModel)"
    )

    # 컨테이너 환경 변수
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    '''
    args.model = 'ViT'
    ViT_mask, ViT_gender, ViT_age = inference(data_dir, '/data/ephemeral/home/jero/model/ViT', output_dir, args, (224, 224))
    args.model = 'Swin'
    Swin_mask, Swin_gender, Swin_age = inference(data_dir, '/data/ephemeral/home/jero/model/Swin', output_dir, args, (224, 224))
    '''

    inference(data_dir, model_dir, output_dir, args)
