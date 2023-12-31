import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from train import increment_path
import numpy as np

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    """
    저장된 모델의 가중치를 로드하는 함수입니다.

    Args:
        saved_model (str): 모델 가중치가 저장된 디렉토리 경로
        num_classes (int): 모델의 클래수 수
        device (torch.device): 모델이 로드될 장치 (CPU 또는 CUDA)

    Returns:
        model (nn.Module): 가중치가 로드된 모델
    """
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # 모델 가중치를 로드한다.
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
    model = load_model(model_dir, num_classes, device).to(device)
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
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    mask_preds = []
    gender_preds = []
    age_preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            mask_outs, gender_outs, age_outs = model(images)
            mask_pred = torch.argmax(mask_outs, dim=-1)
            gender_pred = torch.argmax(gender_outs, dim=-1)
            age_pred = torch.argmax(age_outs, dim=-1)
            mask_preds.extend(mask_pred.cpu().numpy())
            gender_preds.extend(gender_pred.cpu().numpy())
            age_preds.extend(age_pred.cpu().numpy())

    return mask_preds, gender_preds, age_preds

def inference_output(mask, gender, age):

    # 예측 결과를 데이터프레임에 저장하고 csv 파일로 출력한다.
    info = pd.read_csv('/data/ephemeral/home/competition_1/data/eval/info.csv')
    val = mask*6 + gender*3 + age
    val = val.argmax(dim=-1)

    info["ans"] = val
    save_path = os.path.join(output_dir, f"ensemble.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

def ensemble(out_mask,out_gender,out_age, output_dir):

    info = pd.read_csv('/data/ephemeral/home/competition_1/data/eval/info.csv')

    # 각 행에서 가장 자주 나오는 값을 찾아 대표 값으로 설정

    mask = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=out_mask)
    gender = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=out_gender)
    age = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=out_age)
    val = mask*6 + gender*3 + age

    info["ans"] = val
    save_path = os.path.join(output_dir, f"ensemble.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    # 커맨드 라인 인자를 파싱한다.
    parser = argparse.ArgumentParser()

    # 데이터와 모델 체크포인트 디렉터리 관련 인자
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=(128, 96),
        help="resize size for image when you trained (default: (96, 128))",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )

    # 컨테이너 환경 변수
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/ephemeral/home/competition_1/data/eval"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="~/competition_1/model/exp6"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
        default= "~/output/competition_1_result/level1-imageclassification-cv-03/TY"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp"
    )
    parser.add_argument(
        "--ensemble",
        type=bool,
        default=False,
        help="Model ensemble flag(default: False)"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if not args.ensemble:
        # 모델 추론을 수행한다.    
        out1_mask, out1_gender, out1_age = inference(data_dir, model_dir, output_dir, args)
        inference_output(out1_mask, out1_gender, out1_age)

    else:
        out1_mask, out1_gender, out1_age = inference(data_dir, model_dir, output_dir, args)
        
        args.model='Conv2Model'
        args.resize=(224, 224)
        out2_mask, out2_gender, out2_age = inference(data_dir, '/data/ephemeral/home/competition_1/model/Conv2Model_scaler2', output_dir, args)

        args.model='EvaModel'
        args.resize=(448, 448)
        args.batch_size=128
        out3_mask, out3_gender, out3_age = inference(data_dir, '/data/ephemeral/home/competition_1/model/EvaModel_scaler', output_dir, args)

        args.model='Swin_tiny_scaler2'
        args.resize=(224, 224)
        out4_mask, out4_gender, out4_age = inference(data_dir, '/data/ephemeral/home/competition_1/model/Swin_tiny_scaler2', output_dir, args)

        out_mask = np.stack([out1_mask, out1_mask, out2_mask, out3_mask, out4_mask], 1)
        out_gender = np.stack([out1_gender, out1_gender, out2_gender, out3_gender, out4_gender], 1)
        out_age = np.stack([out1_age, out1_age, out2_age, out3_age, out4_age], 1)

        ensemble(out_mask,out_gender,out_age,output_dir)
