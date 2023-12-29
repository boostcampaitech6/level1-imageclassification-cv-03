import os
import yaml
import gdown
import torch
import streamlit as st
from PIL import Image
from importlib import import_module

from torchvision.transforms import(
    CenterCrop,
    Compose,
    Resize,
    ToTensor,
)

class PreprocessedFunc:
    def __init__(self, resize, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
    
def preprocessed_func(img):
    preprocessed = PreprocessedFunc(resize=(224, 224))
    return preprocessed(img)

def download_model_file(url):
    output = "streamlit_start/best.pth"
    gdown.download(url, output, quiet=False)

def load_model():
    model_cls = getattr(import_module('model'), config['model_name'])
    model = model_cls(18)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("streamlit_start/best.pth"):
        download_model_file(config['model_path'])
    
    model_path = 'streamlit_start/best.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def make_prediction(model, preprocessed_image):
    preprocessed_image_np = preprocessed_image.numpy()

    preprocessed_image_tensor = torch.from_numpy(preprocessed_image_np).unsqueeze(0)

    mask_outs, gender_outs, age_outs = model(preprocessed_image_tensor)
    mask_preds = torch.argmax(mask_outs, dim=-1)
    gender_preds = torch.argmax(gender_outs, dim=-1)
    age_preds = torch.argmax(age_outs, dim=-1)

    preds = mask_preds * 6 + gender_preds * 3 + age_preds
    return preds


if __name__ == '__main__':
        
    st.title('마스크 착용 여부, 성별, 연령을 판단')
    st.text('처음 실행시 모델 다운로드로 인해 오래 걸립니다.')
    
    upload = st.file_uploader(label='Upload Image', type=['png', 'jpg', 'jpeg'])
    categories = {0:["Wear", "Male", "30대 이하"],
                1:["Wear", "Male", "30대 이상 60대 미만"],
                2:["Wear", "Male", "60대 이상"],

                3:["Wear", "Female", "30대 이하"],
                4:["Wear", "Female", "30대 이상 60대 미만"],
                5:["Wear", "Female", "60대 이상"],

                6:["Incorrect", "Male", "30대 이하"],
                7:["Incorrect", "Male", "30대 이상 60대 미만"],
                8:["Incorrect", "Male", "60대 이상"],

                9:["Incorrect", "Female", "30대 이하"],
                10:["Incorrect", "Female" "30대 이상 60대 미만"],
                11:["Incorrect", "Female", "60대 이상"],
                12:["Not Wear", "Male", "30대 이하"],
                13:["Not Wear", "Male", "30대 이상 60대 미만"],
                14:["Not Wear", "Male",  "60대 이상"],

                15:["Not Wear", "Female", "30대 이하"],
                16:["Not Wear", "Female", "30대 이상 60대 미만"],
                17:["Not Wear", "Female", "60대 이상"]}
    
    with open("streamlit_start\config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if upload:        
        img = Image.open(upload).convert("RGB")
        model = load_model()
        preprocessed_image = preprocessed_func(img)
        labels = make_prediction(model, preprocessed_image).item()
        result = categories[labels]
        
        col1, col2 = st.columns(2)
        col1.image(img)

        col2.title("예측 결과")
        col2.text(f"마스크 유무: {result[0]}")
        col2.text(f"성별: {result[1]}")
        col2.text(f"나이: {result[2]}")