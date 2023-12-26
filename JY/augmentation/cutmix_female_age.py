import os
import sys
from glob import glob
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn

# cutmix 함수
def cutmix(img1, label1, img2, label2, beta=1.0):
    # 이미지 크기 가져오기
    height, width = img1.size[0], img1.size[1]
    
    # CutMix 파라미터 계산
    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox((height, width), lam)

    # 첫 번째 이미지에서 잘라낸 부분을 두 번째 이미지에 삽입
    img1.paste(img2.crop((bbx1, bby1, bbx2, bby2)), (bbx1, bby1))
    
    # 라벨을 섞음
    label1 = label1 * lam + label2 * (1. - lam)

    return img1, label1

# 바운딩 박스 랜덤으로 생성
def rand_bbox(size, lam):
    W = size[1]
    H = size[0]
    '''
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # 중심 좌표 랜덤 선택
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # bbox 생성
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    '''
    reduced_size = int(np.sqrt(lam) * min(W, H) / 0.05)
    reduced_height = int(H * 0.3)
    
    offset_x = 0  # 가로 길이
    offset_y = 0  # 세로 길이

    # bbox 생성
    bbx1 = offset_x
    bby1 = offset_y
    bbx2 = W
    bby2 = H - reduced_height

    return bbx1, bby1, bbx2, bby2

# 마스크 라벨링
_file_names = {
    "mask1": 0,
    "mask2": 0,
    "mask3": 0,
    "mask4": 0,
    "mask5": 0,
    "incorrect_mask": 1,
    "normal": 2,
}

def from_number(value: str) -> int:
    try:
        value = int(value)
    except Exception:
        raise ValueError(f"Age value should be numeric, {value}")

    if value < 30:
        return 0
    elif value < 60:
        return 1
    else:
        return 2

# 나이와 마스크 착용 유무
young0, young1, young2, middle0, middle1, middle2, old0, old1, old2 = [], [], [], [], [], [], [], [], []
y_paths0, y_paths1, y_paths2, m_paths0, m_paths1, m_paths2, o_paths0, o_paths1, o_paths2 = [], [], [], [], [], [], [], [], []

data_dir = '/data/ephemeral/home/jero/multi_ens_code/data/Female'
profiles = os.listdir(data_dir)

# 컷믹스 과정
for profile in profiles:
    img_folder = os.path.join(data_dir, profile)
    for file_name in os.listdir(img_folder):
        _file_name, ext = os.path.splitext(file_name)
            
        img_path = os.path.join(
            data_dir, profile, file_name
        )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
        if _file_name in _file_names:
            mask_label = _file_names[_file_name]
        else:
            continue

        id, gender, race, age = profile.split("_")
        age = from_number(age)
        
        # 조건에 맞게 데이터 구분
        if age == 0 and mask_label == 0:
            young0.append(file_name)
            y_paths0.append(img_path)
        elif age == 0 and mask_label == 1:
            young1.append(file_name)
            y_paths1.append(img_path)
        elif age == 0 and mask_label == 2:
            young2.append(file_name)
            y_paths2.append(img_path)
        elif age == 1 and mask_label == 0:
            middle0.append(file_name)
            m_paths0.append(img_path)
        elif age == 1 and mask_label == 1:
            middle1.append(file_name)
            m_paths1.append(img_path)
        elif age == 1 and mask_label == 2:
            middle2.append(file_name)
            m_paths2.append(img_path)
        elif age == 2 and mask_label == 0:
            old0.append(file_name)
            o_paths0.append(img_path)
        elif age == 2 and mask_label == 1:
            old1.append(file_name)
            o_paths1.append(img_path)
        elif age == 2 and mask_label == 2:
            old2.append(file_name)
            o_paths2.append(img_path)

base_path = '/data/ephemeral/home/jero/multi_ens_code/output_female_age'

# young + (old)
young_id = 8040
for i in range(30):
    folder_name = f"{young_id:06d}_female_Asian_60"
    folder_path = os.path.join(base_path, folder_name)
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")
    young_id += 1

    for j in range(3):
        if j == 0:
            img1_path = y_paths0[i+30]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = o_paths0[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "mask1.jpg")
            mixed_img.save(output_path)
        elif j == 1:
            img1_path = y_paths1[i+30]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = o_paths1[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "incorrect.jpg")
            mixed_img.save(output_path)
        elif j == 2:
            img1_path = y_paths2[i+30]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = o_paths2[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "normal.jpg")
            mixed_img.save(output_path)


# middle + (old)
middle_id = 8130
for i in range(30):
    folder_name = f"{middle_id:06d}_female_Asian_60"
    folder_path = os.path.join(base_path, folder_name)
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")
    middle_id += 1

    for j in range(3):
        if j == 0:
            img1_path = m_paths0[i+30]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = o_paths0[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "mask1.jpg")
            mixed_img.save(output_path)
        elif j == 1:
            img1_path = m_paths1[i+30]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = o_paths1[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "incorrect.jpg")
            mixed_img.save(output_path)
        elif j == 2:
            img1_path = m_paths2[i+30]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = o_paths2[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "normal.jpg")
            mixed_img.save(output_path)

'''
# old + (young)
old_1_id = 8200
for i in range(30):
    folder_name = f"{old_1_id:06d}_female_Asian_20"
    folder_path = os.path.join(base_path, folder_name)
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")
    old_1_id += 1

    for j in range(3):
        if j == 0:
            img1_path = o_paths0[i]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = y_paths0[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "mask1.jpg")
            mixed_img.save(output_path)
        elif j == 1:
            img1_path = o_paths1[i]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = y_paths1[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "incorrect.jpg")
            mixed_img.save(output_path)
        elif j == 2:
            img1_path = o_paths2[i]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = y_paths2[i]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "normal.jpg")
            mixed_img.save(output_path)
'''

# old + (middle)
old_2_id = 8230
for i in range(30):
    folder_name = f"{old_2_id:06d}_female_Asian_40"
    folder_path = os.path.join(base_path, folder_name)
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")
    old_2_id += 1

    for j in range(3):
        if j == 0:
            img1_path = o_paths0[i]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = m_paths0[i+30]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "mask1.jpg")
            mixed_img.save(output_path)
        elif j == 1:
            img1_path = o_paths1[i]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = m_paths1[i+30]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "incorrect.jpg")
            mixed_img.save(output_path)
        elif j == 2:
            img1_path = o_paths2[i]
            img1 = Image.open(img1_path)
            label1 = torch.tensor([1.0, 0.0])

            img2_path = m_paths2[i+30]
            img2 = Image.open(img2_path)
            label2 = torch.tensor([0.0, 1.0])

            mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

            output_path = os.path.join(folder_path, "normal.jpg")
            mixed_img.save(output_path)

# print(len(young1))
# print(len(middle1))
# print(len(old1))
'''
img1_path = o_paths0[18]
img1 = Image.open(img1_path)
label1 = torch.tensor([1.0, 0.0])

img2_path = y_paths0[18]
img2 = Image.open(img2_path)
label2 = torch.tensor([0.0, 1.0])

mixed_img, mixed_label = cutmix(img1, label1, img2, label2)
# m_f.append(mixed_img)

output_path = os.path.join(base_path, f"male_female_normal.jpg")
mixed_img.save(output_path)
'''