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

# 성별에 따른 마스크 착용 유무 구분해서 데이터 저장
female0, male0, female1, male1, female2, male2 = [], [], [], [], [], []
f_paths0, m_paths0, f_paths1, m_paths1, f_paths2, m_paths2 = [], [], [], [], [], []

data_dir = '/data/ephemeral/home/jero/multi_ens_code/OLD'
profiles = os.listdir(data_dir)

# 여자에 남자 추가
output_f_m = '/data/ephemeral/home/jero/multi_ens_code/output_f_m'
# 남자에 여자 추가
output_m_f = '/data/ephemeral/home/jero/multi_ens_code/output_m_f'

# 컷믹스 과정
for profile in profiles:
    img_folder = os.path.join(data_dir, profile)
    for file_name in os.listdir(img_folder):
        _file_name, ext = os.path.splitext(file_name)
            
        img_path = os.path.join(
            data_dir, profile, file_name
        )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
        mask_label = _file_names[_file_name]

        id, gender, race, age = profile.split("_")
        
        # 조건에 맞게 데이터 구분
        if gender == "female" and mask_label == 0:
            female0.append(file_name)
            f_paths0.append(img_path)
        elif gender == "female" and mask_label == 1:
            female1.append(file_name)
            f_paths1.append(img_path)
        elif gender == "female" and mask_label == 2:
            female2.append(file_name)
            f_paths2.append(img_path)
        elif gender == "male" and mask_label == 0:
            male0.append(file_name)
            m_paths0.append(img_path)
        elif gender == "male" and mask_label == 1:
            male1.append(file_name)
            m_paths1.append(img_path)
        elif gender == "male" and mask_label == 2:
            male2.append(file_name)
            m_paths2.append(img_path)
        
# 마스크 착용 여자 데이터에 마스크 착용 남자 데이터 추가
for i in range(len(female0)):
    cnt = 005600

    img1_path = f_paths0[i]
    img1 = Image.open(img1_path)
    label1 = torch.tensor([1.0, 0.0])

    img2_path = m_paths0[i]
    img2 = Image.open(img2_path)
    label2 = torch.tensor([0.0, 1.0])

    mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

    output_path = os.path.join(output_f_m, f"{cnt}_female_Asian_60.jpg")
    mixed_img.save(output_path)

    cnt += 1

for i in range(len(male0)):
    cnt = 006200

    img1_path = f_paths0[i]
    img1 = Image.open(img1_path)
    label1 = torch.tensor([1.0, 0.0])

    img2_path = m_paths0[i]
    img2 = Image.open(img2_path)
    label2 = torch.tensor([0.0, 1.0])

    mixed_img, mixed_label = cutmix(img1, label1, img2, label2)   

    output_path = os.path.join(output_f_m, f"{cnt}_male_Asian_60.jpg")
    mixed_img.save(output_path)

    cnt += 1

for i in range(len(female1)):
    img1 = Image.open(female1[i])
    label1 = torch.tensor([1.0, 0.0])

    img2 = Image.open(male1[i])
    label2 = torch.tensor([0.0, 1.0])

    mixed_img, mixed_label = cutmix(img1, label1, img2, label2)
    #f_m.append(mixed_img)

    output_path = os.path.join(output_f_m, f"female_male_incorrect_{i}.jpg")
    mixed_img.save(output_path)

for i in range(len(male1)):
    img1 = Image.open(male1[i])
    label1 = torch.tensor([1.0, 0.0])

    img2 = Image.open(female1[i])
    label2 = torch.tensor([0.0, 1.0])

    mixed_img, mixed_label = cutmix(img1, label1, img2, label2)
    m_f.append(mixed_img)

    output_path = os.path.join(output_m_f, f"male_female_incorrect_{i}.jpg")
    mixed_img.save(output_path)

for i in range(len(female2)):
    img1 = Image.open(female2[i])
    label1 = torch.tensor([1.0, 0.0])

    img2 = Image.open(male2[i])
    label2 = torch.tensor([0.0, 1.0])

    mixed_img, mixed_label = cutmix(img1, label1, img2, label2)
    #f_m.append(mixed_img)

    output_path = os.path.join(output_f_m, f"female_male_normal_{i}.jpg")
    mixed_img.save(output_path)

for i in range(len(male2)):
    img1_path = m_paths2[1]
    img1 = Image.open(img1_path)
    label1 = torch.tensor([1.0, 0.0])

    img2_path = f_paths2[1]
    img2 = Image.open(img2_path)
    label2 = torch.tensor([0.0, 1.0])

    mixed_img, mixed_label = cutmix(img1, label1, img2, label2)
    m_f.append(mixed_img)

    output_path = os.path.join(output_m_f, f"male_female_normal_{i}.jpg")
    mixed_img.save(output_path) 

'''
print(len(female0))
print(len(female1))
print(len(female2))
print(len(male0))
print(len(male1))
print(len(male2))
'''
img1_path = m_paths2[1]
img1 = Image.open(img1_path)
label1 = torch.tensor([1.0, 0.0])

img2_path = f_paths2[1]
img2 = Image.open(img2_path)
label2 = torch.tensor([0.0, 1.0])

mixed_img, mixed_label = cutmix(img1, label1, img2, label2)
# m_f.append(mixed_img)

output_path = os.path.join(output_m_f, f"male_female_normal.jpg")
mixed_img.save(output_path)