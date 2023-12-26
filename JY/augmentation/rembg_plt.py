import matplotlib.pyplot as plt
import numpy as np
import torch
from rembg import remove
from PIL import Image
from torchvision.transforms import (Resize,ToTensor,Normalize,Compose,CenterCrop,ColorJitter,)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
            ]
        )

    def __call__(self, image):
        image_before_remove = image.copy()  # 원본 이미지 복사
        image_data = np.array(image_before_remove)
        image_data = remove(image_data)  # remove 함수에 이미지 데이터 전달
        image_data = image_data[:, :, :3]
        image = Image.fromarray(image_data)

        image_after_transform = self.transform(image)

        image_after_transform_np = np.transpose(image_after_transform.squeeze(0).numpy(), (1, 2, 0))

        # 이미지 시각화
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(image_before_remove)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Image after Remove")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Image after Transform")
        plt.imshow(image_after_transform_np)  # 투명 영역 시각화
        plt.axis('off')

        plt.ion()
        plt.show()

        return image_after_transform


image_path = "/data/ephemeral/home/jero/train/images/000001_female_Asian_45/incorrect_mask.jpg"  # 실제 파일 경로로 변경하세요.

# 이미지 로드
original_image = Image.open(image_path)

# BaseAugmentation 클래스 초기화
base_augmentation = BaseAugmentation(resize=(128, 96), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

# BaseAugmentation 적용 및 시각화
transformed_image = base_augmentation(original_image)
transformed_np = np.transpose(transformed_image.squeeze(0).numpy(), (1, 2, 0))

# 원본 이미지 시각화
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis('off')

# 변환된 이미지 시각화
plt.subplot(1, 2, 2)
plt.title("Transformed Image")
plt.imshow(transformed_np)
plt.axis('off')

plt.show()