import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_b5,
    EfficientNet_B5_Weights,
    resnet18,
    ResNet18_Weights,
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_classes 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
                        
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(512*4*3, num_classes),
            nn.Softmax()
        )
        

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return self.layer(x)



def EfficientNetB5(num_classes: int):
    # Best available weights (currently alias for IMAGENET1K_V2)
    # Note that these weights may change across versions
    model = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes),
        nn.Softmax()
    )

    for param in model.parameters(): # model의 모든 parameter 를 freeze
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

def Resnet18(num_classes):
    model = resnet18(ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
        nn.Softmax()
    )

    for param in model.parameters(): # model의 모든 parameter 를 freeze
        param.requires_grad = True

    return model


class MultiLabelResNet18(nn.Module):
    def __init__(self, num_classes) -> None:
        super(MultiLabelResNet18, self).__init__()
        resnet18_mask = torch.hub.load('pytorch/vision:v0.11.1', 'resnet18', pretrained=True)
        resnet18_mask.fc = nn.Linear(resnet18_mask.fc.in_features, 3)
        self.resnet_mask = self.init_weights(resnet18_mask)

        resnet18_gender = torch.hub.load('pytorch/vision:v0.11.1', 'resnet18', pretrained=True)
        resnet18_gender.fc = nn.Linear(resnet18_gender.fc.in_features, 2)
        self.resnet_gender = self.init_weights(resnet18_gender)

        resnet18_age = torch.hub.load('pytorch/vision:v0.11.1', 'resnet18', pretrained=True)
        resnet18_age.fc = nn.Linear(resnet18_age.fc.in_features, 3)
        self.resnet_age = self.init_weights(resnet18_age)

    def forward(self, x):
        output_mask = self.resnet_mask(x)
        output_gender = self.resnet_gender(x)
        output_age = self.resnet_age(x)
        output = output_mask[1]*6 + output_gender[1]*3 + output_age[1]
        
        return (output_mask, output_gender, output_age, output)
    
    def init_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True

        return model
    
class Resnet50(nn.Module):
    # Best available weights (currently alias for IMAGENET1K_V2)
    # Note that these weights may change across versions
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU()
        )
        
        self.mask_fc = nn.Linear(512, 3)
        self.gender_fc = nn.Linear(512, 2)
        self.age_fc = nn.Linear(512, 3)

        self.model = self.init_weights(self.model)

    def init_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True

        return model

    def forward(self, x):
        out = self.model(x)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age
    
    
class Convnext_tiny(nn.Module):
    # Best available weights (currently alias for IMAGENET1K_V2)
    # Note that these weights may change across versions
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.model.classifier.Linear = nn.Linear(1000, 64, bias=True)
        
        self.mask_fc = nn.Linear(64, 3)
        self.gender_fc = nn.Linear(64, 2)
        self.age_fc = nn.Linear(64, 3)

        self.model = self.init_weights(self.model)

    def init_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True

        return model

    def forward(self, x):
        out = self.model(x)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age
    
class Vit_tiny(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = timm.models.vit_tiny_r_s16_p8_224(pretrained=True)
        self.mask_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 3)
        )
        self.gender_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 2)
        )
        self.age_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 3)
        )

        self.model = self.init_weights(self.model)
    
    def init_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True

        return model

    def forward(self, x):
        out = self.model(x)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age
    
class Swin_tiny(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = timm.models.create_model("swin_s3_tiny_224", pretrained=True, num_classes=64)
        self.mask_fc = nn.Linear(64, 3)
        self.gender_fc = nn.Linear(64, 2)
        self.age_fc = nn.Linear(64, 3)

        self.model = self.init_weights(self.model)
    
    def init_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True

        return model

    def forward(self, x):
        out = self.model(x)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age    
    
class Swinv2_tiny(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model=timm.create_model("swinv2_cr_tiny_224", pretrained=False, num_classes=64)
        self.mask_fc = nn.Linear(64, 3)
        self.gender_fc = nn.Linear(64, 2)
        self.age_fc = nn.Linear(64, 3)

        self.model = self.init_weights(self.model)
    
    def init_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True

        return model

    def forward(self, x):
        out = self.model(x)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age    
    
class Resnext50(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = resnext50_32x4d(pretrained=True)
        self.fc_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 64),
            nn.LeakyReLU(0.2)
        )
        self.mask_fc = nn.Linear(64, 3)
        self.gender_fc = nn.Linear(64, 2)
        self.age_fc = nn.Linear(64, 3)

        self.model = self.init_weights(self.model)
    
    def init_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True

        return model

    def forward(self, x):
        out = self.model(x)
        out = self.fc_layer(out)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age
    

# 선우형 모델
class Swin3Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.swin = timm.create_model("swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",pretrained=True,num_classes=64)
        for param in self.swin.parameters():
            param.requires_grad=False
        for param in self.swin.layers[3].parameters():
            param.requires_grad=True
        for param in self.swin.head.parameters():
            param.requires_grad=True

        self.mask_fc = nn.Linear(64, 3)
        self.gender_fc = nn.Linear(64, 2)
        self.age_fc = nn.Linear(64, 3)      

    def forward(self, x):
        out = self.swin(x)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age    
    
class EvaModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eva = timm.create_model("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",pretrained=True,num_classes=64)
        for param in self.eva.parameters():
            param.requires_grad=False
        for param in self.eva.head.parameters():
            param.requires_grad=True

        self.mask_fc = nn.Linear(64, 3)
        self.gender_fc = nn.Linear(64, 2)
        self.age_fc = nn.Linear(64, 3)

    def forward(self, x):
        out = self.eva(x)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age    

class Conv2Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = timm.create_model("convnext_xxlarge.clip_laion2b_soup_ft_in1k",pretrained=True,num_classes=64)
        for param in self.conv.parameters():
            param.requires_grad=False
        for param in self.conv.stages[3].blocks[2].parameters():
            param.requires_grad=True
        for param in self.conv.head.parameters():
            param.requires_grad=True
            
        self.mask_fc = nn.Linear(64, 3)
        self.gender_fc = nn.Linear(64, 2)
        self.age_fc = nn.Linear(64, 3)

    def forward(self, x):
        out = self.conv(x)
        mask = self.mask_fc(out)
        gender = self.gender_fc(out)
        age = self.age_fc(out)
        return mask, gender, age    
