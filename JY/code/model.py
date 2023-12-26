import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseModel(nn.Module):
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


class ResNet16(nn.Module):
    def __init__(self, num_classes):
        super(ResNet16, self).__init__()
        self.resnet = models.resnet16(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        num_features = self.resnet.fc.in_features
        self.mask = nn.Linear(1000, 3, bias=False)
        self.age = nn.Linear(1000, 3, bias=False)
        self.gender = nn.Linear(1000, 2, bias=False)

    def forward(self, x):
        x = self.resnet(x)
        mask = self.mask(x)
        age = self.age(x)
        gender = self.gender(x)
        return mask, gender, age


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        num_features = self.resnet.fc.in_features
        self.mask = nn.Linear(1000, 3, bias=False)
        self.age = nn.Linear(1000, 3, bias=False)
        self.gender = nn.Linear(1000, 2, bias=False)

    def forward(self, x):
        x = self.resnet(x)
        mask = self.mask(x)
        age = self.age(x)
        gender = self.gender(x)
        return mask, gender, age


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.vggent = models.vggnet16(pretrained=True)
        num_features = self.vggent.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vggnet(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ConvNeXt, self).__init__()
        self.convnext = timm.create_model('convnext_tiny',pretrained=pretrained)

        for param in self.convnext.parameters():
            param.requires_grad=False

        # self.convnext.fc = nn.Linear(1000, num_classes)
        self.mask = nn.Linear(1000, 3, bias=False)
        self.gender = nn.Linear(1000, 2, bias=False)
        self.age = nn.Linear(1000, 3, bias=False)

    def forward(self, x):
        x = self.convnext(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age


class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        for param in self.vit.parameters():
            param.requires_grad=False

        # self.fc = nn.Linear(21843, num_classes)
        self.mask = nn.Linear(1000, 3, bias=False)
        self.gender = nn.Linear(1000, 2, bias=False)
        self.age = nn.Linear(1000, 3, bias=False)

    def forward(self, x):
        x = self.vit(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age


class Swin(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Swin, self).__init__()
        self.swin = timm.create_model('swin_small_patch4_window7_224',pretrained=pretrained)
        
        for param in self.swin.parameters():
            param.requires_grad=False

        self.mask = nn.Linear(1000, 3, bias=False)
        self.gender = nn.Linear(1000, 2, bias=False)
        self.age = nn.Linear(1000, 3, bias=False)

    def forward(self, x):
        x = self.swin(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age


class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        self.dense = timm.create_model('densenet169',pretrained=True)
        
        for param in self.dense.parameters():
            param.requires_grad=False

        self.mask = nn.Linear(1000, 3, bias=False)
        self.gender = nn.Linear(1000, 2, bias=False)
        self.age = nn.Linear(1000, 3, bias=False)

    def forward(self, x):
        x = self.dense(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.efficient = timm.create_model('efficientnet_b0',pretrained=True)
        
        for param in self.efficient.parameters():
            param.requires_grad=False

        self.mask = nn.Linear(1000, 3, bias=False)
        self.gender = nn.Linear(1000, 2, bias=False)
        self.age = nn.Linear(1000, 3, bias=False)

    def forward(self, x):
        x = self.efficient(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age