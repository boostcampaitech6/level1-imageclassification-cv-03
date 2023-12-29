import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)
    
class Convnext_tiny(nn.Module):
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