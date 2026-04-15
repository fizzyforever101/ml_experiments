import torch
import torchvision.models as models

def get_model():
    model = models.resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model