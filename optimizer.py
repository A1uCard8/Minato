import torch
from torch import nn
from torchvision import models
from torch.quantization import quantize_dynamic

#Quantizes a PyTorch model for faster inference and smaller size.
def optimize_model(model: nn.Module, example_input, dtype=torch.qint8):
    model.eval()
    quantized_model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=dtype)
    return quantized_model

if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    optimized = optimize_model(model, None)
    torch.save(optimized.state_dict(), "resnet18_optimized.pth")
