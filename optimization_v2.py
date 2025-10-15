import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.quantization import quantize_dynamic
import copy

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

calib_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=32, shuffle=True)

#Compute output variance per layer
layer_sensitivities = {}

def hook_fn(module, input, output):
    layer_sensitivities[module] = output.var().item()

hooks = []
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        hooks.append(layer.register_forward_hook(hook_fn))

with torch.no_grad():
    for i, (images, _) in enumerate(calib_loader):
        _ = model(images)
        if i >= 5:  # only a few batches
            break
for h in hooks:
    h.remove()

#layers with variance < threshold -> quantize
variance_threshold = 0.01
layers_to_keep_fp = [layer for layer, var in layer_sensitivities.items() if var > variance_threshold]

print("Keeping high-variance layers in FP32:", len(layers_to_keep_fp))

quantized_model = copy.deepcopy(model)
for name, layer in quantized_model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if layer not in layers_to_keep_fp:
            # Dynamic quantization for this layer
            quantized_layer = torch.quantization.quantize_dynamic(
                layer, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            setattr(quantized_model, name.split('.')[-1], quantized_layer)

device = torch.device("cpu")
quantized_model.to(device)

dummy_input = torch.randn(1, 3, 224, 224, device=device)
with torch.no_grad():
    import time
    start = time.time()
    output = quantized_model(dummy_input)
    print("Quantized inference latency:", (time.time() - start) * 1000, "ms")

torch.save(quantized_model.state_dict(), "resnet18_mixed_precision.pth")
