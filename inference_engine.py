import torch
from torchvision import models
import time

class InferenceEngine:
    def __init__(self, model_path=None):
        if model_path:
            self.model = models.resnet18()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model = models.resnet18(pretrained=True)
        self.model.eval()

    def run(self, input_tensor):
        start = time.time()
        with torch.no_grad():
            output = self.model(input_tensor)
        latency = (time.time() - start) * 1000
        print(f"Latency: {latency:.2f} ms")
        return output
