from fastapi import FastAPI, UploadFile, File
import torch
import time
import os
from torchvision import models
import psutil

app = FastAPI()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_model(file_path):
    # For demo, allow only ResNet18
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024*1024)

@app.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"status": "uploaded", "file_path": file_path}

@app.get("/benchmark")
def benchmark_model():
    model = load_model(None)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    start_mem = get_memory_mb()
    start_time = time.time()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    latency_ms = (time.time() - start_time) * 1000
    memory_mb = get_memory_mb() - start_mem
    
    return {
        "model_name": "ResNet18",
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "output_shape": list(output.shape)
    }
