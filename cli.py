import argparse
from edgeopt.optimizer import optimize_model
from inference.engine import InferenceEngine
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", type=str, help="Path to model")
    parser.add_argument("--run", type=str, help="Run model inference on optimized model")
    args = parser.parse_args()

    if args.optimize:
        model = torch.load(args.optimize)
        optimized = optimize_model(model, None)
        torch.save(optimized.state_dict(), "optimized_model.pth")
        print("✅ Model optimized and saved.")

    if args.run:
        engine = InferenceEngine(args.run)
        dummy_input = torch.randn(1,3,224,224)
        engine.run(dummy_input)

if __name__ == "__main__":
    main()
