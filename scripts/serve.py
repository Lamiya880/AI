import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Advanced LLM Inference Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from inference.server import app, load_model

    print(f"Loading model from {args.model_path}...")
    load_model(args.model_path, device=args.device)
    print(f"Starting server on {args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
