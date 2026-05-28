import json
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .generate import generate, generate_stream

app = FastAPI(title="Advanced LLM Server")

_model = None
_tokenizer = None
_device = "cuda"


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int


def load_model(model_path: str, device: str = "cuda"):
    global _model, _tokenizer, _device
    _device = device

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.config import ModelConfig
    from model.transformer import HybridModel

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        config = ModelConfig.from_dict(json.loads(config_path.read_text()))
    else:
        config = ModelConfig()

    _model = HybridModel(config)

    weights_path = Path(model_path) / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path))
    else:
        weights_path = Path(model_path) / "model.pt"
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    _model.load_state_dict(state_dict)
    _model = _model.to(device).eval()


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    if _model is None:
        return GenerateResponse(text="Model not loaded", tokens_generated=0)

    input_ids = torch.tensor(
        [_tokenizer.encode(req.prompt).ids], device=_device
    ) if _tokenizer else torch.tensor([[1]], device=_device)

    if req.stream:
        return StreamingResponse(
            _stream_generate(input_ids, req),
            media_type="text/event-stream",
        )

    output_ids = generate(
        _model, input_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    text = _decode(output_ids[0])
    return GenerateResponse(text=text, tokens_generated=new_tokens)


async def _stream_generate(input_ids, req):
    for token in generate_stream(
        _model, input_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    ):
        text = _decode(token[0])
        yield f"data: {json.dumps({'token': text})}\n\n"
    yield "data: [DONE]\n\n"


def _decode(token_ids):
    if _tokenizer:
        return _tokenizer.decode(token_ids.tolist())
    return str(token_ids.tolist())


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}
