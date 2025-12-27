# inference.py
import argparse
from pathlib import Path
import yaml

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel

from model.model import ClipLM, LMConfig, ClipLMLightning
from bpe.bpe import RegexTokenizer


def safe_yaml_check(cfg: dict):
    required_top = ["run", "model_args", "data_args", "optim_args", "train_args"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"Missing top-level key: '{k}'")
    print("✓ Config sanity check passed")


def build_model(cfg: dict) -> ClipLMLightning:
    model_args = cfg["model_args"]
    optim_args = cfg["optim_args"]
    train_args = cfg["train_args"]

    base_model = ClipLM(LMConfig(**model_args))

    lightning_module = ClipLMLightning(
        model=base_model,
        lr=float(optim_args.get("lr", 3e-4)),
        min_lr=float(optim_args.get("min_lr", 0.0)),
        weight_decay=float(optim_args.get("weight_decay", 0.0)),
        betas=tuple(optim_args.get("betas", (0.9, 0.95))),
        warmup_steps=int(optim_args.get("warmup_steps", 0)),
        grad_clip_val=float(train_args.get("gradient_clip_val", 0.0)),
        ignore_index=int(cfg["model_args"].get("pad_token", -100)),
    )
    return lightning_module


def load_ckpt_into_lightning(lit_model: ClipLMLightning, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"WARNING unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"Loaded checkpoint: {ckpt_path}")


def load_regex_tokenizer(model_path: str) -> RegexTokenizer:
    tok = RegexTokenizer()
    tok.load(model_path)
    return tok


def strip_specials(text: str) -> str:
    return (
        text.replace("<bos>", "")
            .replace("<eos>", "")
            .replace("<pad>", "")
            .strip()
    )


def decode_caption(tok: RegexTokenizer, token_ids: list[int]) -> str:
    try:
        return tok.decode([int(x) for x in token_ids])
    except Exception:
        return " ".join(map(str, token_ids))


def pick_device(requested: str) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def clip_pooled_embedding(
    image: Image.Image,
    clip_model_name: str,
    device: str,
) -> torch.Tensor:
    """
    Match your preprocessing script:
      processor = AutoProcessor.from_pretrained(MODEL_NAME)
      model = CLIPVisionModel.from_pretrained(MODEL_NAME)
      pooled = outputs.pooler_output   # (1, 768) for ViT-B/32
    """
    processor = AutoProcessor.from_pretrained(clip_model_name)
    vision = CLIPVisionModel.from_pretrained(clip_model_name).to(device).eval()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = vision(**inputs)
    pooled = outputs.pooler_output  # (1, 768)
    return pooled.float()


@torch.no_grad()
def run_inference(
    cfg: dict,
    ckpt_path: str,
    tokenizer_model: str,
    image_path: str,
    device: str,
    clip_model_name: str,
    method: str,
    top_k: int,
    temperature: float,
    max_new_tokens: int,
) -> str:
    # tokenizer
    tok = load_regex_tokenizer(tokenizer_model)
    bos_id = tok.encode("<bos>", allowed_special="all")[0]
    eos_id = tok.encode("<eos>", allowed_special="all")[0]

    # caption model
    lit_model = build_model(cfg)
    load_ckpt_into_lightning(lit_model, ckpt_path)
    lit_model.eval().to(device)
    base_model: ClipLM = lit_model.model

    # image -> pooled CLIP embedding
    img = Image.open(image_path).convert("RGB")
    clip_emb = clip_pooled_embedding(img, clip_model_name=clip_model_name, device=device)

    # warn if cfg expects a different dim
    expected = cfg["model_args"].get("clip_emb_dim", None)
    if expected is not None and int(clip_emb.shape[-1]) != int(expected):
        print(f"WARNING: pooled clip_emb dim {clip_emb.shape[-1]} != cfg.model_args.clip_emb_dim {expected}")
        print("Your pooled output for openai/clip-vit-base-patch32 should be 768-dim.")

    # generate
    prompt = torch.full((1, 1), int(bos_id), dtype=torch.long, device=device)
    out_ids = base_model.generate(
        clip_emb=clip_emb,
        input_ids=prompt,
        max_new_tokens=max_new_tokens,
        method=method,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=int(eos_id),
    ).tolist()[0]

    # decode
    seq = out_ids
    if len(seq) > 0 and seq[0] == bos_id:
        seq = seq[1:]
    if eos_id in seq:
        seq = seq[: seq.index(eos_id)]

    pred = decode_caption(tok, [bos_id] + seq + [eos_id])
    return strip_specials(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tokenizer_model", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    ap.add_argument("--method", type=str, default="greedy", choices=["greedy", "topk"])
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=40)

    args = ap.parse_args()
    device = pick_device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    safe_yaml_check(cfg)

    caption = run_inference(
        cfg=cfg,
        ckpt_path=args.ckpt,
        tokenizer_model=args.tokenizer_model,
        image_path=args.image,
        device=device,
        clip_model_name=args.clip_model_name,
        method=args.method,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"[{Path(args.image).name}] {caption}")


if __name__ == "__main__":
    main()