# eval.py
import argparse
import csv
from pathlib import Path
import yaml

import torch
from tqdm import tqdm
import sacrebleu

from model.model import ClipLM, LMConfig, ClipLMLightning
from dataset.dataset import ClipCaptionDataModule
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
        ignore_index=int(model_args.get("pad_token", -100)),
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
    # remove training wrappers (and as extra safety, remove <pad> if it appears)
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


def get_file_names_from_batch(batch: dict):
    # Add your real key here if needed
    for k in ("file_name", "image", "image_name", "image_id", "filename"):
        if k in batch:
            return [str(x) for x in batch[k]]
    return None


def load_done_set(out_csv: Path) -> set[str]:
    done = set()
    if not out_csv.exists():
        return done
    try:
        with open(out_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "file_name" not in reader.fieldnames:
                return done
            for row in reader:
                fn = row.get("file_name", "")
                if fn:
                    done.add(fn)
    except Exception as e:
        print(f"WARNING: Could not read existing out_csv for resume ({e}). Resume disabled.")
        return set()
    return done


@torch.no_grad()
def run_eval(
    cfg: dict,
    ckpt_path: str,
    out_csv: str,
    tokenizer_model: str,
    split: str = "val",
    device: str = "mps",
    method: str = "greedy",
    top_k: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 40,
    resume: bool = True,
):
    tok = load_regex_tokenizer(tokenizer_model)
    bos_id = tok.encode("<bos>", allowed_special="all")[0]
    eos_id = tok.encode("<eos>", allowed_special="all")[0]
    pad_id = int(cfg["data_args"]["pad_id"])

    dm = ClipCaptionDataModule(**cfg["data_args"])
    dm.setup()
    loader = dm.val_dataloader() if split == "val" else dm.train_dataloader()

    lit_model = build_model(cfg)
    load_ckpt_into_lightning(lit_model, ckpt_path)
    lit_model.eval()
    lit_model.to(device)
    base_model: ClipLM = lit_model.model

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()

    done_files = set()
    if resume:
        done_files = load_done_set(out_path)
        if done_files:
            print(f"Resuming: {len(done_files)} rows already in {out_path}")

    f = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=["file_name", "original_text", "predicted_text", "bleu"]
    )
    if write_header:
        writer.writeheader()
        f.flush()

    pbar = tqdm(loader, desc=f"Eval ({split})", unit="batch")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)   # (B,T) padded
        clip_emb  = batch["clip_emb"].to(device).float()

        file_names = get_file_names_from_batch(batch)
        if file_names is None:
            file_names = [""] * input_ids.size(0)

        # resume-skip if filenames exist
        if resume and any(file_names):
            keep_mask = [fn not in done_files for fn in file_names]
            if not any(keep_mask):
                continue
            keep_idx = torch.tensor([i for i, k in enumerate(keep_mask) if k], device=device, dtype=torch.long)
            input_ids = input_ids.index_select(0, keep_idx)
            clip_emb  = clip_emb.index_select(0, keep_idx)
            file_names = [fn for fn, k in zip(file_names, keep_mask) if k]

        # ---- decode GT (strip pads BEFORE decode; optionally stop at eos) ----
        gt_texts = []
        for seq in input_ids.tolist():
            seq = [int(x) for x in seq if int(x) != pad_id]  # remove <pad>
            if eos_id in seq:
                seq = seq[: seq.index(eos_id) + 1]           # keep through <eos>
            gt = decode_caption(tok, seq)
            gt_texts.append(strip_specials(gt))

        # ---- generate ----
        prompt = torch.full((clip_emb.size(0), 1), int(bos_id), dtype=torch.long, device=device)

        out_ids = base_model.generate(
            clip_emb=clip_emb,
            input_ids=prompt,
            max_new_tokens=max_new_tokens,
            method=method,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=int(eos_id),
        ).tolist()

        pred_texts = []
        for seq in out_ids:
            if len(seq) > 0 and seq[0] == bos_id:
                seq = seq[1:]
            if eos_id in seq:
                seq = seq[: seq.index(eos_id)]
            # wrap for decode consistency
            pred = decode_caption(tok, [bos_id] + seq + [eos_id])
            pred_texts.append(strip_specials(pred))

        # ---- write rows with sentence BLEU ----
        for fn, gt, pr in zip(file_names, gt_texts, pred_texts):
            bleu = sacrebleu.sentence_bleu(pr, [gt]).score  # 0..100
            writer.writerow(
                {
                    "file_name": fn,
                    "original_text": gt,
                    "predicted_text": pr,
                    "bleu": float(bleu),
                }
            )
            if resume and fn:
                done_files.add(fn)

        f.flush()

    f.close()
    print(f"Saved predictions to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--tokenizer_model", type=str, required=True)

    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--device", type=str, default="mps")

    ap.add_argument("--method", type=str, default="greedy", choices=["greedy", "topk"])
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--no_resume", action="store_true")

    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    safe_yaml_check(cfg)

    run_eval(
        cfg=cfg,
        ckpt_path=args.ckpt,
        out_csv=args.out_csv,
        tokenizer_model=args.tokenizer_model,
        split=args.split,
        device=args.device,
        method=args.method,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        resume=(not args.no_resume),
    )


if __name__ == "__main__":
    main()