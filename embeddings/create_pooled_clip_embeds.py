import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModel
from safetensors.torch import save_file

IMAGE_DIR = Path("/Users/swornimchhetri/Desktop/all_codes/github_stuff/coco/val2014")
OUT_DIR = Path("/Users/swornimchhetri/Desktop/all_codes/github_stuff/Image-Captioning/embeddings/pooled_clip_output/val")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Settings
# -------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32           # 16–64 depending on MPS memory; 32 is usually safe
SHARD_SIZE = 8192         # embeddings per shard (8192 -> ~12.5MB per shard in fp16)
DTYPE_SAVE = torch.float16
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

def chunked(lst: List[Path], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def shard_path(shard_idx: int) -> Path:
    return OUT_DIR / f"clip_pooled_train_shard_{shard_idx:05d}.safetensors"

def shard_names_path(shard_idx: int) -> Path:
    return OUT_DIR / f"clip_pooled_train_shard_{shard_idx:05d}_names.json"

def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)

def main():
    print(f"Device: {DEVICE}")
    print(f"Image dir: {IMAGE_DIR}")
    print(f"Out dir:   {OUT_DIR}")

    # Load model + processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # Enumerate images
    image_paths = list_images(IMAGE_DIR)
    n = len(image_paths)
    if n == 0:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    num_shards = math.ceil(n / SHARD_SIZE)
    print(f"Found {n} images -> {num_shards} shards (shard_size={SHARD_SIZE})")

    # Main index mapping filename -> (shard,row)
    # Store only basename by default; if you have collisions, store relative path instead.
    index: Dict[str, Dict[str, int]] = {}

    # Metadata
    meta = {
        "model": MODEL_NAME,
        "device_used": DEVICE,
        "embedding_type": "pooler_output",
        "embedding_dim": None,   # filled after first batch
        "dtype": str(DTYPE_SAVE).replace("torch.", ""),
        "batch_size": BATCH_SIZE,
        "shard_size": SHARD_SIZE,
        "num_images": n,
        "image_dir": str(IMAGE_DIR),
    }

    # Process shard-by-shard to keep RAM predictable
    global_i = 0
    shard_idx = 0

    with torch.no_grad():
        while global_i < n:
            shard_start = global_i
            shard_end = min(n, shard_start + SHARD_SIZE)
            shard_imgs = image_paths[shard_start:shard_end]

            # Accumulate embeddings for this shard on CPU
            shard_emb_list: List[torch.Tensor] = []
            shard_names: List[str] = []

            for batch_paths in tqdm(list(chunked(shard_imgs, BATCH_SIZE)),
                                    desc=f"Shard {shard_idx:05d}",
                                    leave=False):
                images = []
                names = []

                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        images.append(img)
                        names.append(p.name)  # or str(p.relative_to(IMAGE_DIR))
                    except Exception as e:
                        print(f"Skipping {p} (load error): {e}")

                if not images:
                    continue

                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                outputs = model(**inputs)
                pooled = outputs.pooler_output  # (B, 768 for ViT-B/32)

                if meta["embedding_dim"] is None:
                    meta["embedding_dim"] = int(pooled.shape[-1])

                pooled_cpu = pooled.detach().to("cpu").to(DTYPE_SAVE)  # fp16 on disk
                shard_emb_list.append(pooled_cpu)
                shard_names.extend(names)

            if not shard_emb_list:
                raise RuntimeError(f"No embeddings produced for shard {shard_idx}")

            shard_emb = torch.cat(shard_emb_list, dim=0)  # (N_shard_actual, D)
            assert shard_emb.shape[0] == len(shard_names)

            # Write shard safetensors
            spath = shard_path(shard_idx)
            save_file({"emb": shard_emb}, str(spath))

            # Write shard names (row order)
            npath = shard_names_path(shard_idx)
            atomic_write_json(npath, shard_names)

            # Update global index
            for row, name in enumerate(shard_names):
                if name in index:
                    # If you might have duplicate basenames, switch to relative paths instead of name.
                    raise RuntimeError(f"Duplicate filename key in index: {name}")
                index[name] = {"shard": shard_idx, "row": row}

            # advance
            global_i = shard_end
            shard_idx += 1

    # Write top-level index + meta
    atomic_write_json(OUT_DIR / "index.json", index)
    atomic_write_json(OUT_DIR / "meta.json", meta)

    print("Done.")
    print(f"Wrote shards to: {OUT_DIR}")
    print(f"Index: {OUT_DIR / 'index.json'}")
    print(f"Meta:  {OUT_DIR / 'meta.json'}")

if __name__ == "__main__":
    main()