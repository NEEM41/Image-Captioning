import ast
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
import lightning as L


def parse_list_cell(x):
    """
    Parse a CSV cell that stores token ids as space-separated integers:
    e.g. "390 540 4427 329"
    """
    if x is None:
        return []

    if isinstance(x, list):
        return [int(t) for t in x]

    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []

    return [int(t) for t in s.split()]

def collate_caption_batch(batch, pad_id: int, max_len: Optional[int] = None):
    """
    Pads input_ids/labels to max length in batch (or max_len).
    Keeps clip_emb stacked as (B, D).
    """
    B = len(batch)
    clip = torch.stack([b["clip_emb"] for b in batch], dim=0)  # (B, D)

    lengths = [b["input_ids"].numel() for b in batch]

    T = max_len if max_len is not None else max(lengths)

    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), pad_id, dtype=torch.long)  # ignore padding in CE

    images = [b["image"] for b in batch]

    for i, b in enumerate(batch):
        x = b["input_ids"][:T]
        input_ids[i, : x.numel()] = x
        labels[i, : x.numel()] = x

    return {
        "image": images,
        "clip_emb": clip,
        "input_ids": input_ids,
        "labels": labels,
        "lengths": torch.tensor([min(l, T) for l in lengths], dtype=torch.long),
    }

class ClipPooledCaptionDataset(Dataset):
    """
    Loads:
      - caption token ids from a CSV
      - pooled CLIP embeddings from sharded safetensors via index.json

    Returns per item:
      {
        "image": <image filename>,
        "clip_emb": FloatTensor (D,),
        "input_ids": LongTensor (T,),
        "labels": LongTensor (T,)
      }

    Notes:
      - labels are standard next-token LM labels (shifted inside collate or model),
        BUT here we just set labels=input_ids and let your training step shift as you like.
      - shard cache keeps only a few shards in memory.
    """

    def __init__(
        self,
        split: str,
        csv_path: str | Path,
        embeddings_root: str | Path,
        device: Optional[str] = None,
        shard_cache_size: int = 2,
        image_col: str = "file_name",
        caption_ids_col: str = "caption_ids",
        shard_prefix: Optional[str] = None,
    ):
        assert split in {"train", "val"}, f"split must be 'train' or 'val', got: {split}"
        self.split = split
        self.csv_path = Path(csv_path)

        self.embeddings_root = Path(embeddings_root) / split  # .../pooled_clip_output/{split}
        self.index = json.loads((self.embeddings_root / "index.json").read_text())

        # Shard file name pattern differs if you used train/val in the shard name.
        # If you used: clip_pooled_train_shard_00000.safetensors
        # set shard_prefix="clip_pooled_train_shard_"
        # If val: shard_prefix="clip_pooled_val_shard_"
        
        # TODO: right now both got saved as train. Need to replace this not a big deal for now
        # shard_prefix = f"clip_pooled_train_shard_"
        shard_prefix = f"clip_lhs_train_shard_"

        # if shard_prefix is None:
        #     shard_prefix = f"clip_pooled_{split}_shard_"
        self.shard_prefix = shard_prefix

        self.df = pd.read_csv(self.csv_path)

        self.image_col = image_col
        self.caption_ids_col = caption_ids_col

        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.shard_cache_size = int(shard_cache_size)
        self._shard_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()

    def __len__(self):
        return len(self.df)

    def _get_shard_tensor(self, shard_id: int) -> torch.Tensor:
        # LRU cache
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            return self._shard_cache[shard_id]

        shard_path = self.embeddings_root / f"{self.shard_prefix}{shard_id:05d}.safetensors"
        emb = load_file(str(shard_path))["emb"]  # (N, D) fp16 on CPU
        emb = emb.to(self.device)  # move once

        self._shard_cache[shard_id] = emb
        if len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)  # evict LRU
        return emb

    def get_clip_pooled(self, image_filename: str) -> torch.Tensor:
        rec = self.index[image_filename]
        shard_id, row = rec["shard"], rec["row"]
        shard = self._get_shard_tensor(shard_id)
        emb = shard[row].float()  # (D,)

        # Optional: normalize (often helps)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return emb

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_name = str(row[self.image_col])

        caption_ids = parse_list_cell(row[self.caption_ids_col])
        input_ids = torch.tensor(caption_ids, dtype=torch.long)

        clip_emb = self.get_clip_pooled(image_name)

        return {
            "image": image_name,
            "clip_emb": clip_emb,       # (D,)
            "input_ids": input_ids,     # (T,)
            "labels": input_ids.clone() # (T,) (you can shift later)
        }


class ClipCaptionDataModule(L.LightningDataModule):
    """
    embeddings_root layout:
      embeddings_root/train/index.json + shards...
      embeddings_root/val/index.json + shards...
    """

    def __init__(
        self,
        train_csv: str | Path,
        val_csv: str | Path,
        embeddings_root: str | Path,
        pad_id: int,
        batch_size: int = 64,
        max_len: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        shuffle_train: bool = True,
        shard_cache_size: int = 2,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.embeddings_root = Path(embeddings_root)

        self.pad_id = int(pad_id)
        self.batch_size = int(batch_size)
        self.max_len = max_len
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.shuffle_train = bool(shuffle_train)
        self.shard_cache_size = int(shard_cache_size)

        self.dataset_kwargs = dataset_kwargs or {}

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None):
        self.train_ds = ClipPooledCaptionDataset(
            split="train",
            csv_path=self.train_csv,
            embeddings_root=self.embeddings_root,
            shard_cache_size=self.shard_cache_size,
            **self.dataset_kwargs,
        )

        self.val_ds = ClipPooledCaptionDataset(
            split="val",
            csv_path=self.val_csv,
            embeddings_root=self.embeddings_root,
            shard_cache_size=self.shard_cache_size,
            **self.dataset_kwargs,
        )

    def _collate_fn(self):
        return lambda batch: collate_caption_batch(
            batch,
            pad_id=self.pad_id,
            max_len=self.max_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self._collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self._collate_fn(),
        )

if __name__ == '__main__':
    from tqdm import tqdm 

    dm = ClipCaptionDataModule(
        train_csv="/Users/swornimchhetri/Desktop/all_codes/github_stuff/Image-Captioning/csvs/coco_train_tok.csv",
        val_csv="/Users/swornimchhetri/Desktop/all_codes/github_stuff/Image-Captioning/csvs/coco_val_tok.csv",
        embeddings_root="/Users/swornimchhetri/Desktop/all_codes/github_stuff/Image-Captioning/embeddings/token_embedding",
        pad_id=8192,          # set to your tokenizer pad id
        batch_size=1,
        max_len=69, # 69 Text + 8 prefix tokens
        num_workers=0,     # start with 0 on Mac
    )

    dm.setup()

    train_loader = dm.train_dataloader()

    # Check if all the dataset works as intended
    max_len = 0
    for batch in tqdm(train_loader):
        breakpoint()
        max_len = max(max_len, batch['input_ids'].shape[1])

    print(max_len) 

        