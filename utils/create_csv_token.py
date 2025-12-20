import pandas as pd
from tqdm import tqdm
from bpe.bpe import RegexTokenizer
import pandas as pd
import re

tok = RegexTokenizer()
tok.load("./../bpe/coco_8k_150k_subset.model")

tqdm.pandas()
df = pd.read_csv("./../csvs/coco_val.csv")

def clean_caption(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r", " ").replace("\n", " ")   # remove embedded newlines
    s = re.sub(r"\s+", " ", s).strip()           # collapse whitespace
    return s

def enc(s: str) -> str:
    s = clean_caption(s)
    text = f"<bos>{s}<eos>"
    # print(tok.special_tokens)
    ids = tok.encode(text, allowed_special='all')  # or add <bos>/<eos> here if you want
    return " ".join(map(str, ids))

df["caption_ids"] = df["caption"].astype(str).progress_apply(enc)
df["caption_len"] = df["caption_ids"].str.count(" ") + 1

df.to_csv("./../csvs/coco_val_tok.csv", index=False)