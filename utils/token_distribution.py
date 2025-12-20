import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# load csv
df = pd.read_csv("./../csvs/coco_train_tok.csv")

# count tokens
token_counts = Counter()
for s in tqdm(df["caption_ids"].astype(str), desc="Counting tokens"):
    token_counts.update(map(int, s.split()))

vocab_size = 8192  # or whatever you trained
used = len(token_counts)
print("Used tokens:", used)
print("Usage ratio:", used / vocab_size)

# dataframe sorted by token id
df_tokens = (
    pd.DataFrame(token_counts.items(), columns=["token_id", "frequency"])
    .sort_values("token_id")
)

# plot
plt.figure(figsize=(12, 5))
plt.bar(
    df_tokens["token_id"],
    df_tokens["frequency"],
    width=1.0
)

plt.yscale("log")  # absolutely necessary
plt.xlabel("Token ID")
plt.ylabel("Frequency (log scale)")
plt.title("Token ID vs Frequency")

plt.tight_layout()
plt.savefig("./../analysis/token_id_vs_frequency_bar.png", dpi=200)
plt.close()