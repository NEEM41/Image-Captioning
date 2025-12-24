import json
import pandas as pd

# ---- inputs ----
pred_csv = "./output.csv"          # file_name,original_text,predicted_text,bleu
map_csv  = "./csvs/coco_val.csv"         # image_id,file_name,caption   (caption column not needed)
out_json = "./prediction.json"        # output: [{"image_id":..., "caption":...}, ...]

# ---- load ----
pred_df = pd.read_csv(pred_csv)
map_df  = pd.read_csv(map_csv)

# keep only what we need
pred_df = pred_df[["file_name", "predicted_text"]].copy()
map_df  = map_df[["image_id", "file_name"]].copy()

# normalize types / whitespace (helps avoid silent mismatches)
pred_df["file_name"] = pred_df["file_name"].astype(str).str.strip()
map_df["file_name"]  = map_df["file_name"].astype(str).str.strip()

# ---- join on file_name -> attach image_id ----
merged = pred_df.merge(map_df, on="file_name", how="left")

# check missing matches
missing = merged[merged["image_id"].isna()]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} rows had file_name not found in mapping CSV.")
    print("Examples:", missing["file_name"].head(10).tolist())

# drop unmatched
merged = merged.dropna(subset=["image_id"]).copy()
merged["image_id"] = merged["image_id"].astype(int)

# optional: if you have duplicates per image, keep the first
merged = merged.drop_duplicates(subset=["image_id"], keep="first")

# ---- build COCO results list ----
results = [
    {"image_id": int(row.image_id), "caption": str(row.predicted_text)}
    for row in merged.itertuples(index=False)
]

# ---- save JSON ----
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)

print(f"Saved {len(results)} predictions to {out_json}")