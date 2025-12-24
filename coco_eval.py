from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

annotation_file = './../coco/annotations/captions_val2014.json'
results_file = './prediction.json'

coco = COCO(annotation_file)
coco_res = coco.loadRes(results_file)

img_ids = coco_res.getImgIds()

# Build STRICT string-only dicts for ALL metrics (avoids the dict/split crash)
gts = {}
res = {}

for img_id in img_ids:
    # references (list[str])
    gts_caps = [ann.get("caption", "") for ann in coco.imgToAnns.get(img_id, [])]
    gts_caps = [c for c in gts_caps if isinstance(c, str) and c.strip()]
    gts[img_id] = gts_caps

    # predictions (list[str]) - keep first if multiple
    res_caps = [ann.get("caption", "") for ann in coco_res.imgToAnns.get(img_id, [])]
    res_caps = [c for c in res_caps if isinstance(c, str) and c.strip()]
    res[img_id] = [res_caps[0]] if len(res_caps) > 0 else [""]

# keep only ids that have at least 1 ref and a non-empty pred
valid_ids = [i for i in img_ids if len(gts.get(i, [])) > 0 and isinstance(res.get(i, [""])[0], str) and res[i][0].strip()]
gts = {i: gts[i] for i in valid_ids}
res = {i: res[i] for i in valid_ids}

print(f"Evaluating on {len(valid_ids)} images")

# --- CIDEr ---
cider = Cider()
cider_score, _ = cider.compute_score(gts, res)
cider_norm = cider_score
print(f"CIDEr: {cider_norm:.3f}")

# --- BLEU ---
bleu = Bleu(4)
bleu_scores, _ = bleu.compute_score(gts, res)
print(f"Bleu_4: {bleu_scores[3]:.3f}")
print(f"Bleu_3: {bleu_scores[2]:.3f}")
print(f"Bleu_2: {bleu_scores[1]:.3f}")
print(f"Bleu_1: {bleu_scores[0]:.3f}")

# --- ROUGE_L ---
rouge = Rouge()
rouge_score, _ = rouge.compute_score(gts, res)
print(f"ROUGE_L: {rouge_score:.3f}")