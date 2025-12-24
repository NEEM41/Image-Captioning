import argparse
from pathlib import Path
import yaml

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from model.model import ClipLM, LMConfig, ClipLMLightning
from dataset.dataset import ClipCaptionDataModule
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
from datetime import datetime

from pathlib import Path
from lightning.pytorch.callbacks import Callback

class CheckpointIntoMLflowArtifacts(Callback):
    def on_fit_start(self, trainer, pl_module):
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            raise RuntimeError("MLflow logger is required to place checkpoints in MLflow artifacts.")

        tracking_uri = getattr(logger, "_tracking_uri", None) or getattr(logger, "tracking_uri", None)
        tracking_uri = str(tracking_uri) if tracking_uri is not None else ""

        if not tracking_uri.startswith("file:"):
            raise RuntimeError(
                "This checkpoint-into-artifacts approach requires MLflow file store (tracking_uri='file:...')."
            )

        base = Path(tracking_uri.replace("file:", ""))  # .../output_dir/mlruns
        exp_id = str(logger.experiment_id)
        run_id = str(logger.run_id)

        artifacts_ckpt_dir = base / exp_id / run_id / "artifacts" / "checkpoints"
        artifacts_ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb is None:
            raise RuntimeError("No checkpoint callback found on trainer (enable checkpointing / add ModelCheckpoint).")

        ckpt_cb.dirpath = str(artifacts_ckpt_dir)
        print(f"Checkpoint dir set to MLflow artifacts: {ckpt_cb.dirpath}")

def safe_yaml_check(cfg: dict):
    required_top = ["run", "model_args", "data_args", "optim_args", "train_args"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"Missing top-level key: '{k}'")

    model = cfg["model_args"]
    data = cfg["data_args"]
    optim = cfg["optim_args"]
    train = cfg["train_args"]

    # --- cast optimization values ---
    optim["lr"] = float(optim["lr"])
    optim["weight_decay"] = float(optim["weight_decay"])
    optim["betas"] = tuple(float(b) for b in optim["betas"])

    if "min_lr" in optim:
        optim["min_lr"] = float(optim["min_lr"])

    if "warmup_steps" in optim:
        optim["warmup_steps"] = int(optim["warmup_steps"])

    # --- cast training values ---
    train["max_steps"] = int(train["max_steps"])
    train["val_check_interval"] = int(train["val_check_interval"])
    train["log_every_n_steps"] = int(train["log_every_n_steps"])

    if "gradient_clip_val" in train:
        train["gradient_clip_val"] = float(train["gradient_clip_val"])

    if "deterministic" in train:
        train["deterministic"] = bool(train["deterministic"])

    # --- model sanity ---
    for k in ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]:
        if not isinstance(model[k], int) or model[k] <= 0:
            raise ValueError(f"model_args.{k} must be a positive int")

    if model["n_embd"] % model["n_head"] != 0:
        raise ValueError("model_args.n_embd must be divisible by model_args.n_head")

    if model.get("clip_dim", 0) <= 0:
        raise ValueError("model_args.clip_dim must be > 0")

    if model.get("prefix_len", 0) < 0:
        raise ValueError("model_args.prefix_len must be >= 0")

    if model.get("pad_token", 0) >= model["vocab_size"]:
        raise ValueError("model_args.pad_token must be < vocab_size")

    # --- data sanity ---
    for p in ["train_csv", "val_csv", "embeddings_root"]:
        if not Path(data[p]).exists():
            raise ValueError(f"Path does not exist: {data[p]}")

    if data["pad_id"] >= model["vocab_size"]:
        raise ValueError("data_args.pad_id must be < vocab_size")

    if data["max_len"] <= 0:
        raise ValueError("data_args.max_len must be > 0")

    if data["batch_size"] <= 0:
        raise ValueError("data_args.batch_size must be > 0")

    # --- optim sanity ---
    if optim["lr"] <= 0:
        raise ValueError("optim_args.lr must be > 0")

    if optim["weight_decay"] < 0:
        raise ValueError("optim_args.weight_decay must be >= 0")

    if not isinstance(optim["betas"], tuple) or len(optim["betas"]) != 2:
        raise ValueError("optim_args.betas must be length 2")

    # --- train sanity ---
    if train["max_steps"] <= 0:
        raise ValueError("train_args.max_steps must be > 0")

    if train["log_every_n_steps"] <= 0:
        raise ValueError("train_args.log_every_n_steps must be > 0")

    print("✓ Config sanity check passed")


def build_model(cfg: dict) -> ClipLMLightning:
    model_args = cfg["model_args"]
    optim_args = cfg["optim_args"]
    train_args = cfg["train_args"]

    base_model = ClipLM(LMConfig(**model_args))

    lightning_module = ClipLMLightning(
        model=base_model,
        lr=float(optim_args["lr"]),
        min_lr=float(optim_args.get("min_lr", 0.0)),
        weight_decay=float(optim_args["weight_decay"]),
        betas=tuple(optim_args["betas"]),
        warmup_steps=int(optim_args.get("warmup_steps", 0)),
        grad_clip_val=float(train_args.get("gradient_clip_val", 0.0)),
        ignore_index=int(model_args['pad_token']),
    )
    return lightning_module


def build_datamodule(cfg: dict) -> ClipCaptionDataModule:
    return ClipCaptionDataModule(**cfg["data_args"])

def build_logger(cfg: dict):
    out = Path(cfg["run"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"file:{out / 'mlruns'}"

    return MLFlowLogger(
        experiment_name=cfg["run"].get("mlflow_experiment", cfg["run"]["run_name"]),
        run_name=cfg["run"]["run_name"],
        tracking_uri=tracking_uri,
    )

def build_callbacks(cfg: dict):
    train = cfg["train_args"]
    n = int(train["val_check_interval"])

    checkpoint_cb = ModelCheckpoint(
        dirpath='./placeholder/',
        filename="step{step:07d}",
        monitor="step",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=n,
        save_on_train_epoch_end=False,
        every_n_epochs=0,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [CheckpointIntoMLflowArtifacts(), checkpoint_cb, lr_monitor]
 
def find_best_lr(
    cfg: dict,
    min_lr: float = 1e-6,
    max_lr: float = 1.0,
    num_training: int = 200,
    save_dir: str | None = None,
):
    """
    Uses Lightning's LR finder and saves the loss-vs-lr plot.

    Returns:
        best_lr (float)
    """

    cfg_tmp = yaml.safe_load(yaml.safe_dump(cfg))
    L.seed_everything(cfg_tmp["run"].get("seed", 42), workers=True)

    model = build_model(cfg_tmp)
    datamodule = build_datamodule(cfg_tmp)

    trainer = L.Trainer(
        accelerator=cfg_tmp["train_args"].get("accelerator", "auto"),
        devices=cfg_tmp["train_args"].get("devices", "auto"),
        strategy=cfg_tmp["train_args"].get("strategy", "auto"),
        max_steps=num_training,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )

    tuner = Tuner(trainer)

    lr_finder = tuner.lr_find(
        model,
        datamodule=datamodule,
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=num_training,
        mode="exponential",
    )

    best_lr = lr_finder.suggestion()
    if best_lr is None:
        raise RuntimeError("LR finder failed to suggest a learning rate")

    print(f"Suggested learning rate: {best_lr:.2e}")

    # Decide where to save the plot
    if save_dir is None:
        save_dir = cfg_tmp["run"]["output_dir"]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_path = save_dir / "lr_finder.png"

    # Generate and save plot
    fig = lr_finder.plot(suggest=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"LR finder plot saved to: {plot_path}")

    # Log to MLflow if available
    logger = trainer.logger
    if logger is not None and hasattr(logger, "experiment"):
        try:
            logger.experiment.log_artifact(
                logger.run_id,
                str(plot_path),
                artifact_path="lr_finder",
            )
        except Exception as e:
            print(f"WARNING: Failed to log LR finder plot to MLflow: {e}")

    return float(best_lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--find_lr", action="store_true")
    parser.add_argument("--num_steps", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    safe_yaml_check(cfg)
    L.seed_everything(cfg["run"].get("seed", 42), workers=True)

    if args.find_lr:
        best_lr = find_best_lr(cfg, num_training=args.num_steps)
        cfg["optim_args"]["lr"] = best_lr
        exit()

    model = build_model(cfg)
    datamodule = build_datamodule(cfg)
    logger = build_logger(cfg)
    callbacks = build_callbacks(cfg)

    trainer = L.Trainer(
        accelerator=cfg["train_args"].get("accelerator", "auto"),
        devices=cfg["train_args"].get("devices", "auto"),
        strategy=cfg["train_args"].get("strategy", "auto"),
        max_steps=cfg["train_args"]["max_steps"],
        val_check_interval=cfg["train_args"]["val_check_interval"],
        log_every_n_steps=cfg["train_args"]["log_every_n_steps"],
        gradient_clip_val=cfg["train_args"].get("gradient_clip_val", 0.0),
        deterministic=cfg["train_args"].get("deterministic", True),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=cfg["run"]["output_dir"],
        check_val_every_n_epoch=None,
    )

    ckpt_path = "/Users/swornimchhetri/Desktop/all_codes/github_stuff/Image-Captioning/runs/clip_prefix_simple_transformer/mlruns/620745588439826271/57f341b3b26d4ac189bc98fbad69432a/artifacts/checkpoints/step0010000.ckpt"

    logger.log_hyperparams(cfg)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()