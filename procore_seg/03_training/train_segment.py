from __future__ import annotations

"""Stage-2 supervised segmentation fine-tuning for ProCore-Seg."""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import Batch, set_seed
from .losses import DensityWeightedCrossEntropy
from .train_pretrain import (
    _load_architecture_module,
    build_dataloaders,
    load_checkpoint,
    save_checkpoint,
    setup_logging,
    str2bool,
)

LOGGER = logging.getLogger("procore_seg.segment")


@dataclass
class MetricAccumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    core_tp: int = 0
    core_fp: int = 0
    core_fn: int = 0
    core_total: int = 0

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor, core_mask: torch.Tensor) -> None:
        y_true = y_true.int()
        y_pred = y_pred.int()
        tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
        fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
        fn = int(((y_pred == 0) & (y_true == 1)).sum().item())
        tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        if core_mask.any():
            core_true = y_true[core_mask]
            core_pred = y_pred[core_mask]
            self.core_tp += int(((core_pred == 1) & (core_true == 1)).sum().item())
            self.core_fp += int(((core_pred == 1) & (core_true == 0)).sum().item())
            self.core_fn += int(((core_pred == 0) & (core_true == 1)).sum().item())
            self.core_total += int(core_mask.sum().item())

    def compute(self) -> Dict[str, float]:
        total = self.tp + self.fp + self.fn + self.tn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0.0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        iou = self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) > 0 else 0.0
        core_iou = (
            self.core_tp / (self.core_tp + self.core_fp + self.core_fn)
            if (self.core_tp + self.core_fp + self.core_fn) > 0
            else 0.0
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "core_iou": core_iou,
        }


def compute_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, core_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    if core_mask is None:
        core_mask = torch.zeros_like(y_true, dtype=torch.bool)
    accumulator = MetricAccumulator()
    accumulator.update(y_true, y_pred, core_mask)
    return accumulator.compute()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Batch],
    criterion: DensityWeightedCrossEntropy,
    optimizer: Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_max_norm: float,
    log_interval: int,
    density_idx: int,
    density_threshold: float,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    running_loss = 0.0
    running_count = 0
    metrics_acc = MetricAccumulator()

    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc="train", leave=False)
    for step, batch in progress:
        sparse_inputs = batch.sparse_inputs.to(device)
        atom_features = batch.atom_features.to(device)
        atom_labels = batch.atom_labels.to(device)
        inverse_map = batch.inverse_map_atom2voxel.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(sparse_inputs)
            loss = criterion(
                logits.F,
                inverse_map,
                atom_labels,
                atom_features,
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_max_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
            optimizer.step()

        running_loss += loss.item()
        running_count += 1

        with torch.no_grad():
            logits_atom = logits.F[inverse_map]
            preds = logits_atom.argmax(dim=1)
            core_mask = atom_features[:, density_idx] >= density_threshold
            metrics_acc.update(atom_labels, preds, core_mask)

        if log_interval > 0 and step % log_interval == 0:
            LOGGER.info("step=%d loss=%.6f", step, loss.item())
        progress.set_postfix(loss=running_loss / max(running_count, 1))

    return running_loss / max(running_count, 1), metrics_acc.compute()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Batch],
    criterion: DensityWeightedCrossEntropy,
    device: torch.device,
    amp_enabled: bool,
    density_idx: int,
    density_threshold: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    running_count = 0
    metrics_acc = MetricAccumulator()

    with torch.no_grad():
        progress = tqdm(dataloader, desc="val", leave=False)
        for batch in progress:
            sparse_inputs = batch.sparse_inputs.to(device)
            atom_features = batch.atom_features.to(device)
            atom_labels = batch.atom_labels.to(device)
            inverse_map = batch.inverse_map_atom2voxel.to(device)

            with autocast(enabled=amp_enabled):
                logits = model(sparse_inputs)
                loss = criterion(logits.F, inverse_map, atom_labels, atom_features)

            running_loss += loss.item()
            running_count += 1

            logits_atom = logits.F[inverse_map]
            preds = logits_atom.argmax(dim=1)
            core_mask = atom_features[:, density_idx] >= density_threshold
            metrics_acc.update(atom_labels, preds, core_mask)

            progress.set_postfix(loss=running_loss / max(running_count, 1))

    return running_loss / max(running_count, 1), metrics_acc.compute()


def histogram_sanity_check(
    model: nn.Module,
    dataloader: DataLoader[Batch],
    device: torch.device,
    density_idx: int,
    density_threshold: float,
) -> None:
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        LOGGER.warning("No batches available for histogram sanity check")
        return

    model.eval()
    with torch.no_grad():
        sparse_inputs = batch.sparse_inputs.to(device)
        logits = model(sparse_inputs)
        preds = logits.F[batch.inverse_map_atom2voxel.to(device)].argmax(dim=1)
        hist = torch.bincount(preds, minlength=2)
        core_mask = batch.atom_features[:, density_idx] >= density_threshold
        core_hist = torch.bincount(preds[core_mask.to(device)], minlength=2)
        LOGGER.info("Prediction histogram: %s", hist.tolist())
        LOGGER.info("Core prediction histogram: %s", core_hist.tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-2 supervised segmentation training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--encoder-weights", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--density-feature-idx", type=int, default=7)
    parser.add_argument("--dwce-T", type=float, default=0.0)
    parser.add_argument("--dwce-tau", type=float, default=1.0)
    parser.add_argument("--class-weight-pos", type=float, default=1.0)
    parser.add_argument("--grad-max-norm", type=float, default=1.0)
    args = parser.parse_args()

    setup_logging()
    pretrain_logger = logging.getLogger("procore_seg.pretrain")
    if not LOGGER.handlers:
        LOGGER.handlers = pretrain_logger.handlers
    LOGGER.setLevel(logging.INFO)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp) and device.type == "cuda"

    train_loader, val_loader = build_dataloaders(
        args.data_dir,
        args.ids_file,
        args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        voxel_size=args.voxel_size,
    )

    segmentation_module = _load_architecture_module("model_segmentation_unet")
    SegmentationConfig = getattr(segmentation_module, "SegmentationConfig")
    SparseSegmentationUNet = getattr(segmentation_module, "SparseSegmentationUNet")
    seg_cfg = SegmentationConfig()
    model: nn.Module = SparseSegmentationUNet(seg_cfg).to(device)

    if args.encoder_weights is not None and args.encoder_weights.is_file():
        state = torch.load(args.encoder_weights, map_location="cpu")
        if "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        model.load_pretrained_encoder(state_dict, strict=False)
        LOGGER.info("Loaded encoder weights from %s", args.encoder_weights)

    class_weights = torch.tensor([1.0, args.class_weight_pos], dtype=torch.float32)
    criterion = DensityWeightedCrossEntropy(
        density_feature_idx=args.density_feature_idx,
        T=args.dwce_T,
        tau=args.dwce_tau,
        class_weights=class_weights,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler: Optional[_LRScheduler] = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_core_iou = 0.0
    if args.resume is not None and args.resume.is_file():
        LOGGER.info("Resuming from checkpoint %s", args.resume)
        start_epoch, best_core_iou = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

    density_threshold = args.dwce_T
    args_dict = vars(args)
    args_dict_serialisable = {k: (str(v) if isinstance(v, Path) else v) for k, v in args_dict.items()}

    for epoch in range(start_epoch, args.epochs):
        LOGGER.info("Epoch %d/%d", epoch + 1, args.epochs)
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            args.grad_max_norm,
            args.log_interval,
            args.density_feature_idx,
            density_threshold,
        )
        LOGGER.info("Train loss: %.6f | metrics: %s", train_loss, train_metrics)

        val_loss = math.inf
        val_metrics = {"core_iou": 0.0}
        if val_loader is not None:
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled,
                args.density_feature_idx,
                density_threshold,
            )
            LOGGER.info("Val loss: %.6f | metrics: %s", val_loss, val_metrics)

        if scheduler is not None:
            scheduler.step()

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if amp_enabled else None,
            "best_metric": best_core_iou,
            "config": args_dict_serialisable,
        }
        save_checkpoint(args.checkpoint_dir / "last.pth", checkpoint_payload)

        if val_loader is not None and val_metrics.get("core_iou", 0.0) > best_core_iou:
            best_core_iou = val_metrics["core_iou"]
            checkpoint_payload["best_metric"] = best_core_iou
            save_checkpoint(args.checkpoint_dir / "best.pth", checkpoint_payload)
            final_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": args_dict_serialisable,
                "best_core_iou": best_core_iou,
            }
            save_checkpoint(args.checkpoint_dir / "segmenter_final.pth", final_payload)
        elif val_loader is None and epoch == args.epochs - 1:
            save_checkpoint(args.checkpoint_dir / "best.pth", checkpoint_payload)
            final_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": args_dict_serialisable,
                "best_core_iou": best_core_iou,
            }
            save_checkpoint(args.checkpoint_dir / "segmenter_final.pth", final_payload)

    histogram_sanity_check(model, val_loader if val_loader is not None else train_loader, device, args.density_feature_idx, density_threshold)
    LOGGER.info("Training completed. Best core IoU: %.4f", best_core_iou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
