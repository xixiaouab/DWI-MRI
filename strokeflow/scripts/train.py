import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from strokeflow.config import StrokeFlowConfig
from strokeflow.data import StrokeFlowDataset
from strokeflow.losses import StrokeFlowLoss
from strokeflow.metrics import (
    density_concentration_ratio,
    dice_coefficient,
    hd95,
    mae,
    psnr,
    soft_dice,
    soft_iou,
)
from strokeflow.models import StrokeFlowUNet3D

def parse_args():
    parser = argparse.ArgumentParser(description="StrokeFlow Professional Training Script")
    # 基础参数
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--logdir", type=str, default="./runs", help="Directory for tensorboard logs and checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # 训练超参覆盖
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override config learning rate")
    parser.add_argument("--accum_iter", type=int, default=1, help="Gradient accumulation steps (simulate larger batch size)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True 

def create_dataloaders(cfg, data_root, num_workers):
    train_ds = StrokeFlowDataset(
        root=data_root,
        split="train",
        use_mask=True,
        img_size=cfg.img_size,
        seed=cfg.seed,
    )
    val_ds = StrokeFlowDataset(
        root=data_root,
        split="val",
        use_mask=True,
        img_size=cfg.img_size,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    accum_iter: int,
    device: str,
    epoch: int,
    writer: SummaryWriter
) -> Dict[str, float]:

    model.train()
    running_metrics = {
        "loss": 0.0, "L_density": 0.0, "L_align": 0.0, 
        "L_smooth": 0.0, "L_div": 0.0, "L_flow": 0.0
    }
    
    num_steps = len(loader)
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        dens_gt = batch["density_gt"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True) if batch["mask"] is not None else None

        with autocast():
            phi, flow = model(img)
            loss_dict = loss_fn(phi, flow, dens_gt, gt_mask=mask, input_image=img)
            loss = loss_dict["loss"] / accum_iter

        scaler.scale(loss).backward()

        if (i + 1) % accum_iter == 0 or (i + 1) == num_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_val = loss_dict["loss"].item()
        running_metrics["loss"] += loss_val
        for k in running_metrics.keys():
            if k != "loss" and k in loss_dict:
                running_metrics[k] += loss_dict[k].item()
        

    for k in running_metrics.keys():
        running_metrics[k] /= num_steps
        
    return running_metrics

@torch.no_grad()
def validate(
    model: nn.Module, 
    loader: DataLoader, 
    device: str
) -> Dict[str, float]:

    model.eval()
    metrics_list = {
        "Dice": [], "MAE": [], "PSNR": [], 
        "SoftDice": [], "SoftIoU": [], "DCR": [], "HD95": []
    }

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        dens_gt = batch["density_gt"].to(device, non_blocking=True)
        mask = batch["mask"]
        
        if mask is None: continue
        mask = mask.to(device, non_blocking=True)

        phi, flow = model(img)

        metrics_list["MAE"].append(mae(phi, dens_gt))
        metrics_list["PSNR"].append(psnr(phi, dens_gt))
        metrics_list["SoftDice"].append(soft_dice(phi, dens_gt).item())
        metrics_list["SoftIoU"].append(soft_iou(phi, dens_gt).item())
        metrics_list["DCR"].append(density_concentration_ratio(phi, mask))

        pred_mask = (phi >= 0.5).float()
        dice_val = dice_coefficient(pred_mask, mask)
        metrics_list["Dice"].append(dice_val)


        pred_np = pred_mask.cpu().numpy()[0, 0]
        mask_np = mask.cpu().numpy()[0, 0]
        
        if np.sum(pred_np) > 0 and np.sum(mask_np) > 0:
            try:
                hd = hd95(pred_np, mask_np)
                metrics_list["HD95"].append(hd)
            except RuntimeError:

                pass
        else:

            pass

    results = {}
    for k, v in metrics_list.items():
        results[k] = float(np.mean(v)) if len(v) > 0 else 0.0
        
    return results

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. 配置管理
    cfg = StrokeFlowConfig()
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    
    # 2. 目录与日志
    log_dir = Path(args.logdir) / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"Training started. Logs at: {log_dir}")
    print(f"Config: Epochs={cfg.epochs}, BS={cfg.batch_size}, Accum={args.accum_iter}, Device={args.device}")
    cfg.save_yaml(str(log_dir / "config.yaml"))

    # 3. 模型构建
    train_loader, val_loader = create_dataloaders(cfg, args.data_root, args.num_workers)
    
    model = StrokeFlowUNet3D(in_channels=cfg.in_channels).to(args.device)
    
    loss_fn = StrokeFlowLoss(
        lambda_d=cfg.lambda_d, lambda_f=cfg.lambda_f,
        alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
        lesion_threshold=cfg.lesion_threshold
    ).to(args.device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # 注意：Resume 时需要处理 Scheduler 的状态
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    
    # 混合精度 Scaler
    scaler = GradScaler()

    # 4. 断点续训逻辑
    start_epoch = 1
    best_dice = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"Resumed at Epoch {start_epoch}, Best Dice: {best_dice:.4f}")

    # 5. 训练循环
    for epoch in range(start_epoch, cfg.epochs + 1):
        start_time = time.time()
        
        # Train
        train_stats = train_one_epoch(
            model, loss_fn, train_loader, optimizer, 
            scaler, args.accum_iter, args.device, epoch, writer
        )
        
        # Validation
        val_stats = validate(model, val_loader, args.device)
        
        # Scheduler Step
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        # Logging to TensorBoard
        writer.add_scalar("LR", curr_lr, epoch)
        for k, v in train_stats.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        for k, v in val_stats.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

        # Logging to Console
        epoch_time = time.time() - start_time
        log_str = (
            f"Ep [{epoch}/{cfg.epochs}] Time:{epoch_time:.0f}s | "
            f"Loss:{train_stats['loss']:.4f} "
            f"(D:{train_stats['L_density']:.3f} F:{train_stats['L_flow']:.3f}) | "
            f"Dice:{val_stats['Dice']:.4f} "
            f"HD95:{val_stats['HD95']:.2f} "
            f"PSNR:{val_stats['PSNR']:.2f}"
        )
        print(log_str)

        # Checkpointing
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": cfg.__dict__,
            "best_dice": best_dice,
        }
        

        torch.save(state, ckpt_dir / "last.pth")


        if val_stats["Dice"] > best_dice:
            best_dice = val_stats["Dice"]
            state["best_dice"] = best_dice
            torch.save(state, ckpt_dir / "best_dice.pth")
            print(f" >> New Best Dice: {best_dice:.4f}")

    writer.close()
    print("Training Finished.")

if __name__ == "__main__":
    main()