import sys
import math
import time
import datetime
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchmetrics.functional import pairwise_cosine_similarity
from torch.utils.tensorboard import SummaryWriter

from utils import cosine_schedule, DINOLoss

@torch.no_grad()
def update_teacher(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    momentum: float
) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_((1.0 - momentum) * ps.detach().data)


def train_one_epoch(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dino_loss: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.Tensor,
    wd_schedule: torch.Tensor,
    momentum_schedule: torch.Tensor,
    epoch: int,
    device: torch.device
) -> dict:
    student.train()
    teacher.eval()

    loss_metric = MeanMetric().to(device)
    cos_metric = MeanMetric().to(device)

    num_iters = len(data_loader)
    
    for it, crops in enumerate(data_loader):
        print(f"\rIteration: {it}/{num_iters}", end="")
        # update weight decay and learning rate according to their schedule
        global_it = epoch * num_iters + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it].item()
            if i == 0:
                param_group["weight_decay"] = wd_schedule[global_it].item()

        # move images to gpu
        crops = [crop.to(device, non_blocking=True) for crop in crops]
        
        # teacher and student forward passes + compute dino loss
        student_out = student(crops)          # all crops
        with torch.no_grad():
            teacher_out = teacher(crops[:2])  # only global crops

        loss = dino_loss(student_out, teacher_out, epoch)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

        # student update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update for the teacher
        update_teacher(
            student,
            teacher,
            momentum_schedule[global_it].item()
        )

        # logging
        loss_metric.update(loss.detach())

        with torch.no_grad():
            cos = pairwise_cosine_similarity(
                student_out[:2], teacher_out
            ).mean()
            cos_metric.update(cos)

    print("\r", end="\n")
    return {
        "loss": loss_metric.compute().item(),
        "cosine_sim": cos_metric.compute().item(),
        "lr": optimizer.param_groups[0]["lr"],
        "wd": optimizer.param_groups[0]["weight_decay"],
        "momentum": momentum_schedule[global_it].item(),
    }


def train_dino(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dino_loss: torch.nn.Module,
    data_loader: DataLoader,
    epochs: int,
    save_path: str,
    base_lr: float = 5e-4,
    min_lr: float = 1e-6,
    weight_decay: float = 0.04,
    weight_decay_end: float = 0.4,
    teacher_momentum: float = 0.996,
    writer: SummaryWriter = None,
    device: torch.device = torch.device("cuda")
) -> None:
    student.to(device)
    teacher.to(device)

    # teacher starts identical
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )

    # ============ init schedulers ... ============
    total_steps = epochs * len(data_loader)

    lr_schedule = cosine_schedule(
        base_lr, min_lr, total_steps, warmup_steps=len(data_loader) * 10
    )

    wd_schedule = cosine_schedule(
        weight_decay, weight_decay_end, total_steps
    )

    momentum_schedule = cosine_schedule(
        teacher_momentum, 1.0, total_steps
    )
    print(f"optimizer and schedulers ready.")

    # ============ training ... ==============
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(epochs):
        epoch_time = time.time()
        stats = train_one_epoch(
            student,
            teacher,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Loss: {stats['loss']:.4f} | "
            f"CosSim: {stats['cosine_sim']:.4f} | "
            f"LR: {stats['lr']:.2e}"
        )
        epoch_time = time.time() - epoch_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        print('Epoch time {}'.format(epoch_time_str))

        if writer is not None:
            writer.add_scalar('Loss', stats['loss'], epoch)
            writer.add_scalar('CosSimilarity', stats['cosine_sim'], epoch)
            writer.add_scalar('LearningRate', stats['lr'], epoch)
            writer.add_scalar('weight_decay', stats['wd'], epoch)
            writer.add_scalar('Params/Momentum', stats['momentum'], epoch)

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'dino_loss': dino_loss.state_dict(),
        }
        torch.save(save_dict, f"{save_path}/dino_{epoch}.pth")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
