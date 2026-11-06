"""Supervised InfMasking optimization; preserve historical operation order."""

import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import _move_to_device
from models.infmasking import infmask_loss


def train(
    args,
    epoch,
    model,
    device,
    dataloader,
    optimizer,
    scheduler,
    logger=None,
    writer=None,
    global_step_start=0,
    log_interval=50,
):
    return train_attn(
        args,
        epoch,
        model,
        nn.CrossEntropyLoss(),
        device,
        dataloader,
        optimizer,
        scheduler,
        logger,
        writer,
        global_step_start,
        log_interval,
    )


def train_attn(
    args,
    epoch,
    model,
    criterion,
    device,
    dataloader,
    optimizer,
    scheduler,
    logger=None,
    writer=None,
    global_step_start: int = 0,
    log_interval: int = 50,
):
    model.train()
    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _loss_mask = 0
    needed_weights = 4 if args.use_masking else 3 if args.use_single else 1
    if len(args.loss_weights) < needed_weights:
        raise ValueError(
            f"loss_weights too short: need >= {needed_weights} active losses (main[/v/a][/mask]), got {len(args.loss_weights)}: {args.loss_weights}"
        )
    for step, data in tqdm(
        enumerate(dataloader), total=len(dataloader), desc=f"Training Epoch {epoch}"
    ):
        X, y = data
        X = [_move_to_device(X[i], device) for i in range(len(args.modalities))]
        label = y.to(device)
        optimizer.zero_grad()
        output = model(X)
        out = output["out"][0]
        loss = criterion(out, label)
        if args.use_masking:
            masked_fused_list = output["masked_fused_list"]
            ratio = args.mask_lambda * ((epoch + 1) * 1.0 / args.epochs)
            mask_loss = infmask_loss(
                fc=model.fusion.fusion_module.FC,
                mask_out_list=masked_fused_list,
                label=label,
                ratio=ratio,
            )
        if args.use_single:
            out_a = output["out_a"]
            out_v = output["out_v"]
            loss_v = criterion(out_v, label)
            loss_a = criterion(out_a, label)
        total_loss = loss * args.loss_weights[0]
        if args.use_single:
            total_loss = total_loss + loss_v * args.loss_weights[1]
            total_loss = total_loss + loss_a * args.loss_weights[2]
        if args.use_masking:
            total_loss = total_loss + mask_loss * args.loss_weights[3]
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if writer is not None and step % max(int(log_interval), 1) == 0:
            gs = int(global_step_start) + int(step)
            lr = float(optimizer.param_groups[-1]["lr"])
            pred = out.argmax(dim=1)
            acc = float((pred == label).float().mean().item())
            writer.add_scalar("train/lr", lr, gs)
            writer.add_scalar("train/loss_main", float(loss.item()), gs)
            if args.use_single:
                writer.add_scalar("train/loss_audio", float(loss_a.item()), gs)
                writer.add_scalar("train/loss_vision", float(loss_v.item()), gs)
            writer.add_scalar("train/loss_total", float(total_loss.item()), gs)
            writer.add_scalar("train/acc_batch", acc, gs)
            if args.use_masking:
                writer.add_scalar("train/loss_mask", float(mask_loss.item()), gs)
                writer.add_scalar("train/mask_ratio", float(args.mask_ratio), gs)
        _loss += loss.item()
        if args.use_single:
            _loss_a += loss_a.item()
            _loss_v += loss_v.item()
        if args.use_masking:
            _loss_mask += mask_loss.item()
    current_lr = optimizer.param_groups[-1]["lr"]
    if logger is not None:
        logger.log({"epoch": epoch, "loss": _loss / len(dataloader), "lr": current_lr})
    return (
        _loss / len(dataloader),
        _loss_a / len(dataloader) if args.use_single else 0,
        _loss_v / len(dataloader) if args.use_single else 0,
        _loss_mask / len(dataloader) if args.use_masking else 0,
    )
