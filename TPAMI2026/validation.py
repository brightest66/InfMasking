import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import _move_to_device


def _set_clean_validation_mode(args, model):
    """Temporarily disable MT/MT2 masked views for a clean evaluation pass."""
    if not getattr(args, "clean_validation", False):
        return None
    fusion = getattr(model, "fusion", None)
    if fusion is None or not hasattr(fusion, "use_masking"):
        return None
    previous = bool(fusion.use_masking)
    fusion.use_masking = False
    return (fusion, previous)


def _restore_masking_mode(state):
    if state is not None:
        fusion, previous = state
        fusion.use_masking = previous


def _valid(
    args,
    model,
    device,
    dataloader,
    epoch,
    logger,
    criterion,
    name="Val",
    writer=None,
    global_step=None,
):
    with torch.no_grad():
        model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        for _step, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"{name} Epoch {epoch}"
        ):
            if data is None:
                continue
            X, y = data
            label = y.to(device)
            X = [_move_to_device(X[i], device) for i in range(len(args.modalities))]
            clean_state = _set_clean_validation_mode(args, model)
            try:
                output = model(X)
            finally:
                _restore_masking_mode(clean_state)
            out = output["out"]
            if isinstance(out, list):
                out = out[0]
            loss = criterion(out, label)
            batch_size = int(label.numel())
            loss_sum += float(loss.item()) * batch_size
            pred = out.argmax(dim=1)
            correct += int((pred == label).sum().item())
            total += int(label.numel())
        avg_loss = loss_sum / max(total, 1)
        acc = float(correct) / max(float(total), 1e-12)
        if logger is not None:
            logger.log({"epoch": epoch, "loss": avg_loss, "acc": acc})
        if writer is not None:
            split = str(name).lower()
            step = epoch if global_step is None else global_step
            writer.add_scalar(f"{split}/loss", avg_loss, step)
            writer.add_scalar(f"{split}/acc", acc, step)
    return acc


def valid(
    args,
    model,
    device,
    dataloader,
    epoch,
    logger,
    name="Val",
    writer=None,
    global_step=None,
):
    return _valid(
        args,
        model,
        device,
        dataloader,
        epoch,
        logger,
        nn.CrossEntropyLoss(),
        name,
        writer,
        global_step,
    )
