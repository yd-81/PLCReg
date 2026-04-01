import os
import glob
import csv
import json
import time
import socket
import getpass
import warnings
import datetime
from collections import defaultdict

import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
import torch.nn as nn
from natsort import natsorted
from tqdm import tqdm

from utils import losses
from utils.config import args
from utils.datagenerators_atlas import Dataset
from Model.STN import SpatialTransformer
from Model.PolaReg import PolaReg

import logging
from logging import handlers

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
)
warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed to align_corners=False since"
)

KEEP_PREFIXES_DEFAULT = [
    "corr_2.", "corr_3.", "corr_4.", "corr_5.",
    "conv_1.","conv_2.", "conv_3.", "conv_4.", "conv_5.",
    "conv_fine_1.","conv_fine_2.","conv_fine_3.","conv_fine_4.",
    "reghead_1.","reghead_2.", "reghead_3.", "reghead_4.", "reghead_5.",
    "fine_reghead_1.", "fine_reghead_2.", "fine_reghead_3.", "fine_reghead_4.",
]


def _parse_train_parts_from_env(logger=None):
    env_v = os.environ.get("POLAREG_TRAIN_PARTS", "").strip()
    if not env_v:
        if logger: logger.info(f"[Freeze] Using default train parts: {KEEP_PREFIXES_DEFAULT}")
        return KEEP_PREFIXES_DEFAULT
    parts = [p.strip() for p in env_v.split(",") if p.strip()]
    parts = [p if p.endswith(".") else (p + ".") for p in parts]
    if logger: logger.info(f"[Freeze] Using train parts from env: {parts}")
    return parts

def make_dirs():
    for p in [args.model_dir, args.log_dir, args.result_dir]:
        if not os.path.exists(p):
            os.makedirs(p)


def setup_run_dirs_and_logging():
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.log_dir, "OASIS", ts)
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("PolaReg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    step_csv_path  = os.path.join(run_dir, "train_steps.csv")
    epoch_csv_path = os.path.join(run_dir, "eval_epoch.csv")
    case_csv_path  = os.path.join(run_dir, "eval_cases.csv")

    step_csv = open(step_csv_path, "w", newline="")
    epoch_csv = open(epoch_csv_path, "w", newline="")
    case_csv = open(case_csv_path, "w", newline="")

    step_writer = csv.writer(step_csv)
    epoch_writer = csv.writer(epoch_csv)
    case_writer = csv.writer(case_csv)

    step_writer.writerow(["epoch", "global_step", "sim_loss", "grad_loss", "total_loss"])
    epoch_writer.writerow(["epoch", "mean_dice", "std_dice"])
    case_writer.writerow(["epoch", "case_name", "dice"])

    config_path = os.path.join(run_dir, "config_snapshot.txt")
    snapshot = {
        "timestamp": ts,
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": args.gpu,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha_smooth": args.alpha,
        "sim_loss": args.sim_loss,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "atlas_file": args.atlas_file,
        "label_dir": args.label_dir,
        "model_dir": args.model_dir,
        "log_dir": args.log_dir,
        "result_dir": args.result_dir,
    }
    with open(config_path, "w", encoding="utf-8") as cf:
        json.dump(snapshot, cf, indent=2, ensure_ascii=False)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Step CSV     : {step_csv_path}")
    logger.info(f"Epoch CSV    : {epoch_csv_path}")
    logger.info(f"Cases CSV    : {case_csv_path}")
    logger.info(f"Config saved : {config_path}")

    return {
        "run_dir": run_dir,
        "logger": logger,
        "step_csv_file": step_csv,
        "epoch_csv_file": epoch_csv,
        "case_csv_file": case_csv,
        "step_writer": step_writer,
        "epoch_writer": epoch_writer,
        "case_writer": case_writer,
    }


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4, logger=None):
    os.makedirs(save_dir, exist_ok=True)
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(model_lists) > max_model_num:
        try:
            os.remove(model_lists[0])
        except Exception as e:
            if logger:
                logger.warning(f"Remove old ckpt failed: {model_lists[0]} - {e}")
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    fpath = os.path.join(save_dir, filename)
    torch.save(state, fpath)
    if logger:
        logger.info(f"Checkpoint saved: {fpath}")


def compute_label_dice(gt, pred):
    cls_lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

def human_readable_count(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.0f}T"

def load_pretrained_strict_match(model: torch.nn.Module,
                                 ckpt_path: str,
                                 logger,
                                 print_limit: int = 80):
    if not os.path.isfile(ckpt_path):
        logger.warning(f"[Pretrain] Checkpoint not found: {ckpt_path}")
        return

    logger.info("=" * 80)
    logger.info(f"[Pretrain] Load (strict name+shape) from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    sd_raw = ckpt.get("state_dict", ckpt)

    model_sd = model.state_dict()
    filtered = {}
    skipped_shape = []  
    skipped_name  = []  
    for k, v in sd_raw.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            skipped_name.append(k)

    missing_in_ckpt = [k for k in model_sd.keys() if k not in filtered]

    msg = model.load_state_dict(filtered, strict=False)

    logger.info(f"[Pretrain] Loaded params    : {len(filtered)}")
    logger.info(f"[Pretrain] Skipped by shape : {len(skipped_shape)}")
    logger.info(f"[Pretrain] Skipped by name  : {len(skipped_name)}")
    logger.info(f"[Pretrain] Model missing    : {len(missing_in_ckpt)}")
    unexpected = getattr(msg, "unexpected_keys", [])
    logger.info(f"[Pretrain] Unexpected keys  : {len(unexpected)}")

    if skipped_shape:
        logger.info("[Pretrain] - Skipped (shape mismatch):")
        for i, (k, s_ckpt, s_model) in enumerate(skipped_shape[:print_limit]):
            logger.info(f"  {k}  ckpt={s_ckpt}  model={s_model}")
        if len(skipped_shape) > print_limit:
            logger.info(f"  ... and {len(skipped_shape)-print_limit} more")

    if skipped_name:
        logger.info("[Pretrain] - Skipped (name not in model):")
        for i, k in enumerate(skipped_name[:print_limit]):
            logger.info(f"  {k}")
        if len(skipped_name) > print_limit:
            logger.info(f"  ... and {len(skipped_name)-print_limit} more")

    if missing_in_ckpt:
        logger.info("[Pretrain] - Missing in ckpt (present in model but not loaded):")
        for i, k in enumerate(missing_in_ckpt[:print_limit]):
            logger.info(f"  {k}")
        if len(missing_in_ckpt) > print_limit:
            logger.info(f"  ... and {len(missing_in_ckpt)-print_limit} more")

    if unexpected:
        logger.info("[Pretrain] - Unexpected (in ckpt but no use after filtering):")
        for i, k in enumerate(unexpected[:print_limit]):
            logger.info(f"  {k}")
        if len(unexpected) > print_limit:
            logger.info(f"  ... and {len(unexpected)-print_limit} more")

    logger.info("=" * 80)


def freeze_except(model: torch.nn.Module, keep_prefixes, logger=None):

    def _keep(name: str):
        return any(name.startswith(p) for p in keep_prefixes)

    total, trainable = 0, 0
    kept_names, frozen_names = [], []
    for name, p in model.named_parameters():
        total += p.numel()
        if _keep(name):
            p.requires_grad = True
            trainable += p.numel()
            kept_names.append(name)
        else:
            p.requires_grad = False
            frozen_names.append(name)

    if logger:
        logger.info(f"[Freeze] Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}")
        preview = 20
        logger.info(f"[Freeze] Kept (trainable) param tensors (show up to {preview}):")
        for n in kept_names[:preview]:
            logger.info(f"  {n}")
        if len(kept_names) > preview:
            logger.info(f"  ... and {len(kept_names)-preview} more")
        logger.info(f"[Freeze] Frozen param tensors (show up to {preview}):")
        for n in frozen_names[:preview]:
            logger.info(f"  {n}")
        if len(frozen_names) > preview:
            logger.info(f"  ... and {len(frozen_names)-preview} more")


def set_frozen_modules_eval(model: torch.nn.Module, keep_prefixes, logger=None):
    def _keep(name: str):
        return any(name.startswith(p) for p in keep_prefixes)

    cnt_eval = 0
    for name, m in model.named_modules():
        if not name: 
            continue
        if not _keep(name):
            m.eval()
            for p in m.parameters(recurse=False):
                p.requires_grad = False
            cnt_eval += 1
    if logger:
        logger.info(f"[Freeze] Set frozen submodules to eval(): {cnt_eval}")

def human_readable_count(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.0f}T"

def train():
    make_dirs()
    io = setup_run_dirs_and_logging()
    logger = io["logger"]
    step_writer = io["step_writer"]
    epoch_writer = io["epoch_writer"]
    case_writer = io["case_writer"]
    run_dir = io["run_dir"]

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed_np = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]     
    vol_size = input_fixed_np.shape[2:]

    input_fixed_eval = torch.from_numpy(input_fixed_np).to(device).float()
    input_fixed = np.repeat(input_fixed_np, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    fixed_label = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(args.label_dir, "OASIS_0395_0000.nii.gz"))
    )[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()

    net = PolaReg(args).to(device)

    print("=" * 80)
    print(f"PLCReg Structure")
    print("=" * 80)
    print(net)  

    total, trainable = count_parameters(net)
    # print("\n[Parameter Count]")
    # print(f"Total params     : {human_readable_count(total)} ({total:,})")
    # print(f"Trainable params : {human_readable_count(trainable)} ({trainable:,})")

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)

    contTrain = True
    if contTrain:
        pretrain_path = './Checkpoint/OASIS/final.pth.tar'
        load_pretrained_strict_match(model=net, ckpt_path=pretrain_path, logger=logger)
    iterEpoch = 1 

    keep_prefixes = _parse_train_parts_from_env(logger)
    freeze_except(net, keep_prefixes, logger)
    set_frozen_modules_eval(net, keep_prefixes, logger)

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    opt = Adam(trainable_params, lr=args.lr, weight_decay=0, amsgrad=True)
    # logger.info(f"[Optim] Trainable tensors: {len(trainable_params)}")

    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = Dataset(files=train_files)
    # logger.info(f"Number of training images: {len(DS)}")
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    global_step = 0
    best_mean_dice = -1.0
    best_ckpt_path = os.path.join("./Checkpoint/OASIS/", "best.pth.tar")

   
    for epoch in tqdm(range(iterEpoch, args.epochs + 1), desc="Epochs"):
        net.train()
        set_frozen_modules_eval(net, keep_prefixes, logger)

        STN.train()

        for input_moving, fig_name in tqdm(DL, desc=f"  Epoch {epoch}", leave=False):
            global_step += 1
            input_moving = input_moving.to(device).float()

            flow_m2f = net(input_fixed, input_moving)
            m2f = STN(input_fixed, flow_m2f)

            sim_loss = sim_loss_fn(m2f, input_moving)
            grad_loss = grad_loss_fn(flow_m2f)
            loss = sim_loss + args.alpha * grad_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            flow_m2f = net(input_moving, input_fixed)
            m2f = STN(input_moving, flow_m2f)

            sim_loss_b = sim_loss_fn(m2f, input_fixed)
            grad_loss_b = grad_loss_fn(flow_m2f)
            loss_b = sim_loss_b + args.alpha * grad_loss_b

            opt.zero_grad()
            loss_b.backward()
            opt.step()

            step_writer.writerow([
                epoch,
                global_step,
                float((sim_loss.item() + sim_loss_b.item()) / 2.0),
                float((grad_loss.item() + grad_loss_b.item()) / 2.0),
                float((loss.item() + loss_b.item()) / 2.0),
            ])

        test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))

        net.eval()
        STN.eval()
        STN_label.eval()

        DSC = []
        with torch.no_grad():
            for file in test_file_lst:
                name = os.path.split(file)[1]

                input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
                input_moving = torch.from_numpy(input_moving).to(device).float()

                label_file = glob.glob(os.path.join(args.label_dir, name[:10] + "*"))[0]
                moving_label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
                moving_label = torch.from_numpy(moving_label_np).to(device).float()

                pred_flow = net(input_moving, input_fixed_eval)
                pred_img = STN(input_moving, pred_flow)

                pred_label = STN_label(moving_label, pred_flow)   

                dice = compute_label_dice(
                    fixed_label[0, 0, ...].cpu().numpy(),
                    pred_label[0, 0, ...].cpu().numpy()
                )
                DSC.append(dice)

                case_writer.writerow([epoch, name, float(dice)])

                del pred_flow, pred_img, pred_label, input_moving, moving_label

        mean_dice = float(np.mean(DSC)) if len(DSC) else 0.0
        std_dice  = float(np.std(DSC)) if len(DSC) else 0.0
        logger.info(f"[Eval] Epoch {epoch} - mean(DSC): {mean_dice:.6f}, std(DSC): {std_dice:.6f}")
        epoch_writer.writerow([epoch, mean_dice, std_dice])

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': opt.state_dict(),
            'mean_dice': mean_dice,
        }, save_dir='./Checkpoint/OASIS/', filename='dsc{:.4f}epoch{:0>3d}.pth.tar'.format(mean_dice, epoch + 1), logger=logger)

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': opt.state_dict(),
                'mean_dice': mean_dice,
            }, best_ckpt_path)
            logger.info(f"[Best] Updated best model @ epoch {epoch}, mean_dice={mean_dice:.6f} -> {best_ckpt_path}")

    io["step_csv_file"].close()
    io["epoch_csv_file"].close()
    io["case_csv_file"].close()
    logger.info("Training done.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
