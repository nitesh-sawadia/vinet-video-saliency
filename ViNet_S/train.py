import argparse
import datetime
import glob
import json
import os
import re
import shutil
import sys
import time

import cv2
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np

from ViNet_S_dataloader import *
from loss import *
from ViNet_S_model import *
from outplay_dataset import OutplayWindowDataset
from dataset_version import verify_dataset_version, check_resume_dataset_version
from utils import *

OUTPLAY_MLFLOW_URI = "http://104.105.26.182:5000"
OUTPLAY_MLFLOW_EXPERIMENT = "sports-saliency-ViNet-S"
OUTPLAY_SPLIT_ROOT = "/home/nitesh_sawadia/mnt/ml-fanpulse/sports_saliency/vinet_outplay/split_stride8"
OUTPLAY_RUNS_ROOT = "/home/nitesh_sawadia/vinet_s/runs"
OUTPLAY_TB_ROOT = "/home/nitesh_sawadia/vinet_s/tensorboard"
OUTPLAY_VAL_IMAGE_INDEXES = (0, 65, 130, 195, 260, 325, 390, 455)
_OVERLAY_ALPHA = 0.5
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=40, type=int)
parser.add_argument('--lr',default=1e-4, type=float)
parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=True, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--nss_emlnet',default=False, type=bool)
parser.add_argument('--nss_norm',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)
parser.add_argument('--lr_sched',default=False, type=bool)
parser.add_argument('--lr_factor',default=0.5, type=float,
                     help="ReduceLROnPlateau multiplicative factor when val loss plateaus")
parser.add_argument('--lr_patience',default=3, type=int,
                     help="ReduceLROnPlateau epochs with no val-loss improvement before LR drop")
parser.add_argument('--lr_threshold',default=1e-3, type=float,
                     help="ReduceLROnPlateau minimum relative improvement to count as progress")
parser.add_argument('--lr_min',default=1e-7, type=float,
                     help="ReduceLROnPlateau lower bound on learning rate")
parser.add_argument('--optim',default="Adam", type=str)

parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--step_size',default=5, type=int)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=0.0, type=float)
parser.add_argument('--nss_coeff',default=0.0, type=float)
parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)

parser.add_argument('--batch_size',default=8, type=int)
parser.add_argument('--log_interval',default=5, type=int)
parser.add_argument('--no_workers',default=2, type=int)
parser.add_argument('--prefetch_factor',default=4, type=int)

parser.add_argument('--clip_size',default=32, type=int)

parser.add_argument('--frames_path', default='frames', type=str)
parser.add_argument('--decoder_upsample',default=1, type=int)

parser.add_argument('--checkpoint_path',default="None", type=str)
parser.add_argument('--num_hier',default=3, type=int)
parser.add_argument('--dataset',default="DHF1KDataset", type=str)
parser.add_argument('--alternate',default=1, type=int)

parser.add_argument('--split',default=-1, type=int)

parser.add_argument('--use_wandb',default=False, type=bool)
parser.add_argument('--wandb_project',default="vinet", type=str)
parser.add_argument('--wandb_username',default="bhavberi", type=str)
parser.add_argument('--pin_memory',default=False, type=bool)
parser.add_argument('--combine_datasets', default=False, type=bool)

parser.add_argument('--grouped_conv',default=False, type=bool)
parser.add_argument('--root_grouping', default=False, type=bool)
parser.add_argument('--depth_grouping', default=False, type=bool)
parser.add_argument('--efficientnet', default=False, type=bool)

parser.add_argument('--use_trilinear_upsampling', default=False, type=bool)

parser.add_argument('--videos_root_path', default='', type=str)

parser.add_argument('--videos_frames_root_path', default='', type=str)


parser.add_argument('--fixation_data_path', default='', type=str)

parser.add_argument('--gt_sal_maps_path', default='', type=str)

parser.add_argument('--model_save_path', default='', type=str)

parser.add_argument('--fold_lists_path',default='',type=str)

parser.add_argument('--split_root', default=OUTPLAY_SPLIT_ROOT, type=str)
parser.add_argument('--dataset_version', default='', type=str,
                     help="optional lakefs://<repo>/<tag> pin; omitted, training reads --split_root with no Cohub check")
parser.add_argument('--save_every', default=1, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--clear_tensorboard_media', action='store_true')



args = parser.parse_args()

if args.clear_tensorboard_media and not args.resume:
    print("--clear_tensorboard_media requires --resume")
    sys.exit(1)

if args.dataset == "outplay":
    args.grouped_conv = True
    args.root_grouping = True
    args.depth_grouping = False
    args.efficientnet = False
    args.num_hier = 3
    args.clip_size = 32
    # Optional Cohub pin. Frame reads stay on --split_root either way.
    # The manifest check runs only when --dataset_version is set.
    verify_dataset_version(args.dataset_version, args.split_root)

print(args)

if args.grouped_conv and args.efficientnet:
    print("Grouped conv and efficientnet cannot be used together")
    exit(1)

if args.root_grouping and args.depth_grouping:
    print("Root grouping and depth grouping cannot be used together")
    exit(1)

if args.checkpoint_path == 'None':
    file_weight = os.path.expanduser('~/ICASSP_Saliency/pretrained_models/S3D_kinetics400.pt')
    assert os.path.exists(file_weight), "Pretrained model not found at {}".format(file_weight)
else:
    file_weight = ''



if args.use_wandb:
    import wandb
    config = {
        "lr": args.lr, 
        "model_type": "VideoSaliency",
        "Backbone": "S3D",
        "optimizer": args.optim,
        "criterion": "KLDivLoss",
        "num_epochs": args.no_epochs,
        "clip_size": args.clip_size,
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "gpu_id": 0,
        "grouping": args.grouped_conv,
        "root_grouping": args.root_grouping,
        "depth_grouping": args.depth_grouping,
        "wandb_run_name": "bhav"
    }

    wandb.init(
            entity = args.wandb_username,   # wandb username
            project = args.wandb_project,   # wandb project name. New project will be created if given project is missing.
            config = config         # Config dict
            )
    wandb.run.name = f"{config['model_type']}_{config['dataset']}_{config['clip_size']}_{config['criterion']}_{config['Backbone']}"
    if args.grouped_conv:
        if args.depth_grouping:
            wandb.run.name += "_depthgrouped"
        elif args.root_grouping:
            wandb.run.name += "_rootgrouped"
        else:
            wandb.run.name += "_grouped"
    if args.pin_memory:
        wandb.run.name += "_pinned"


def _build_saliency_model(args):
    return VideoSaliencyModel(
        use_upsample=bool(args.decoder_upsample),
        num_hier=args.num_hier,
        num_clips=args.clip_size,
        grouped_conv=args.grouped_conv,
        root_grouping=args.root_grouping,
        depth=args.depth_grouping,
        efficientnet=args.efficientnet,
        BiCubic=not args.use_trilinear_upsampling,
    )


model = _build_saliency_model(args)

np.random.seed(0)
torch.manual_seed(0)


if args.dataset == "DHF1K":
    train_dataset = DHF1KDataset(args.clip_size, mode="train", alternate=args.alternate, frames_path=args.frames_path,args = args)
    val_dataset = DHF1KDataset(args.clip_size, mode="val", alternate=args.alternate, frames_path=args.frames_path,args = args)

elif args.dataset == "Hollywood2":
    train_dataset = HollywoodDataset(args.clip_size, mode="train", frames_path=args.frames_path,args = args)
    val_dataset = HollywoodDataset(args.clip_size, mode="val", frames_path=args.frames_path,args = args)

elif args.dataset == "UCF":
    train_dataset = UCFDataset(args.clip_size, mode="train", frames_path=args.frames_path,args = args)
    val_dataset = UCFDataset(args.clip_size, mode="val", frames_path=args.frames_path,args = args)

elif args.dataset == "DIEM":
    train_dataset = OtherAudioVisualDataset(args.clip_size, mode="train", dataset_name=args.dataset, split=args.split ,args = args)
    val_dataset = OtherAudioVisualDataset(args.clip_size, mode="val", dataset_name=args.dataset, split=args.split,args = args)

elif args.dataset == "AVAD":
    train_dataset = OtherAudioVisualDataset(args.clip_size, mode="train", dataset_name=args.dataset, split=args.split ,args = args)
    val_dataset = OtherAudioVisualDataset(args.clip_size, mode="val", dataset_name=args.dataset, split=args.split,args = args)
elif args.dataset == "ETMD_av":
    train_dataset = OtherAudioVisualDataset(args.clip_size, mode="train", dataset_name=args.dataset, split=args.split ,args = args)
    val_dataset = OtherAudioVisualDataset(args.clip_size, mode="val", dataset_name=args.dataset, split=args.split,args = args)
elif args.dataset == "Coutrot_db1":
    train_dataset = OtherAudioVisualDataset(args.clip_size, mode="train", dataset_name=args.dataset, split=args.split ,args = args)
    val_dataset = OtherAudioVisualDataset(args.clip_size, mode="val", dataset_name=args.dataset, split=args.split,args = args)
elif args.dataset == "Coutrot_db2":
    train_dataset = OtherAudioVisualDataset(args.clip_size, mode="train", dataset_name=args.dataset, split=args.split ,args = args)
    val_dataset = OtherAudioVisualDataset(args.clip_size, mode="val", dataset_name=args.dataset, split=args.split,args = args)
elif args.dataset == "mvva":
    train_dataset = MVVADataset(args.clip_size, mode="train", dataset_name=args.dataset, split=args.split ,args = args)
    val_dataset = MVVADataset(args.clip_size, mode="val", dataset_name=args.dataset, split=args.split,args = args)
elif args.dataset == "outplay":
    train_dataset = OutplayWindowDataset(args.split_root, "train", clip_size=args.clip_size)
    val_dataset = OutplayWindowDataset(args.split_root, "val", clip_size=args.clip_size)


loader_kwargs = {
    "num_workers": args.no_workers,
    "pin_memory": args.pin_memory,
}
if args.no_workers > 0:
    loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader_kwargs["persistent_workers"] = True

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    **loader_kwargs,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    **loader_kwargs,
)


def _coerce_state_dict(loaded):
    if not isinstance(loaded, dict):
        print("Checkpoint is %s, expected a state dict" % type(loaded).__name__)
        sys.exit(1)
    if loaded and all(torch.is_tensor(value) for value in loaded.values()):
        return loaded
    for key in ("state_dict", "model"):
        nested = loaded.get(key)
        if isinstance(nested, dict) and nested and all(torch.is_tensor(value) for value in nested.values()):
            print("Using checkpoint entry '%s'" % key)
            return nested
    return loaded


def _strip_module_prefix(state):
    if not any(isinstance(key, str) and key.startswith("module.") for key in state):
        return state
    stripped = {}
    for key, value in state.items():
        if isinstance(key, str) and key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped


def _load_state_into_model(model, state):
    state = _strip_module_prefix(state)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        model_sd = model.state_dict()
        missing = [key for key in model_sd.keys() if key not in state]
        unexpected = [key for key in state.keys() if key not in model_sd]
        print("Strict load_state_dict failed.")
        print(exc)
        print("missing keys (%d):" % len(missing))
        for key in missing:
            print(key)
        print("unexpected keys (%d):" % len(unexpected))
        for key in unexpected:
            print(key)
        sys.exit(1)


def _load_saliency_checkpoint(model, path):
    if not os.path.isfile(path):
        print("Saliency checkpoint not found: %s" % path)
        sys.exit(1)
    print("Loading pretrained saliency model weights:", path)
    _load_state_into_model(model, _coerce_state_dict(torch.load(path, map_location="cpu")))
    print("Loaded saliency checkpoint with strict=True")


def _is_training_bundle(loaded):
    model_state = loaded.get("model") if isinstance(loaded, dict) else None
    return (
        isinstance(model_state, dict)
        and "epoch" in loaded
        and bool(model_state)
        and all(torch.is_tensor(value) for value in model_state.values())
    )


def _epoch_files(ckpt_dir):
    found = []
    if not os.path.isdir(ckpt_dir):
        return found
    for filename in os.listdir(ckpt_dir):
        match = re.fullmatch(r"epoch_(\d+)\.pt", filename)
        if match:
            found.append((int(match.group(1)), os.path.join(ckpt_dir, filename)))
    found.sort()
    return found


def _legacy_completed_epoch(path):
    name = os.path.basename(path)
    match = re.fullmatch(r"epoch_(\d+)\.pt", name)
    if match:
        print("Legacy weight file; completed epoch %s comes from the filename" % match.group(1))
        return int(match.group(1))
    if name == "last.pt":
        epochs = _epoch_files(os.path.dirname(path))
        if epochs:
            epoch = epochs[-1][0]
            print("Legacy weight file; completed epoch %d comes from the newest epoch_*.pt" % epoch)
            return epoch
    if name == "best.pt":
        print("best.pt has no epoch or optimizer. Pass last.pt or epoch_<N>.pt to continue the counter.")
    print("Checkpoint has no epoch; training starts at epoch 0 with these weights")
    return None


def _read_training_checkpoint(path):
    loaded = torch.load(path, map_location="cpu")
    if _is_training_bundle(loaded):
        best_loss = loaded.get("best_loss")
        if best_loss is None:
            best_loss = float("inf")
        optimizer_state = loaded.get("optimizer")
        scheduler_state = loaded.get("scheduler")
        return {
            "model": _strip_module_prefix(loaded["model"]),
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "completed_epoch": int(loaded["epoch"]),
            "best_loss": float(best_loss),
            "has_optimizer": optimizer_state is not None,
            "has_scheduler": scheduler_state is not None,
        }
    completed = _legacy_completed_epoch(path)
    return {
        "model": _strip_module_prefix(_coerce_state_dict(loaded)),
        "optimizer": None,
        "scheduler": None,
        "completed_epoch": completed,
        "best_loss": float("inf"),
        "has_optimizer": False,
        "has_scheduler": False,
    }


def _find_run_dir(spec):
    if not os.path.isdir(OUTPLAY_RUNS_ROOT):
        print("Runs directory not found:", OUTPLAY_RUNS_ROOT)
        sys.exit(1)
    names = sorted(os.listdir(OUTPLAY_RUNS_ROOT))
    matches = [name for name in names if name == spec or name.endswith("_" + spec)]
    if len(matches) == 1:
        return os.path.join(OUTPLAY_RUNS_ROOT, matches[0])
    if not matches:
        print("No run matches:", spec)
        sys.exit(1)
    print("Multiple runs match %s:" % spec)
    for name in matches:
        print(name)
    sys.exit(1)


def _resolve_resume(spec):
    spec = os.path.expanduser(spec.strip())
    if os.path.isfile(spec):
        ckpt = os.path.abspath(spec)
        parent = os.path.dirname(ckpt)
        if os.path.basename(parent) != "checkpoints":
            print("Resume checkpoint must be inside a run checkpoints directory:", ckpt)
            sys.exit(1)
        return os.path.dirname(parent), ckpt
    if os.path.isdir(spec):
        run_dir = os.path.abspath(spec)
    else:
        run_dir = _find_run_dir(spec)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt = os.path.join(ckpt_dir, "last.pt")
    if not os.path.isfile(ckpt):
        epochs = _epoch_files(ckpt_dir)
        if not epochs:
            print("No last.pt or epoch checkpoint in", run_dir)
            sys.exit(1)
        ckpt = epochs[-1][1]
        print("last.pt not found; resuming from", ckpt)
    return run_dir, ckpt


def _mlflow_run_id(run_dir):
    config_path = os.path.join(run_dir, "config", "args.json")
    if os.path.isfile(config_path):
        with open(config_path) as handle:
            payload = json.load(handle)
        run_id = payload.get("mlflow_run_id")
        if run_id:
            return run_id
    name = os.path.basename(run_dir.rstrip("/"))
    if "_" not in name:
        print("Cannot find an MLflow run id for", run_dir)
        sys.exit(1)
    return name.rsplit("_", 1)[-1]


resume_state = None
if args.resume:
    if args.dataset != "outplay":
        print("--resume is only implemented for dataset=outplay")
        sys.exit(1)
    run_dir, ckpt_path = _resolve_resume(args.resume)
    if args.dataset == "outplay":
        check_resume_dataset_version(args.dataset_version, run_dir)
    resume_state = _read_training_checkpoint(ckpt_path)
    resume_state["run_dir"] = run_dir
    resume_state["checkpoint_path"] = ckpt_path
    print("Resuming run %s from %s" % (os.path.basename(run_dir), ckpt_path))
    if args.checkpoint_path != "None":
        print("--checkpoint_path is ignored because --resume loads this run")
    _load_state_into_model(model, resume_state["model"])
    print("Loaded resume weights with strict=True")
elif args.checkpoint_path != "None":
    _load_saliency_checkpoint(model, args.checkpoint_path)
elif os.path.isfile(file_weight):
    print ('loading kinetics weight file')
    weight_dict = torch.load(file_weight)
    model_dict = model.backbone.state_dict()
    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if 'base.' in name:
            bn = int(name.split('.')[1])
            sn_list = [0, 5, 8, 14]
            sn = sn_list[0]
            if bn >= sn_list[1] and bn < sn_list[2]:
                sn = sn_list[1]
            elif bn >= sn_list[2] and bn < sn_list[3]:
                sn = sn_list[2]
            elif bn >= sn_list[3]:
                sn = sn_list[3]
            name = '.'.join(name.split('.')[2:])
            name = 'base%d.%d.'%(sn_list.index(sn)+1, bn-sn)+name
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print (' size? ' + name, param.size(), model_dict[name].size())
        else:
            print (' name? ' + name)

    print (' loaded')
    model.backbone.load_state_dict(model_dict)
else:
    print ('weight file?')

if args.checkpoint_path == "None":
    assert os.path.exists(args.checkpoint_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

params = list(filter(lambda p: p.requires_grad, model.parameters())) 
optimizer = torch.optim.Adam(params, lr=args.lr)
if resume_state is not None and resume_state["has_optimizer"]:
    optimizer.load_state_dict(resume_state["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    print("Restored optimizer state, lr", optimizer.param_groups[0]["lr"])
elif resume_state is not None:
    print("Checkpoint has no optimizer state; Adam starts fresh")

scheduler = None
if args.lr_sched:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        min_lr=args.lr_min,
    )
    if resume_state is not None and resume_state["has_scheduler"]:
        scheduler.load_state_dict(resume_state["scheduler"])
        print(
            "Restored ReduceLROnPlateau state, lr",
            optimizer.param_groups[0]["lr"],
        )
    elif resume_state is not None:
        print("Checkpoint has no scheduler state; ReduceLROnPlateau starts fresh")
    else:
        print(
            "ReduceLROnPlateau enabled: factor=%s patience=%s threshold=%s min_lr=%s"
            % (args.lr_factor, args.lr_patience, args.lr_threshold, args.lr_min)
        )

print(device)

def _denormalize_frame(frame_chw):
    frame = frame_chw.detach().float().cpu().numpy()
    image = frame * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def _minmax_for_display(sal):
    low = float(np.min(sal))
    high = float(np.max(sal))
    if high <= low:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - low) / (high - low)).astype(np.float32, copy=False)


def _jet_overlay(frame_chw, sal_hw):
    frame = _denormalize_frame(frame_chw)
    sal = sal_hw.detach().float().cpu().numpy()
    height, width = frame.shape[1], frame.shape[2]
    if sal.shape != (height, width):
        sal = cv2.resize(sal, (width, height), interpolation=cv2.INTER_LINEAR)
    sal = _minmax_for_display(sal)
    heat_bgr = cv2.applyColorMap(np.round(sal * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame_hwc = np.transpose(frame, (1, 2, 0))
    blended = (1.0 - _OVERLAY_ALPHA) * frame_hwc + _OVERLAY_ALPHA * heat
    return np.ascontiguousarray(np.transpose(np.clip(blended, 0.0, 1.0), (2, 0, 1)))


def _as_video(frames_tchw):
    # add_video wants (N, T, C, H, W). N=1 keeps one clip instead of a padded grid.
    return torch.from_numpy(np.ascontiguousarray(frames_tchw)).unsqueeze(0)


def _saliency_map(sal_map):
    sal_map = sal_map.detach()
    if sal_map.dim() == 3 and sal_map.shape[0] == 1:
        sal_map = sal_map[0]
    return sal_map


def _log_comparison_video(writer, clip_id, frames, gt_maps, pred_map, step, tag="samples/%s"):
    """One 1x3 video: raw clip | ground-truth overlay | prediction overlay."""
    pred = _saliency_map(pred_map)
    panels = []
    for t in range(int(frames.shape[0])):
        frame = frames[t]
        gt = gt_maps[t] if gt_maps.dim() == 3 else gt_maps
        panel = np.concatenate(
            [
                _denormalize_frame(frame),
                _jet_overlay(frame, _saliency_map(gt)),
                _jet_overlay(frame, pred),
            ],
            axis=2,
        )
        panels.append(np.ascontiguousarray(panel))
    writer.add_video(
        tag % clip_id,
        _as_video(np.stack(panels, axis=0)),
        step,
        fps=8,
    )


def _log_train_preview(writer, model, loader, device, pin_memory, step):
    batch = next(iter(loader))
    clips = batch[0]
    all_maps = batch[2]
    count = min(8, int(clips.shape[0]))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch_clips = clips[:count].to(device, non_blocking=pin_memory)
        preds = model(batch_clips.permute(0, 2, 1, 3, 4)).cpu()
    if was_training:
        model.train()
    for index in range(count):
        _log_comparison_video(
            writer,
            str(index),
            clips[index],
            all_maps[index],
            preds[index],
            step,
        )
    writer.flush()


def _log_pretrained_validation(writer, dataset, device, pin_memory, checkpoint_path, indexes):
    """Log the initial checkpoint on the fixed validation clips. The training model is left unchanged."""
    if checkpoint_path in ("", "None") or not os.path.isfile(checkpoint_path):
        print("Pretrained validation videos skipped; checkpoint not found:", checkpoint_path)
        return
    pretrained = _build_saliency_model(args)
    _load_saliency_checkpoint(pretrained, checkpoint_path)
    pretrained.to(device)
    pretrained.eval()
    try:
        with torch.no_grad():
            for index in indexes:
                clip, _gt, all_maps = dataset[index]
                pred = pretrained(
                    clip.unsqueeze(0).to(device, non_blocking=pin_memory).permute(0, 2, 1, 3, 4)
                )
                _log_comparison_video(
                    writer,
                    str(index),
                    clip,
                    all_maps,
                    pred[0].cpu(),
                    0,
                    tag="pretrained/%s",
                )
        writer.flush()
    finally:
        del pretrained
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("Logged pretrained validation videos from", checkpoint_path)


def _save_training_state(
    path,
    model,
    optimizer,
    epoch,
    best_loss,
    run_name,
    val_cc=None,
    val_sim=None,
    scheduler=None,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "val_cc": None if val_cc is None else float(val_cc),
        "val_sim": None if val_sim is None else float(val_sim),
        "run_name": run_name,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def _step_lr_scheduler(scheduler, metric):
    if scheduler is None:
        return
    prev_lr = scheduler.optimizer.param_groups[0]["lr"]
    scheduler.step(metric)
    new_lr = scheduler.optimizer.param_groups[0]["lr"]
    if new_lr != prev_lr:
        print("LR {:.3e} -> {:.3e} (ReduceLROnPlateau)".format(prev_lr, new_lr))


def _drop_media(event):
    if event.file_version:
        return None, 0
    if not event.summary.value:
        return event, 0
    kept = []
    dropped = 0
    for value in event.summary.value:
        plugin = ""
        if value.HasField("metadata") and value.metadata.HasField("plugin_data"):
            plugin = value.metadata.plugin_data.plugin_name
        if value.HasField("image") or plugin in ("images", "videos"):
            dropped += 1
            continue
        kept.append(value)
    if dropped == 0:
        return event, 0
    if not kept:
        return None, dropped
    from tensorboard.compat.proto.event_pb2 import Event

    clone = Event()
    clone.wall_time = event.wall_time
    clone.step = event.step
    clone.summary.CopyFrom(event.summary)
    del clone.summary.value[:]
    clone.summary.value.extend(kept)
    return clone, dropped


def _clear_tensorboard_media(logdir):
    if not os.path.isdir(logdir):
        print("TensorBoard directory not found:", logdir)
        sys.exit(1)
    paths = sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*")))
    if not paths:
        print("No TensorBoard event files in", logdir)
        return
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    kept_events = []
    removed = 0
    for path in paths:
        for event in EventFileLoader(path).Load():
            filtered, dropped = _drop_media(event)
            removed += dropped
            if filtered is not None:
                kept_events.append(filtered)
    temp_dir = logdir.rstrip("/") + ".clearing"
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    writer = EventFileWriter(temp_dir)
    try:
        for event in kept_events:
            writer.add_event(event)
    finally:
        writer.close()
    for path in paths:
        os.remove(path)
    for name in os.listdir(temp_dir):
        os.rename(os.path.join(temp_dir, name), os.path.join(logdir, name))
    os.rmdir(temp_dir)
    print("Removed %d image/video summaries from %s" % (removed, logdir))


def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()
    
    total_loss = AverageMeter()
    cur_loss = AverageMeter()
    total_cc = AverageMeter()
    total_sim = AverageMeter()

    for idx, sample in enumerate(loader):
        img_clips = sample[0]
        gt_sal = sample[1]

        img_clips = img_clips.to(device, non_blocking=args.pin_memory)
        img_clips = img_clips.permute((0,2,1,3,4))
        gt_sal = gt_sal.to(device, non_blocking=args.pin_memory)
        
        optimizer.zero_grad()

        pred_sal = model(img_clips)

        assert pred_sal.size() == gt_sal.size()

        loss = loss_func(pred_sal, gt_sal, args)
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
        cur_loss.update(loss.item())
        if args.dataset == "outplay":
            with torch.no_grad():
                total_cc.update(cc(pred_sal.detach(), gt_sal.detach()).item())
                total_sim.update(similarity(pred_sal.detach(), gt_sal.detach()).item())

        if idx%args.log_interval==(args.log_interval-1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss.avg, (time.time()-tic)/60))
            cur_loss.reset()
            sys.stdout.flush()
    
    time_taken = (time.time()-tic)/60
    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss.avg))
    sys.stdout.flush()

    # return total_loss.avg

    data_to_log = {
        'epoch': epoch + 1,
        'train_loss': total_loss.avg,
        'lr': optimizer.param_groups[0]['lr'],
        'train_time': time_taken
    }
    if args.dataset == "outplay":
        data_to_log['cc'] = total_cc.avg
        data_to_log['sim'] = total_sim.avg

    return data_to_log

def validate(model, loader, epoch, device, args, tb_writer=None, image_indexes=()):
    model.eval()
    tic = time.time()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    tic = time.time()
    image_index_set = set(image_indexes)
    for idx, sample in enumerate(loader):
        img_clips = sample[0]
        gt_sal = sample[1]

        img_clips = img_clips.to(device, non_blocking=args.pin_memory)
        img_clips = img_clips.permute((0,2,1,3,4))
        

        pred_sal = model(img_clips)
        if tb_writer is not None and idx in image_index_set:
            _log_comparison_video(
                tb_writer,
                str(idx),
                sample[0][0],
                sample[2][0],
                pred_sal[0],
                epoch,
            )
        
        gt_sal = gt_sal.squeeze(0).numpy()

        pred_sal = pred_sal.cpu().squeeze(0).numpy()
        pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
        pred_sal = torch.FloatTensor(pred_sal).unsqueeze(0).cuda()

        gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

        assert pred_sal.size() == gt_sal.size()

        loss = loss_func(pred_sal, gt_sal, args)
        cc_loss = cc(pred_sal, gt_sal)
        sim_loss = similarity(pred_sal, gt_sal)

        total_loss.update(loss.item())
        total_cc_loss.update(cc_loss.item())
        total_sim_loss.update(sim_loss.item())
    
    time_taken = (time.time()-tic)/60

    print('[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f} sim_loss : {:.5f}, time : {:3f}'.format(epoch, total_loss.avg, total_cc_loss.avg, total_sim_loss.avg, time_taken))
    sys.stdout.flush()

    # return total_loss.avg

    data_to_log = {
        'val_loss': total_loss.avg,
        'val_cc_loss': total_cc_loss.avg,
        'val_sim_loss': total_sim_loss.avg,
        'time': time_taken
    }

    return data_to_log

def _run_outplay(model, optimizer, train_loader, val_loader, device, args, resume_state, scheduler=None):
    if args.save_every < 1:
        print("save_every must be >= 1")
        sys.exit(1)
    missing_indexes = [index for index in OUTPLAY_VAL_IMAGE_INDEXES if index >= len(val_loader.dataset)]
    if missing_indexes:
        print("Validation image indexes are outside the manifest:", missing_indexes)
        sys.exit(1)

    import mlflow
    from torch.utils.tensorboard import SummaryWriter

    if resume_state is None:
        start_epoch = 0
        best_loss = float("inf")
    else:
        completed = resume_state["completed_epoch"]
        start_epoch = 0 if completed is None else completed + 1
        best_loss = resume_state["best_loss"]
        if start_epoch >= args.no_epochs:
            print(
                "Checkpoint finished epoch %d and --no_epochs is %d. Raise --no_epochs to continue."
                % (start_epoch - 1, args.no_epochs)
            )
            sys.exit(1)
        if not resume_state["has_optimizer"] and completed is not None:
            print("Continuing at epoch %d. This checkpoint has weights only, so Adam starts fresh." % start_epoch)
        if best_loss == float("inf"):
            print("Best loss was not stored; best.pt updates when validation improves")

    mlflow.set_tracking_uri(OUTPLAY_MLFLOW_URI)
    mlflow.set_experiment(OUTPLAY_MLFLOW_EXPERIMENT)
    if resume_state is None:
        run_context = mlflow.start_run()
    else:
        run_context = mlflow.start_run(run_id=_mlflow_run_id(resume_state["run_dir"]))

    with run_context as run:
        if resume_state is None:
            utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = run.info.run_id
            run_name = "%s_%s" % (utc, run_id)
            run_dir = os.path.join(OUTPLAY_RUNS_ROOT, run_name)
            tb_dir = os.path.join(OUTPLAY_TB_ROOT, run_name)
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            config_dir = os.path.join(run_dir, "config")
            tb_train = os.path.join(tb_dir, "train")
            tb_val = os.path.join(tb_dir, "val")
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(config_dir, exist_ok=True)
            payload = dict(vars(args))
            payload["mlflow_run_id"] = run_id
            payload["mlflow_experiment"] = OUTPLAY_MLFLOW_EXPERIMENT
            payload["mlflow_tracking_uri"] = OUTPLAY_MLFLOW_URI
            payload["run_dir"] = run_dir
            payload["tensorboard_train"] = tb_train
            payload["tensorboard_val"] = tb_val
            with open(os.path.join(config_dir, "args.json"), "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            mlflow.log_params({key: str(value) for key, value in vars(args).items()})
        else:
            run_dir = resume_state["run_dir"]
            run_name = os.path.basename(run_dir.rstrip("/"))
            tb_dir = os.path.join(OUTPLAY_TB_ROOT, run_name)
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            config_dir = os.path.join(run_dir, "config")
            tb_train = os.path.join(tb_dir, "train")
            tb_val = os.path.join(tb_dir, "val")
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "args.json")
            payload = {}
            if os.path.isfile(config_path):
                with open(config_path) as handle:
                    payload = json.load(handle)
            payload["resumed_from"] = resume_state["checkpoint_path"]
            payload["resumed_at_epoch"] = start_epoch
            with open(config_path, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            if args.clear_tensorboard_media:
                _clear_tensorboard_media(tb_train)
                _clear_tensorboard_media(tb_val)
            print("Continuing MLflow run", run.info.run_id, "at epoch", start_epoch)

        print("Run directory:", run_dir)
        print("TensorBoard directory:", tb_dir)
        train_writer = None
        val_writer = None
        try:
            train_writer = SummaryWriter(log_dir=tb_train)
            val_writer = SummaryWriter(log_dir=tb_val)
            _log_pretrained_validation(
                val_writer,
                val_loader.dataset,
                device,
                args.pin_memory,
                args.checkpoint_path,
                OUTPLAY_VAL_IMAGE_INDEXES,
            )
            _log_train_preview(train_writer, model, train_loader, device, args.pin_memory, start_epoch)
            for epoch in range(start_epoch, args.no_epochs):
                data_to_log_train = train(model, optimizer, train_loader, epoch, device, args)
                with torch.no_grad():
                    data_to_log_val = validate(
                        model,
                        val_loader,
                        epoch,
                        device,
                        args,
                        tb_writer=val_writer,
                        image_indexes=OUTPLAY_VAL_IMAGE_INDEXES,
                    )

                train_writer.add_scalar("loss", data_to_log_train["train_loss"], epoch)
                train_writer.add_scalar("cc", data_to_log_train["cc"], epoch)
                train_writer.add_scalar("sim", data_to_log_train["sim"], epoch)
                train_writer.add_scalar("lr", data_to_log_train["lr"], epoch)
                val_writer.add_scalar("loss", data_to_log_val["val_loss"], epoch)
                val_writer.add_scalar("cc", data_to_log_val["val_cc_loss"], epoch)
                val_writer.add_scalar("sim", data_to_log_val["val_sim_loss"], epoch)
                train_writer.flush()
                val_writer.flush()

                mlflow.log_metric("train_loss", data_to_log_train["train_loss"], step=epoch)
                mlflow.log_metric("train_cc", data_to_log_train["cc"], step=epoch)
                mlflow.log_metric("train_sim", data_to_log_train["sim"], step=epoch)
                mlflow.log_metric("lr", data_to_log_train["lr"], step=epoch)
                mlflow.log_metric("val_loss", data_to_log_val["val_loss"], step=epoch)
                mlflow.log_metric("val_cc", data_to_log_val["val_cc_loss"], step=epoch)
                mlflow.log_metric("val_sim", data_to_log_val["val_sim_loss"], step=epoch)

                val_loss = data_to_log_val["val_loss"]
                val_cc = data_to_log_val["val_cc_loss"]
                val_sim = data_to_log_val["val_sim_loss"]
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    _save_training_state(
                        best_path,
                        model,
                        optimizer,
                        epoch,
                        best_loss,
                        run_name,
                        val_cc=val_cc,
                        val_sim=val_sim,
                        scheduler=scheduler,
                    )
                    print("[{:2d}, save, {}]".format(epoch, best_path))
                if (epoch + 1) % args.save_every == 0:
                    epoch_path = os.path.join(ckpt_dir, "epoch_%d.pt" % epoch)
                    _save_training_state(
                        epoch_path,
                        model,
                        optimizer,
                        epoch,
                        best_loss,
                        run_name,
                        val_cc=val_cc,
                        val_sim=val_sim,
                        scheduler=scheduler,
                    )
                    _save_training_state(
                        os.path.join(ckpt_dir, "last.pt"),
                        model,
                        optimizer,
                        epoch,
                        best_loss,
                        run_name,
                        val_cc=val_cc,
                        val_sim=val_sim,
                        scheduler=scheduler,
                    )
                    print("[{:2d}, save, {}]".format(epoch, epoch_path))

                if args.use_wandb:
                    logged = {**data_to_log_train, **data_to_log_val}
                    print(logged)
                    wandb.log(logged)
                _step_lr_scheduler(scheduler, val_loss)
        finally:
            if train_writer is not None:
                train_writer.close()
            if val_writer is not None:
                val_writer.close()


summary(model, (3, args.clip_size, 224, 384), args.batch_size)

if args.dataset == "outplay":
    _run_outplay(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        args,
        resume_state,
        scheduler=scheduler,
    )
else:
    best_model = None
    for epoch in range(0, args.no_epochs):
        data_to_log_train = train(model, optimizer, train_loader, epoch, device, args)

        with torch.no_grad():
            data_to_log_val = validate(model, val_loader, epoch, device, args)

            val_loss = data_to_log_val['val_loss']
            if epoch == 0 :
                val_loss = np.inf
                best_loss = val_loss
            if val_loss <= best_loss:
                best_loss = val_loss
                best_model = model

                os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

                print('[{:2d},  save, {}]'.format(epoch, args.model_save_path))
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), args.model_save_path)
                else:
                    torch.save(model.state_dict(), args.model_save_path)

        data_to_log = {**data_to_log_train, **data_to_log_val}

        if args.use_wandb:
            print(data_to_log)
            wandb.log(data_to_log)

        _step_lr_scheduler(scheduler, data_to_log_val["val_loss"])

if args.use_wandb:
    wandb.finish()
