import argparse
import datetime
import json
import os
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
from utils import *

OUTPLAY_MLFLOW_URI = "http://104.105.26.182:5000"
OUTPLAY_MLFLOW_EXPERIMENT = "sports-saliency-ViNet-S"
OUTPLAY_SPLIT_ROOT = "/home/nitesh_sawadia/mnt/ml-fanpulse/sports_saliency/vinet_outplay/split"
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
parser.add_argument('--save_every', default=1, type=int)



args = parser.parse_args()

if args.dataset == "outplay":
    args.grouped_conv = True
    args.root_grouping = True
    args.depth_grouping = False
    args.efficientnet = False
    args.num_hier = 3
    args.clip_size = 32

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


model = VideoSaliencyModel(
    use_upsample=bool(args.decoder_upsample),
    num_hier=args.num_hier,
    num_clips=args.clip_size,
    grouped_conv=args.grouped_conv,
    root_grouping=args.root_grouping,
    depth=args.depth_grouping,
    efficientnet=args.efficientnet,
    BiCubic = not args.use_trilinear_upsampling,
)

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


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers, pin_memory=args.pin_memory)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers, pin_memory=args.pin_memory)


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


def _load_saliency_checkpoint(model, path):
    if not os.path.isfile(path):
        print("Saliency checkpoint not found: %s" % path)
        sys.exit(1)
    print("Loading pretrained saliency model weights:", path)
    state = _strip_module_prefix(_coerce_state_dict(torch.load(path, map_location="cpu")))
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
    print("Loaded saliency checkpoint with strict=True")


if args.checkpoint_path != "None":
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

print(device)

def _denormalize_frame(frame_chw):
    frame = frame_chw.detach().float().cpu().numpy()
    image = frame * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def _jet_overlay(frame_chw, sal_hw):
    frame = _denormalize_frame(frame_chw)
    sal = sal_hw.detach().float().cpu().numpy()
    height, width = frame.shape[1], frame.shape[2]
    if sal.shape != (height, width):
        sal = cv2.resize(sal, (width, height), interpolation=cv2.INTER_LINEAR)
    sal = np.clip(sal, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap(np.round(sal * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame_hwc = np.transpose(frame, (1, 2, 0))
    blended = (1.0 - _OVERLAY_ALPHA) * frame_hwc + _OVERLAY_ALPHA * heat
    return np.ascontiguousarray(np.transpose(np.clip(blended, 0.0, 1.0), (2, 0, 1)))


def _as_video(frames_tchw):
    # add_video wants (N, T, C, H, W). N=1 keeps one clip instead of a padded grid.
    return torch.from_numpy(np.ascontiguousarray(frames_tchw)).unsqueeze(0)


def _log_clip_video(writer, clip_id, frames, maps, step, overlay_kind):
    single_map = maps.dim() == 2
    rgb_frames = []
    overlay_frames = []
    for t in range(int(frames.shape[0])):
        frame = frames[t]
        saliency = maps if single_map else maps[t]
        rgb_frames.append(_denormalize_frame(frame))
        overlay_frames.append(_jet_overlay(frame, saliency))
    writer.add_video(
        "samples/%s/clip" % clip_id,
        _as_video(np.stack(rgb_frames, axis=0)),
        step,
        fps=8,
    )
    writer.add_video(
        "samples/%s/%s" % (clip_id, overlay_kind),
        _as_video(np.stack(overlay_frames, axis=0)),
        step,
        fps=8,
    )


def _log_train_gt_preview(writer, loader):
    batch = next(iter(loader))
    clips = batch[0]
    all_maps = batch[2]
    for index in range(min(8, int(clips.shape[0]))):
        _log_clip_video(writer, str(index), clips[index], all_maps[index], 0, "gt_overlay")
    writer.flush()


def _save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


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

        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0,2,1,3,4))
        gt_sal = gt_sal.to(device)
        
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

        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0,2,1,3,4))
        

        pred_sal = model(img_clips)
        if tb_writer is not None and idx in image_index_set:
            _log_clip_video(
                tb_writer,
                str(idx),
                sample[0][0],
                pred_sal[0],
                epoch,
                "pred_overlay",
            )
        
        gt_sal = gt_sal.squeeze(0).numpy()

        pred_sal = pred_sal.cpu().squeeze(0).numpy()
        pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
        pred_sal = blur(pred_sal).unsqueeze(0).cuda()

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

def _run_outplay(model, optimizer, train_loader, val_loader, device, args):
    if args.save_every < 1:
        print("save_every must be >= 1")
        sys.exit(1)
    missing_indexes = [index for index in OUTPLAY_VAL_IMAGE_INDEXES if index >= len(val_loader.dataset)]
    if missing_indexes:
        print("Validation image indexes are outside the manifest:", missing_indexes)
        sys.exit(1)

    import mlflow
    from torch.utils.tensorboard import SummaryWriter

    mlflow.set_tracking_uri(OUTPLAY_MLFLOW_URI)
    mlflow.set_experiment(OUTPLAY_MLFLOW_EXPERIMENT)
    utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with mlflow.start_run() as run:
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

        print("Run directory:", run_dir)
        print("TensorBoard directory:", tb_dir)
        train_writer = None
        val_writer = None
        best_loss = float("inf")
        try:
            train_writer = SummaryWriter(log_dir=tb_train)
            val_writer = SummaryWriter(log_dir=tb_val)
            _log_train_gt_preview(train_writer, train_loader)
            for epoch in range(0, args.no_epochs):
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
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    _save_checkpoint(model, best_path)
                    print("[{:2d}, save, {}]".format(epoch, best_path))
                if (epoch + 1) % args.save_every == 0:
                    epoch_path = os.path.join(ckpt_dir, "epoch_%d.pt" % epoch)
                    _save_checkpoint(model, epoch_path)
                    _save_checkpoint(model, os.path.join(ckpt_dir, "last.pt"))
                    print("[{:2d}, save, {}]".format(epoch, epoch_path))

                if args.use_wandb:
                    logged = {**data_to_log_train, **data_to_log_val}
                    print(logged)
                    wandb.log(logged)
                if args.lr_sched:
                    scheduler.step()
        finally:
            if train_writer is not None:
                train_writer.close()
            if val_writer is not None:
                val_writer.close()


summary(model, (3, args.clip_size, 224, 384), args.batch_size)

if args.dataset == "outplay":
    _run_outplay(model, optimizer, train_loader, val_loader, device, args)
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

        if args.lr_sched:
            scheduler.step()

if args.use_wandb:
    wandb.finish()
