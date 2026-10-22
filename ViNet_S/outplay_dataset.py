import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FRAME_COUNT = 32
# PIL size is (width, height). ViNet-S consumes 224x384.
FRAME_SIZE = (384, 224)
_TO_TENSOR = transforms.ToTensor()
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def frame_names():
    return ["%04d.png" % index for index in range(1, FRAME_COUNT + 1)]


def window_basename(manifest_dir):
    """Last path segment. The outputs/saliency_vinet prefix is not an S3 key."""
    text = str(manifest_dir).replace("\\", "/").rstrip("/")
    name = text.split("/")[-1]
    if name in ("", ".", ".."):
        raise ValueError("Outplay window dir has no basename: %s" % manifest_dir)
    return name


def window_path(split_root, window):
    missing = [key for key in ("split", "sport", "game", "dir") if key not in window]
    if missing:
        raise KeyError("Outplay window is missing %s: %s" % (missing, window))
    return os.path.join(
        split_root,
        window["split"],
        window["sport"],
        window["game"],
        window_basename(window["dir"]),
    )


def _require_window(path):
    if not os.path.isdir(path):
        raise FileNotFoundError("Outplay window is missing: %s" % path)
    expected = frame_names()
    for kind in ("images", "maps"):
        folder = os.path.join(path, kind)
        if not os.path.isdir(folder):
            raise FileNotFoundError("Outplay window is missing %s: %s" % (kind, path))
        present = set(os.listdir(folder))
        missing = [name for name in expected if name not in present]
        if missing:
            raise FileNotFoundError(
                "Outplay window %s does not have %d %s files; missing %s"
                % (path, FRAME_COUNT, kind, missing)
            )


def _open_resized(path, mode):
    image = Image.open(path).convert(mode)
    if image.size != FRAME_SIZE:
        image = image.resize(FRAME_SIZE, Image.BILINEAR)
    return image


def _load_frame(path):
    image = _open_resized(path, "RGB")
    return _NORMALIZE(_TO_TENSOR(image)).float()


def _load_map(path):
    image = _open_resized(path, "L")
    array = np.asarray(image, dtype=np.float32)
    peak = float(array.max()) if array.size else 0.0
    if peak > 1.0:
        array = array / 255.0
    return torch.from_numpy(np.ascontiguousarray(array))


class OutplayWindowDataset(Dataset):
    """One manifest window is one 32-frame clip.

    Returns (clip, gt, all_maps). gt is maps/0032.png and is the only map the
    loss reads. all_maps is maps/0001.png-0032.png for TensorBoard.
    """

    def __init__(self, split_root, split, clip_size=FRAME_COUNT):
        if clip_size != FRAME_COUNT:
            raise ValueError(
                "Outplay windows are %d frames, got clip_size=%s" % (FRAME_COUNT, clip_size)
            )
        if split not in ("train", "val"):
            raise ValueError("Outplay split must be 'train' or 'val', got %s" % split)

        manifest_path = os.path.join(split_root, "split_manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError("Outplay manifest not found: %s" % manifest_path)
        with open(manifest_path, "r") as handle:
            manifest = json.load(handle)
        windows = manifest.get("windows") if isinstance(manifest, dict) else None
        if not isinstance(windows, list):
            raise ValueError("Outplay manifest has no windows list: %s" % manifest_path)

        self.split = split
        self.paths = []
        for window in windows:
            if not isinstance(window, dict) or window.get("split") != split:
                continue
            path = window_path(split_root, window)
            _require_window(path)
            self.paths.append(path)
        if not self.paths:
            raise RuntimeError("Outplay split '%s' has no windows in %s" % (split, manifest_path))
        print("outplay", split, len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        frames = []
        maps = []
        for name in frame_names():
            frame_path = os.path.join(path, "images", name)
            map_path = os.path.join(path, "maps", name)
            if not os.path.isfile(frame_path) or not os.path.isfile(map_path):
                raise FileNotFoundError(
                    "Outplay window does not have 32 frames and 32 maps: %s" % path
                )
            frames.append(_load_frame(frame_path))
            maps.append(_load_map(map_path))

        clip = torch.stack(frames, dim=0).float()
        all_maps = torch.stack(maps, dim=0).float()
        gt = all_maps[-1].clone()
        if clip.shape != (FRAME_COUNT, 3, 224, 384):
            raise RuntimeError("Bad clip shape %s at %s" % (tuple(clip.shape), path))
        if all_maps.shape != (FRAME_COUNT, 224, 384) or gt.shape != (224, 384):
            raise RuntimeError("Bad map shape at %s" % path)
        if clip.dtype != torch.float32 or gt.dtype != torch.float32 or all_maps.dtype != torch.float32:
            raise RuntimeError("Outplay tensors must be float32 at %s" % path)
        return clip, gt, all_maps
