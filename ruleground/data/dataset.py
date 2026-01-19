"""
SportR Dataset

PyTorch Dataset implementation for the SportR benchmark.
Supports video loading via decord with fallback to torchvision,
predicate label loading from extraction pipeline, and
Q1/Q2/Q5 annotation parsing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _load_video_decord(path: str, num_frames: int) -> Tensor:
    """Load video using decord (fast C++ backend)."""
    from decord import VideoReader, cpu
    import decord

    decord.bridge.set_bridge("torch")

    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    indices = torch.linspace(0, total - 1, num_frames).long().tolist()
    frames = vr.get_batch(indices)  # [T, H, W, C] uint8
    frames = frames.float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    return frames


def _load_video_torchvision(path: str, num_frames: int) -> Tensor:
    """Fallback video loader using torchvision."""
    import torchvision

    video, _, info = torchvision.io.read_video(path, pts_unit="sec")
    # video: [T, H, W, C] uint8
    total = video.shape[0]
    indices = torch.linspace(0, total - 1, num_frames).long()
    frames = video[indices].float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    return frames


def load_video(path: str, num_frames: int = 16) -> Tensor:
    """Load video with decord, falling back to torchvision.

    Args:
        path:       Path to video file.
        num_frames: Number of frames to sample uniformly.

    Returns:
        Video tensor [T, C, H, W] in [0, 1] float range.
    """
    try:
        return _load_video_decord(path, num_frames)
    except (ImportError, Exception) as e:
        logger.debug(f"decord failed ({e}), falling back to torchvision")
        return _load_video_torchvision(path, num_frames)


class SportRDataset(Dataset):
    """PyTorch Dataset for SportR benchmark.

    Expected directory structure:
        root/
        ├── annotations/
        │   ├── train.json
        │   ├── val.json
        │   └── test.json
        └── videos/
            ├── <video_id>.mp4
            └── ...

    Annotation format (per sample):
        {
            "video_id": str,
            "sport": str,
            "q1_label": int,       # 0=no infraction, 1=infraction
            "q2_label": int,       # foul class index
            "q5_span": [float, float] | null,  # temporal span [start, end]
            "rationale": str | null
        }

    Args:
        root:            Path to SportR dataset root.
        split:           'train', 'val', or 'test'.
        num_frames:      Number of frames to sample per clip.
        transform:       Optional video transform callable.
        predicate_path:  Path to extracted predicate JSON.
        include_rationale: Whether to include rationale text.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        predicate_path: Optional[str | Path] = None,
        include_rationale: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.include_rationale = include_rationale

        # Load annotations
        anno_path = self.root / "annotations" / f"{split}.json"
        if anno_path.exists():
            with open(anno_path) as f:
                self.annotations = json.load(f)
        else:
            logger.warning(f"Annotation file not found: {anno_path}")
            self.annotations = []

        # Load predicate labels if available
        self.predicates: Dict[str, Any] = {}
        if predicate_path:
            pred_path = Path(predicate_path)
            if pred_path.exists():
                with open(pred_path) as f:
                    self.predicates = json.load(f)
                logger.info(f"Loaded predicates for {len(self.predicates)} videos")

        logger.info(
            f"SportRDataset: split={split}, samples={len(self.annotations)}, "
            f"predicates={'yes' if self.predicates else 'no'}"
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        anno = self.annotations[idx]
        video_id = anno["video_id"]

        # Load video
        video_path = str(self.root / "videos" / f"{video_id}.mp4")
        video = load_video(video_path, self.num_frames)

        # Apply transforms
        if self.transform is not None:
            video = self.transform(video)

        item: Dict[str, Any] = {
            "video": video,
            "video_id": video_id,
            "q1_label": anno["q1_label"],
            "q2_label": anno["q2_label"],
            "sport": anno["sport"],
        }

        # Q5 temporal span
        if "q5_span" in anno and anno["q5_span"] is not None:
            item["q5_span"] = anno["q5_span"]
        else:
            item["q5_span"] = None

        # Predicate labels from extraction
        if video_id in self.predicates:
            item["predicates"] = self.predicates[video_id]
        else:
            item["predicates"] = None

        # Rationale text (for extraction/debugging)
        if self.include_rationale and "rationale" in anno:
            item["rationale"] = anno["rationale"]

        return item


def build_dataloader(
    root: str,
    split: str = "train",
    num_frames: int = 16,
    batch_size: int = 16,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    predicate_path: Optional[str] = None,
) -> torch.utils.data.DataLoader:
    """Convenience factory for building a SportR DataLoader.

    Args:
        root:           Dataset root path.
        split:          'train', 'val', or 'test'.
        num_frames:     Frames to sample per clip.
        batch_size:     Batch size.
        num_workers:    DataLoader workers.
        transform:      Video transform.
        predicate_path: Path to extracted predicates.

    Returns:
        Configured DataLoader.
    """
    from ruleground.data.collate import sportr_collate_fn

    dataset = SportRDataset(
        root=root,
        split=split,
        num_frames=num_frames,
        transform=transform,
        predicate_path=predicate_path,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=sportr_collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
