"""
Video Transforms for SportR

Augmentations and preprocessing for video clips.
Operates on tensors of shape [T, C, H, W].
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


class VideoTransform:
    """Composable video transform pipeline."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, video: Tensor) -> Tensor:
        for t in self.transforms:
            video = t(video)
        return video


class Resize:
    """Resize frames to target size."""

    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size

    def __call__(self, video: Tensor) -> Tensor:
        # video: [T, C, H, W]
        return F.interpolate(video, size=self.size, mode="bilinear", align_corners=False)


class Normalize:
    """ImageNet normalization."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, video: Tensor) -> Tensor:
        mean = self.mean.to(video.device, video.dtype)
        std = self.std.to(video.device, video.dtype)
        return (video - mean) / std


class RandomHorizontalFlip:
    """Randomly flip all frames horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, video: Tensor) -> Tensor:
        if random.random() < self.p:
            return video.flip(-1)
        return video


class TemporalCrop:
    """Randomly crop a contiguous temporal window."""

    def __init__(self, num_frames: int):
        self.num_frames = num_frames

    def __call__(self, video: Tensor) -> Tensor:
        T = video.shape[0]
        if T <= self.num_frames:
            return video
        start = random.randint(0, T - self.num_frames)
        return video[start : start + self.num_frames]


class ColorJitter:
    """Simple color jitter for video frames."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, video: Tensor) -> Tensor:
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            video = video * factor

        if self.contrast > 0:
            mean = video.mean(dim=(-2, -1), keepdim=True)
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            video = (video - mean) * factor + mean

        return video.clamp(0, 1)


class UniformTemporalSubsample:
    """Sample num_frames evenly from the video."""

    def __init__(self, num_frames: int = 16):
        self.num_frames = num_frames

    def __call__(self, video: Tensor) -> Tensor:
        T = video.shape[0]
        if T == self.num_frames:
            return video
        indices = torch.linspace(0, T - 1, self.num_frames).long()
        return video[indices]


def get_train_transform(
    frame_size: Tuple[int, int] = (224, 224),
    num_frames: int = 16,
) -> VideoTransform:
    """Standard training transforms."""
    return VideoTransform([
        UniformTemporalSubsample(num_frames),
        Resize(frame_size),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2),
        Normalize(),
    ])


def get_eval_transform(
    frame_size: Tuple[int, int] = (224, 224),
    num_frames: int = 16,
) -> VideoTransform:
    """Standard evaluation transforms (no augmentation)."""
    return VideoTransform([
        UniformTemporalSubsample(num_frames),
        Resize(frame_size),
        Normalize(),
    ])
