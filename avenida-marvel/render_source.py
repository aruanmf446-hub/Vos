#!/usr/bin/env python3
"""Animate edited keyframes using motion measured only in the source video.

Example:
  python render.py --indices 10,135,144,205,301,378 --output work/samples
  python render.py

Every source frame is named NNNN.png. Edited keyframes use the same names in
work/anchors. Hard cuts are the first frame of each new shot, never crossed by
optical flow. Source-adjacent bidirectional DIS fields are cached on disk.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from pathlib import Path
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def numbers(value: str) -> list[int]:
    return sorted(set(int(x.strip()) for x in value.split(",") if x.strip()))


def remap(array: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return cv2.remap(
        array, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


class SourceMotion:
    def __init__(self, source: Path, cache: Path, flow_width: int, preset: str):
        self.source = source
        self.cache = cache / f"dis-{preset}-{flow_width}-v1"
        self.cache.mkdir(parents=True, exist_ok=True)
        first = self.read(0)
        self.height, self.width = first.shape[:2]
        self.fw = flow_width
        self.fh = round(self.height * flow_width / self.width)
        self.small_size = (self.fw, self.fh)
        self.full_size = (self.width, self.height)
        yy, xx = np.mgrid[:self.fh, :self.fw].astype(np.float32)
        self.grid = np.dstack([xx, yy])
        yy, xx = np.mgrid[:self.height, :self.width].astype(np.float32)
        self.full_grid = np.dstack([xx, yy])
        self.grays: dict[int, np.ndarray] = {}
        self.pairs: OrderedDict[int, tuple[np.ndarray, ...]] = OrderedDict()
        presets = {
            "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        self.dis = cv2.DISOpticalFlow_create(presets[preset])
        self.dis.setUseSpatialPropagation(True)
        self.dis.setUseMeanNormalization(True)
        self.dis.setVariationalRefinementIterations(5)

    def read(self, index: int) -> np.ndarray:
        path = self.source / f"{index:04d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return image

    def gray(self, index: int) -> np.ndarray:
        if index not in self.grays:
            image = cv2.cvtColor(self.read(index), cv2.COLOR_BGR2GRAY)
            self.grays[index] = cv2.resize(image, self.small_size, interpolation=cv2.INTER_AREA)
        return self.grays[index]

    def validity(self, mapping: np.ndarray) -> np.ndarray:
        return (
            (mapping[..., 0] >= 0) & (mapping[..., 0] <= self.fw - 1)
            & (mapping[..., 1] >= 0) & (mapping[..., 1] <= self.fh - 1)
        ).astype(np.float32)

    def consistent(self, flow: np.ndarray, reverse: np.ndarray) -> np.ndarray:
        mapping = self.grid + flow
        residual = np.linalg.norm(flow + remap(reverse, mapping), axis=2)
        allowance = 1.2 + 0.05 * np.linalg.norm(flow, axis=2)
        return np.maximum(1e-4, np.exp(-residual / allowance) * self.validity(mapping))

    def pair(self, index: int) -> tuple[np.ndarray, ...]:
        """Return index->index+1, reverse, and their forward/backward confidence."""
        if index in self.pairs:
            self.pairs.move_to_end(index)
            return self.pairs[index]
        path = self.cache / f"{index:04d}.npz"
        if path.exists():
            with np.load(path) as saved:
                forward = saved["forward"].astype(np.float32)
                backward = saved["backward"].astype(np.float32)
        else:
            a, b = self.gray(index), self.gray(index + 1)
            forward = self.dis.calc(a, b, None)
            backward = self.dis.calc(b, a, None)
            temp = path.with_suffix(".partial.npz")
            np.savez_compressed(temp, forward=forward.astype(np.float16), backward=backward.astype(np.float16))
            temp.replace(path)
        result = (
            forward, backward,
            self.consistent(forward, backward),
            self.consistent(backward, forward),
        )
        self.pairs[index] = result
        while len(self.pairs) > 48:
            self.pairs.popitem(last=False)
        return result

    def anchor_maps(self, anchor: int, targets: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Build backward target->anchor maps by composing source-adjacent flow.

        G(t->a)(x) = f(t->p)(x) + G(p->a)(x + f(t->p)(x)).
        Confidence is the geometric mean of adjacent forward/backward checks,
        then weighted by agreement of the warped original anchor with source t.
        """
        wanted = set(targets)
        maps = {}
        for direction in (-1, 1):
            selected = [t for t in targets if (t - anchor) * direction > 0]
            if not selected:
                continue
            last = min(selected) if direction == -1 else max(selected)
            accumulated = np.zeros_like(self.grid)
            log_confidence = np.zeros((self.fh, self.fw), np.float32)
            steps = 0
            for current in range(anchor + direction, last + direction, direction):
                if direction == 1:
                    _, step, _, reliability = self.pair(current - 1)
                else:
                    step, _, reliability, _ = self.pair(current)
                mapping = self.grid + step
                accumulated = step + remap(accumulated, mapping)
                log_confidence = np.log(reliability) + remap(log_confidence, mapping)
                steps += 1
                if current in wanted:
                    anchor_mapping = self.grid + accumulated
                    photo_error = np.abs(
                        remap(self.gray(anchor), anchor_mapping).astype(np.float32)
                        - self.gray(current).astype(np.float32)
                    )
                    photo_error = cv2.GaussianBlur(photo_error, (5, 5), 0)
                    confidence = np.exp(log_confidence / steps) * np.exp(-photo_error / 32.0)
                    confidence = (0.12 + 0.88 * confidence) * self.validity(anchor_mapping)
                    maps[current] = (accumulated.copy(), confidence)
        if anchor in wanted:
            maps[anchor] = (np.zeros_like(self.grid), np.ones((self.fh, self.fw), np.float32))
        return maps

    def warp(self, edited: np.ndarray, flow: np.ndarray) -> np.ndarray:
        full = cv2.resize(flow, self.full_size, interpolation=cv2.INTER_LINEAR)
        full[..., 0] *= self.width / self.fw
        full[..., 1] *= self.height / self.fh
        return remap(edited, self.full_grid + full)

