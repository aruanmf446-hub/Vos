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
        self.dis.setVariationalRefinementIterations(3)

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


def blend_warps(a: np.ndarray, b: np.ndarray, ca: np.ndarray, cb: np.ndarray,
                fraction: float, size: tuple[int, int]) -> np.ndarray:
    """Blend reliable warps, narrowing transitions where generated details differ.

    A long dissolve ghosts independently generated faces. Large disagreements
    therefore use a short smooth transition near the interval midpoint, while
    matching regions retain a longer dissolve. Confidence suppresses occluded
    and poorly mapped regions; exact keyframes are always copied separately.
    """
    low_a = cv2.resize(a, (ca.shape[1], ca.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_b = cv2.resize(b, (cb.shape[1], cb.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    difference = np.mean(np.abs(low_a - low_b), axis=2) / 255.0
    difference = cv2.GaussianBlur(difference, (9, 9), 0)
    conflict = np.clip((difference - 0.025) / 0.16, 0.0, 1.0)
    half_window = 0.40 - 0.32 * conflict
    phase = np.clip((fraction - (0.5 - half_window)) / (2 * half_window), 0.0, 1.0)
    phase = phase * phase * (3.0 - 2.0 * phase)
    wa = np.maximum(1.0 - phase, 0.001) * ca
    wb = np.maximum(phase, 0.001) * cb
    denominator = wa + wb
    alpha = np.divide(wb, denominator, out=np.full_like(wb, fraction), where=denominator > 1e-8)
    alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)[..., None]
    mixed = a.astype(np.float32) * (1.0 - alpha) + b.astype(np.float32) * alpha
    return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "work/source")
    parser.add_argument("--anchors", type=Path, default=ROOT / "work/anchors")
    parser.add_argument("--output", type=Path, default=ROOT / "work/rendered")
    parser.add_argument("--cache", type=Path, default=ROOT / "work/flow")
    parser.add_argument("--cuts", type=numbers, default=[143, 294])
    parser.add_argument("--indices", type=numbers, help="Render only these comma-separated frame indices")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, help="Exclusive frame index")
    parser.add_argument("--flow-width", type=int, default=288)
    parser.add_argument("--flow-preset", choices=["ultrafast", "fast", "medium"], default="medium")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Overwrite rendered frames")
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    source_files = sorted(args.source.glob("[0-9][0-9][0-9][0-9].png"))
    if not source_files:
        parser.error(f"No source PNG frames in {args.source}")
    frame_count = int(source_files[-1].stem) + 1
    anchors = {int(p.stem): p for p in args.anchors.glob("[0-9][0-9][0-9][0-9].png")}
    if not anchors:
        parser.error(f"No edited keyframes in {args.anchors}")
    selected = args.indices if args.indices is not None else list(range(args.start, args.end or frame_count))
    if any(t < 0 or t >= frame_count for t in selected):
        parser.error("Requested index outside source frame range")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [t for t in selected if args.force or not (args.output / f"{t:04d}.png").exists()]
    if not selected:
        print("All requested frames already exist.", flush=True)
        return
    motion = SourceMotion(args.source, args.cache, args.flow_width, args.flow_preset)
    edited_cache: dict[int, np.ndarray] = {}

    def edited(index: int) -> np.ndarray:
        if index not in edited_cache:
            im = cv2.imread(str(anchors[index]), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(anchors[index])
            if im.shape[:2] != (motion.height, motion.width):
                im = cv2.resize(im, motion.full_size, interpolation=cv2.INTER_LANCZOS4)
            edited_cache[index] = im
        return edited_cache[index]

    cuts = [0] + [c for c in args.cuts if 0 < c < frame_count] + [frame_count]
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for shot_start, shot_end in zip(cuts, cuts[1:]):
        shot_indices = [t for t in selected if shot_start <= t < shot_end]
        if not shot_indices:
            continue
        shot_anchors = sorted(a for a in anchors if shot_start <= a < shot_end)
        if not shot_anchors:
            parser.error(f"Shot [{shot_start}, {shot_end}) has no edited anchor")
        for t in shot_indices:
            pos = bisect_left(shot_anchors, t)
            if pos < len(shot_anchors) and shot_anchors[pos] == t:
                pair = (t, t)
            elif pos == 0:
                pair = (shot_anchors[0], shot_anchors[0])
            elif pos == len(shot_anchors):
                pair = (shot_anchors[-1], shot_anchors[-1])
            else:
                pair = (shot_anchors[pos - 1], shot_anchors[pos])
            groups[pair].append(t)

    started = time.monotonic()
    done = 0
    for (a, b), targets in groups.items():
        print(f"Anchors {a:04d}/{b:04d}: {len(targets)} requested frames", flush=True)
        ma = motion.anchor_maps(a, targets)
        mb = motion.anchor_maps(b, targets) if b != a else ma
        for t in targets:
            if t == a:
                result = edited(a)
            elif t == b:
                result = edited(b)
            else:
                wa = motion.warp(edited(a), ma[t][0])
                if a == b:
                    result = wa
                else:
                    wb = motion.warp(edited(b), mb[t][0])
                    result = blend_warps(wa, wb, ma[t][1], mb[t][1], (t - a) / (b - a), motion.full_size)
            path = args.output / f"{t:04d}.png"
            if not cv2.imwrite(str(path), result, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                raise RuntimeError(f"Could not write {path}")
            done += 1
        print(f"Rendered {done}/{len(selected)} frames in {time.monotonic() - started:.1f}s", flush=True)
        # Images can be large; adjacent groups can reload an anchor inexpensively.
        edited_cache.clear()
    print(f"Finished: {args.output}", flush=True)


if __name__ == "__main__":
    main()
