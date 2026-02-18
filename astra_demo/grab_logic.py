from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class GrabState(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    GRAB = "GRAB"


@dataclass
class GrabContext:
    state: GrabState = GrabState.IDLE
    top_pinch_state: bool = False
    depth_gate_state: bool = False

    hover_key: int = 0
    armed_key: int = 0
    grab_key: int = 0

    top_enter_count: int = 0
    depth_enter_count: int = 0
    exit_count: int = 0


@dataclass
class GrabOutput:
    context: GrabContext
    trigger_on: bool
    target_key: int
    depth_mm: Optional[int]


def sample_depth_5x5(depth_mm: np.ndarray, center_xy: Tuple[int, int]) -> Optional[int]:
    h, w = depth_mm.shape[:2]
    cx, cy = center_xy
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None

    x1 = max(0, cx - 2)
    x2 = min(w, cx + 3)
    y1 = max(0, cy - 2)
    y2 = min(h, cy + 3)

    window = depth_mm[y1:y2, x1:x2].astype(np.int32)
    valid = window[window > 0]
    if valid.size == 0:
        return None
    return int(np.median(valid))


def _update_pinch_state(prev: bool, pinch_dist: Optional[float], enter: float, exit_: float) -> bool:
    if pinch_dist is None:
        return False
    if prev:
        return pinch_dist < exit_
    return pinch_dist < enter


def _update_depth_gate_state(prev: bool, depth_mm: Optional[int], enter_mm: int, exit_mm: int) -> bool:
    if depth_mm is None or depth_mm <= 0:
        return False
    if prev:
        return depth_mm <= exit_mm
    return depth_mm <= enter_mm


def update_grab_state(
    ctx: GrabContext,
    pinch_dist: Optional[float],
    depth_at_mid_mm: Optional[int],
    hover_key: int,
    pinch_enter: float,
    pinch_exit: float,
    depth_enter_mm: int,
    depth_exit_mm: int,
    enter_frames: int,
    exit_frames: int,
) -> GrabOutput:
    next_ctx = GrabContext(**ctx.__dict__)
    next_ctx.hover_key = hover_key

    next_ctx.top_pinch_state = _update_pinch_state(
        prev=ctx.top_pinch_state,
        pinch_dist=pinch_dist,
        enter=pinch_enter,
        exit_=pinch_exit,
    )
    next_ctx.depth_gate_state = _update_depth_gate_state(
        prev=ctx.depth_gate_state,
        depth_mm=depth_at_mid_mm,
        enter_mm=depth_enter_mm,
        exit_mm=depth_exit_mm,
    )

    if next_ctx.state == GrabState.IDLE:
        if next_ctx.top_pinch_state and hover_key > 0:
            next_ctx.top_enter_count += 1
        else:
            next_ctx.top_enter_count = 0

        if next_ctx.top_enter_count >= enter_frames:
            next_ctx.state = GrabState.ARMED
            next_ctx.armed_key = hover_key
            next_ctx.top_enter_count = 0
            next_ctx.depth_enter_count = 0
            next_ctx.exit_count = 0

    elif next_ctx.state == GrabState.ARMED:
        if not next_ctx.top_pinch_state or hover_key == 0:
            next_ctx.state = GrabState.IDLE
            next_ctx.armed_key = 0
            next_ctx.depth_enter_count = 0
            next_ctx.exit_count = 0
            next_ctx.grab_key = 0
        else:
            next_ctx.armed_key = hover_key
            if next_ctx.depth_gate_state:
                next_ctx.depth_enter_count += 1
            else:
                next_ctx.depth_enter_count = 0

            if next_ctx.depth_enter_count >= enter_frames:
                next_ctx.state = GrabState.GRAB
                next_ctx.grab_key = next_ctx.armed_key if next_ctx.armed_key > 0 else hover_key
                next_ctx.depth_enter_count = 0
                next_ctx.exit_count = 0

    else:  # GRAB
        should_release = (not next_ctx.top_pinch_state) or (not next_ctx.depth_gate_state)
        if should_release:
            next_ctx.exit_count += 1
        else:
            next_ctx.exit_count = 0

        if next_ctx.exit_count >= exit_frames:
            next_ctx.state = GrabState.IDLE
            next_ctx.grab_key = 0
            next_ctx.armed_key = 0
            next_ctx.exit_count = 0

    if next_ctx.state != GrabState.GRAB:
        next_ctx.grab_key = 0

    trigger_on = next_ctx.state == GrabState.GRAB and next_ctx.grab_key > 0
    target_key = next_ctx.grab_key if trigger_on else 0
    return GrabOutput(
        context=next_ctx,
        trigger_on=trigger_on,
        target_key=target_key,
        depth_mm=depth_at_mid_mm,
    )
