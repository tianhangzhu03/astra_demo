from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GrabState(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    GRAB = "GRAB"


@dataclass
class GrabContext:
    state: GrabState = GrabState.IDLE
    top_pinch_state: bool = False

    hover_key: int = 0
    armed_key: int = 0
    grab_key: int = 0

    top_enter_count: int = 0
    grab_enter_count: int = 0
    exit_count: int = 0
    pinch_missing_count: int = 0


@dataclass
class GrabOutput:
    context: GrabContext
    trigger_on: bool
    target_key: int


def _update_pinch_state(prev: bool, pinch_dist: Optional[float], enter: float, exit_: float) -> bool:
    if pinch_dist is None:
        return False
    if prev:
        return pinch_dist < exit_
    return pinch_dist < enter


def update_grab_state(
    ctx: GrabContext,
    pinch_dist: Optional[float],
    hover_key: int,
    pinch_enter: float,
    pinch_exit: float,
    enter_frames: int,
    exit_frames: int,
    pinch_missing_hold_frames: int = 0,
    top_hand_present: Optional[bool] = None,
) -> GrabOutput:
    next_ctx = GrabContext(**ctx.__dict__)
    next_ctx.hover_key = hover_key
    top_present = (hover_key > 0) if top_hand_present is None else bool(top_hand_present)

    if pinch_dist is None:
        hold_missing = (
            ctx.top_pinch_state
            and ctx.state in (GrabState.ARMED, GrabState.GRAB)
            and ctx.pinch_missing_count < pinch_missing_hold_frames
        )
        next_ctx.top_pinch_state = hold_missing
        next_ctx.pinch_missing_count = ctx.pinch_missing_count + 1 if hold_missing else 0
    else:
        next_ctx.top_pinch_state = _update_pinch_state(
            prev=ctx.top_pinch_state,
            pinch_dist=pinch_dist,
            enter=pinch_enter,
            exit_=pinch_exit,
        )
        next_ctx.pinch_missing_count = 0

    if next_ctx.state == GrabState.IDLE:
        if next_ctx.top_pinch_state and top_present:
            next_ctx.top_enter_count += 1
        else:
            next_ctx.top_enter_count = 0

        if next_ctx.top_enter_count >= enter_frames:
            next_ctx.state = GrabState.ARMED
            next_ctx.armed_key = hover_key if hover_key > 0 else 0
            next_ctx.top_enter_count = 0
            next_ctx.grab_enter_count = 0
            next_ctx.exit_count = 0

    elif next_ctx.state == GrabState.ARMED:
        if not next_ctx.top_pinch_state or (not top_present):
            next_ctx.state = GrabState.IDLE
            next_ctx.armed_key = 0
            next_ctx.grab_enter_count = 0
            next_ctx.exit_count = 0
            next_ctx.grab_key = 0
        else:
            if hover_key > 0:
                next_ctx.armed_key = hover_key
            next_ctx.grab_enter_count += 1

            if next_ctx.grab_enter_count >= enter_frames:
                next_ctx.state = GrabState.GRAB
                next_ctx.grab_key = next_ctx.armed_key if next_ctx.armed_key > 0 else hover_key
                next_ctx.grab_enter_count = 0
                next_ctx.exit_count = 0

    else:  # GRAB
        should_release = not next_ctx.top_pinch_state
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

    trigger_on = next_ctx.state == GrabState.GRAB
    target_key = next_ctx.grab_key if (trigger_on and next_ctx.grab_key > 0) else 0
    return GrabOutput(
        context=next_ctx,
        trigger_on=trigger_on,
        target_key=target_key,
    )
