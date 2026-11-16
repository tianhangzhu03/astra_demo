from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Optional


class GrabState(str, Enum):
    IDLE = 'IDLE'
    ARMED = 'ARMED'  # Operator prompt sent; waiting for a new nearby open-to-pinch gesture.
    GRAB = 'GRAB'


@dataclass
class GrabContext:
    state: GrabState = GrabState.IDLE
    open_count: int = 0
    open_observed: bool = False
    close_count: int = 0
    last_gesture_ms: Optional[int] = None
    exit_count: int = 0
    release_started_ms: Optional[int] = None


@dataclass
class GrabOutput:
    context: GrabContext
    trigger_on: bool
    target_key: int = 0  # Compatibility with existing session CSV; quadrants are reference only.
    status: str = 'WAIT CUE [G]'


def update_manual_grab(ctx, request, *, tracking_valid):
    """ABC operator confirmation; tracking loss never releases an already held ball."""
    if request == 'release':
        return GrabOutput(GrabContext(), False, status='RELEASED / G to grab')
    held = ctx.state == GrabState.GRAB
    if request == 'grab' and not held:
        if not tracking_valid:
            return GrabOutput(ctx, False, status='No valid position / press G again')
        return GrabOutput(GrabContext(state=GrabState.GRAB), True, status='HELD / G to drop')
    return GrabOutput(ctx, held, status='HELD / G to drop' if held else 'G to grab')


def near_ball(hand_xy, ball_xy, radius, margin=20.0):
    if hand_xy is None:
        return False
    values = (*hand_xy, *ball_xy, radius, margin)
    return (all(math.isfinite(v) for v in values) and radius > 0 and margin >= 0
            and math.dist(hand_xy, ball_xy) <= radius + margin)


def update_grab_state(
    ctx: GrabContext,
    pinch_dist: Optional[float],
    *,
    prompt_sent: bool = False,
    cancel_prompt: bool = False,
    hand_near_ball: bool = False,
    top_hand_present: bool = False,
    pinch_enter: float = .42,
    pinch_ready: float = .60,
    pinch_exit: float = .85,
    enter_frames: int = 2,
    exit_frames: int = 3,
    now_ms: int = 0,
    release_hold_ms: int = 350,
    gesture_gap_ms: int = 500,
    top_anchor_valid: bool = True,
    setup_ready: bool = True,
    manual_release: bool = False,
) -> GrabOutput:
    """Prompt + nearby opening/closure; bridge brief side-tip loss, never attach on it."""
    next_ctx = replace(ctx)
    valid_pinch = pinch_dist is not None and math.isfinite(pinch_dist)
    if manual_release:
        return GrabOutput(GrabContext(), False)

    # Once attached, proximity and operator permission no longer gate retention.
    if ctx.state == GrabState.GRAB:
        if valid_pinch and pinch_dist >= pinch_exit:
            if next_ctx.release_started_ms is None:
                next_ctx.release_started_ms = now_ms
            next_ctx.exit_count += 1
            if (next_ctx.exit_count >= exit_frames
                    and now_ms - next_ctx.release_started_ms >= release_hold_ms):
                return GrabOutput(GrabContext(), False)
        else:
            next_ctx.exit_count = 0
            next_ctx.release_started_ms = None
        return GrabOutput(next_ctx, True, status='HELD')

    if cancel_prompt:
        return GrabOutput(GrabContext(), False)
    if prompt_sent:
        next_ctx = GrabContext(state=GrabState.ARMED)
    if next_ctx.state == GrabState.IDLE:
        return GrabOutput(next_ctx, False)

    if not setup_ready or not top_hand_present or not top_anchor_valid or not hand_near_ball:
        # Cannot close remotely and carry that closed pose into the target area.
        next_ctx.open_count = next_ctx.close_count = 0
        next_ctx.open_observed = False
        next_ctx.last_gesture_ms = None
        status = ('PLACE SIDE BALL' if not setup_ready else
                  'SHOW TOP HAND' if not top_hand_present else
                  'SHOW TOP FINGERS' if not top_anchor_valid else 'MOVE TO BALL')
        return GrabOutput(next_ctx, False, status=status)

    # A folding hand can briefly hide its thumb/index tips in Side View. Keep
    # a CONFIRMED nearby opening for a bounded gap, but clear partial counts:
    # missing frames never count as closure and cannot complete an attachment.
    if (next_ctx.last_gesture_ms is not None
            and not 0 <= now_ms - next_ctx.last_gesture_ms <= gesture_gap_ms):
        next_ctx.open_count = next_ctx.close_count = 0
        next_ctx.open_observed = False
    if not valid_pinch:
        next_ctx.open_count = next_ctx.close_count = 0
        return GrabOutput(next_ctx, False, status='SHOW SIDE FINGERS')
    next_ctx.last_gesture_ms = now_ms

    if pinch_dist >= pinch_ready:
        next_ctx.open_count += 1
        next_ctx.close_count = 0
        if next_ctx.open_count >= enter_frames:
            next_ctx.open_observed = True
    else:
        next_ctx.open_count = 0
        if next_ctx.open_observed and pinch_dist <= pinch_enter:
            next_ctx.close_count += 1
            if next_ctx.close_count >= enter_frames:
                return GrabOutput(GrabContext(state=GrabState.GRAB), True, status='HELD')
        else:
            next_ctx.close_count = 0

    return GrabOutput(next_ctx, False,
                      status='PINCH TO GRAB' if next_ctx.open_observed else 'OPEN FINGERS')
