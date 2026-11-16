"""Compose reference drafts without execution or planning dependencies."""

import numpy as np

from wbc.core.types import MotionDraft, MotionPhase
from wbc.helpers.ik import Worst


def concatenate(drafts: list[MotionDraft]) -> MotionDraft:
    """Chain drafts that each start at the previous one's last frame."""
    if len(drafts) == 1:
        return drafts[0]
    first = drafts[0]  # in a multi-step plan every phase is named after its step: "0:walk", "1:hold", ...
    frames, keyframes = [first.frames], list(first.keyframes)
    phases = [MotionPhase(f"0:{phase.name}", phase.start, phase.stop) for phase in first.phases]
    support = [_support(first)]
    diagnostics = {f"0.{k}": v for k, v in first.diagnostics.items()}
    worst = Worst()
    worst.merge(first.diagnostics)
    offset = len(first.frames) - 1
    for index, draft in enumerate(drafts[1:], start=1):
        frames.append(draft.frames[1:])
        for phase in draft.phases:
            begin, stop = offset + max(phase.start, 1), offset + phase.stop
            if stop > begin:
                phases.append(MotionPhase(f"{index}:{phase.name}", begin, stop))
        support.append(_support(draft)[1:])
        keyframes += [k + offset for k in draft.keyframes if k > 0]
        diagnostics.update({f"{index}.{k}": v for k, v in draft.diagnostics.items()})
        worst.merge(draft.diagnostics, offset)
        offset += len(draft.frames) - 1
    diagnostics.update(worst.diagnostics())
    # Keep hand workspace warnings visible to callers after composing steps.
    reach, offset = {}, 0
    for draft in drafts:
        for side, row in draft.diagnostics.get("hand_reach", {}).items():
            if side not in reach or row["distance_m"] > reach[side]["distance_m"]:
                reach[side] = {**row, "frame": offset + row["frame"]}
        offset += len(draft.frames) - 1
    if reach:
        diagnostics["hand_reach"] = reach
    for name in ("hand_position_error_m", "hand_rotation_error_rad"):
        values = [d.diagnostics[name] for d in drafts if name in d.diagnostics]
        if values:
            diagnostics[name] = max(values)
    limit_stages, offset = [], 0
    for draft in drafts:
        limit_stages.extend((offset + frame, profile) for frame, profile in draft.limit_stages)
        offset += len(draft.frames) - 1
    has_support = any(d.support is not None for d in drafts)
    return MotionDraft(
        np.vstack(frames),
        tuple(phases),
        drafts[-1].end_posture,
        support=np.vstack(support) if has_support else None,
        keyframes=tuple(keyframes),
        diagnostics=diagnostics,
        limit_stages=tuple(limit_stages),
    )


def _support(draft: MotionDraft) -> np.ndarray:
    # Frames without a support claim are never slip-checked.
    return draft.support if draft.support is not None else np.zeros((len(draft.frames), 2), bool)
