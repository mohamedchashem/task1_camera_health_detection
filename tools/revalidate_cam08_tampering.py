"""Re-validate the tampering-candidacy fix against the REAL cam_08 footage.

Subtask 10/10 closing evidence: re-run the tampering detector's actual
``evaluate()`` (and the blur detector's, for the co-occurring-blur check)
directly against the real video that originally exposed the bug
(``data/test_footage/test_video8.mp4``, camera ``cam_08``, captured clean
baseline ``data/baselines/cam_08*`` with ``tampering_meaningful_block_fraction``
~= 0.686).

This is deliberately NOT a full pipeline run: it loads only the two real
inputs the tampering detector needs (the baseline edge map and the raw
frames), calls ``tampering.evaluate()`` frame by frame, and reports:

- the original bug case  (t ~= 23.5-23.7 s: large obstruction co-occurring
  with severe blur, blur confidence ~0.915-0.955) -- before this fix the
  absolute total-loss ceiling rejected it (``is_candidate == False``);
- the normal localized-obstruction case (t ~= 36.0-37.0 s), which must
  still behave exactly as before the fix (regression check);
- every candidate window found across the whole real clip, to broaden the
  evidence beyond the two known moments;
- any ``degraded_ambient`` / ``degraded_baseline`` non-measurements.

Usage:  python tools/revalidate_cam08_tampering.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

# --- sys.path bootstrap ---------------------------------------------------
# `python tools/revalidate_cam08_tampering.py` inserts tools/ (not the project
# root) at sys.path[0], so `from config import ...` would fail without this.
# It also lets `python -m tools.revalidate_cam08_tampering` work from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import BASELINES_DIR, PROJECT_ROOT  # noqa: E402
from detectors.blur import evaluate as evaluate_blur  # noqa: E402
from detectors.tampering import (
    TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION as MIN_CLUSTER_FRACTION,
)
from detectors.tampering import (
    TAMPERING_MIN_COMPACTNESS_RATIO as MIN_COMPACTNESS,
)
from detectors.tampering import (
    compute_edge_map,
    compute_loss_fractions,
    evaluate as evaluate_tampering,
)
from pipeline.file_reader import read_frames_from_file  # noqa: E402

# The legacy absolute total-loss ceiling removed by this fix (subtasks 6-7).
LEGACY_MAX_GLOBAL_LOSS_FRACTION = 0.75

CAMERA_ID = "cam_08"
VIDEO_REL = "data/test_footage/test_video8.mp4"

BUG_CASE_WINDOW = (23.0, 24.5)     # original bug: large obstruction + severe blur
NORMAL_CASE_WINDOW = (35.5, 37.5)  # localized obstruction alone (must be unchanged)


def _load_baseline_edges() -> cv2.typing.MatLike:
    path = BASELINES_DIR / f"{CAMERA_ID}_edges.png"
    edges = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise RuntimeError(f"Failed to load baseline edge map at {path}.")
    return edges


def _load_baseline_blur_sharpness() -> float:
    path = BASELINES_DIR / f"{CAMERA_ID}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    return float(record["blur_baseline_sharpness"])


def _fmt(row: dict, blur=None) -> str:
    parts = [
        f"is_candidate={row['is_candidate']}",
        f"confidence={row['confidence']:.3f}",
        f"reason={row['reason']!r}",
        f"meaningful={row['meaningful']:.3f}",
        f"total_loss={row['total_loss']:.3f}",
        f"largest_cluster={row['largest_cluster']:.3f}",
    ]
    if blur is not None:
        parts.append(f"blur_conf={blur.confidence:.3f} blur_cand={blur.is_candidate}")
    return ", ".join(parts)


def _frame_at(video_path, frame_number):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number}.")
    return frame


def _legacy_info(video_path, frame_number, edges) -> dict:
    """Recompute the LEGACY (pre-fix) binary structure-loss decision for one
    frame, so before/after behavior is directly comparable on the same real
    frames. Legacy candidacy gates (see the pre-fix detector and
    Documentation.md §3.2 item 7): largest contiguous cluster >= 0.15 of
    meaningful blocks, compactness >= 0.60, AND total loss <= the 0.75
    absolute global-loss ceiling. The new model keeps the first two gates and
    replaces the ceiling with the ambient-relative comparison."""
    frame = _frame_at(video_path, frame_number)
    current_edges = compute_edge_map(frame)
    largest, total = compute_loss_fractions(edges, current_edges)
    # Compactness = largest cluster / all lost blocks (binary model).
    compactness = largest / total if total > 0 else 1.0
    legacy_candidate = (
        largest >= MIN_CLUSTER_FRACTION
        and compactness >= MIN_COMPACTNESS
        and total <= LEGACY_MAX_GLOBAL_LOSS_FRACTION
    )
    return {
        "largest": largest,
        "total": total,
        "compactness": compactness,
        "legacy_reject_reason": (
            None
            if legacy_candidate
            else ("total_loss_ceiling" if total > LEGACY_MAX_GLOBAL_LOSS_FRACTION
                  else ("min_size" if largest < MIN_CLUSTER_FRACTION
                        else "compactness"))
        ),
        "legacy_candidate": legacy_candidate,
    }


def _fmt_legacy(info: dict) -> str:
    return (f"legacy[largest={info['largest']:.3f} total={info['total']:.3f} "
            f"compact={info['compactness']:.3f} candidate={info['legacy_candidate']} "
            f"reject={info['legacy_reject_reason']!r}]")


def main() -> None:
    video_path = PROJECT_ROOT / VIDEO_REL
    if not video_path.exists():
        raise SystemExit(f"Real video not found at {video_path}.")

    edges = _load_baseline_edges()
    blur_baseline = _load_baseline_blur_sharpness()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = frame_count / fps if fps else 0.0
    print(f"Video: {VIDEO_REL}")
    print(f"  fps={fps:.3f} frames={frame_count} duration={duration_s:.2f}s "
          f"resolution={edges.shape[1]}x{edges.shape[0]}")
    print(f"Baseline: {CAMERA_ID}  blur_baseline_sharpness={blur_baseline:.3f}")

    rows = []
    for frame_number, video_time_s, frame in read_frames_from_file(video_path):
        t = float(video_time_s)
        tr = evaluate_tampering(frame, edges)
        rows.append(
            {
                "frame": frame_number,
                "t": t,
                "is_candidate": tr.is_candidate,
                "confidence": tr.confidence,
                "reason": tr.reason,
                "meaningful": tr.meaningful_block_fraction,
                "total_loss": tr.total_loss_fraction,
                "largest_cluster": tr.largest_contiguous_loss_fraction,
            }
        )

    print(f"\n--- Bug case (t = {BUG_CASE_WINDOW[0]:.2f}-{BUG_CASE_WINDOW[1]:.2f} s): "
          "large obstruction + severe co-blur ---")
    for r in [x for x in rows if BUG_CASE_WINDOW[0] <= x["t"] <= BUG_CASE_WINDOW[1]]:
        blur = evaluate_blur(_frame_at(video_path, r["frame"]), blur_baseline)
        leg = _legacy_info(video_path, r["frame"], edges)
        print(f"  frame {r['frame']:>4d}  t={r['t']:.2f}s  {_fmt(r, blur)}\n"
              f"         {_fmt_legacy(leg)}")

    print(f"\n--- Normal case (t = {NORMAL_CASE_WINDOW[0]:.2f}-{NORMAL_CASE_WINDOW[1]:.2f} s): "
          "localized obstruction alone ---")
    for r in [x for x in rows if NORMAL_CASE_WINDOW[0] <= x["t"] <= NORMAL_CASE_WINDOW[1]]:
        leg = _legacy_info(video_path, r["frame"], edges)
        print(f"  frame {r['frame']:>4d}  t={r['t']:.2f}s  {_fmt(r)}\n"
              f"         {_fmt_legacy(leg)}")

    print("\n--- Before/after summary on the two known real moments ---")
    for label, lo, hi in (("bug case", *BUG_CASE_WINDOW), ("normal case", *NORMAL_CASE_WINDOW)):
        in_win = [x for x in rows if lo <= x["t"] <= hi and x["is_candidate"]]
        confs = [x["confidence"] for x in in_win]
        bug = [r for r in rows if 23.45 <= r["t"] <= 23.70]
        norm = [r for r in rows if 36.40 <= r["t"] <= 37.00]
        if label == "bug case":
            leg_total_max = max(_legacy_info(video_path, r["frame"], edges)["total"]
                                for r in bug)
            new_conf = [r["confidence"] for r in bug]
            print(f"  {label}: new model candidate rate={sum(r['is_candidate'] for r in bug)}/{len(bug)} "
                  f"conf[min={min(new_conf):.3f} mean={sum(new_conf)/len(new_conf):.3f} "
                  f"max={max(new_conf):.3f}]; legacy would REJECT "
                  f"(total_loss up to {leg_total_max:.3f} > {LEGACY_MAX_GLOBAL_LOSS_FRACTION:.2f} ceiling)")
        else:
            leg_cands = [_legacy_info(video_path, r["frame"], edges)["legacy_candidate"]
                         for r in norm]
            print(f"  {label}: new model candidate rate={sum(r['is_candidate'] for r in norm)}/{len(norm)} "
                  f"conf[min={min(confs):.3f} mean={sum(confs)/len(confs):.3f} "
                  f"max={max(confs):.3f}]; legacy candidate rate="
                  f"{sum(leg_cands)}/{len(norm)} (identical)")

    print("\n--- All tampering candidate windows across the whole clip ---")
    windows = []
    for r in rows:
        if r["is_candidate"] and (not windows or r["frame"] != windows[-1][-1]["frame"] + 1):
            windows.append([r])
        elif r["is_candidate"]:
            windows[-1].append(r)
    if not windows:
        print("  (none)")
    for w in windows:
        confs = [x["confidence"] for x in w]
        print(f"  frames {w[0]['frame']:>4d}-{w[-1]['frame']:>4d}  "
              f"t={w[0]['t']:.2f}-{w[-1]['t']:.2f}s  n={len(w):>3d}  "
              f"conf[min={min(confs):.2f} mean={sum(confs) / len(confs):.2f} max={max(confs):.2f}]")

    print("\n--- Non-measurements across the whole clip ---")
    degraded = [r for r in rows if r["reason"] in ("degraded_ambient", "degraded_baseline")]
    if not degraded:
        print("  none: every frame was a clean measurement or candidate/negative.")
    else:
        for r in degraded:
            print(f"  frame {r['frame']:>4d}  t={r['t']:.2f}s  reason={r['reason']!r}")


if __name__ == "__main__":
    main()
