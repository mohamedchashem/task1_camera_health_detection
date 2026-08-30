# Operational Guide — Camera Health Monitoring Pipeline

This end-to-end guide walks you through setup, capturing reference baselines, running video streams, and executing test suites for the camera health monitoring pipeline.

---

## Prerequisites & Environment Setup

1. **Activate Virtual Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## Operational Workflow

### 1. Capture Reference Baselines

Before running defect monitoring on a camera, you must capture clean baseline metadata and reference edge maps (`cam_XX.jpg`, `cam_XX_edges.png`, `cam_XX.json`) from healthy video footage.

**Syntax:**
```powershell
python -m pipeline.capture_baseline <camera_id> <video_path>
python -m pipeline.capture_baseline <camera_id> <video_path> --acknowledge-degraded   # last resort, see below
```

The source footage used for baseline capture must be: well-lit, in-focus, static (camera not moving), and visually detailed (avoid pointing at a blank wall — the tampering detector needs real structural detail in the baseline to work correctly).

### 2. Verify Baseline Quality (enforced at capture time)

The capture tool now **refuses to persist** a baseline whose `quality_warnings` is non-empty: it prints/logs every warning, explains that tampering detection would be inoperative, writes nothing, and exits non-zero. The fix is to recapture from a genuinely clean segment — do **not** proceed with a degraded baseline.

- **Clean capture (normal case):** the tool persists `cam_XX.json` / `cam_XX.jpg` / `cam_XX_edges.png` and exits 0. The JSON has `"quality_warnings": []`.
- **Degraded capture (refused by default):** the tool exits non-zero and writes nothing. A degraded baseline (dark/blurry/unstructured) causes the tampering detector to silently report `degraded_baseline` (confidence 0.0) on every frame, with no visible runtime error — the single most common and hardest-to-diagnose failure mode in this pipeline.
- **`--acknowledge-degraded` (last resort):** persists the degraded baseline anyway, explicitly recording `"degraded_acknowledged": true` (plus a timestamp) in the JSON. Only use this if you knowingly accept that tampering detection will be inoperative for that camera until a usable baseline is captured. If you later see a baseline JSON with this field, that is why tampering is not firing.

### 3. Clean Prior Output Artifacts (Optional)

To clear old event databases, frame logs, and annotated snapshots before starting a fresh run (PowerShell syntax — this project's runtime data lives under `data/`, not `output/`):

```powershell
Remove-Item -Recurse -Force data\event_frames\*, data\logs\*, data\events.db -ErrorAction SilentlyContinue
```

### 4. Execute the Monitoring Pipeline

To process video streams through the multi-fault decision engine, run `main.py` using the `NAME=SOURCE` syntax for the `--camera` flag.

**Syntax:**
```powershell
python main.py --camera <camera_id>=<video_path>
python main.py --camera <camera_id>=<video_path>   # acknowledged-degraded baseline: no extra flag needed — the acknowledgment was recorded in the baseline JSON at capture time
python main.py --camera <camera_id>=<video_path> --allow-degraded-baseline   # last-resort bypass for an UNacknowledged-degraded camera, see below
python main.py --camera <camera_id_1>=<video_path_1> --camera <camera_id_2>=<video_path_2>   # multiple cameras in one run
```

Multiple cameras can be processed in one run by repeating the `--camera` flag.

**Degraded-baseline startup gate:** at startup, `main.py` recomputes the tampering-degraded condition authoritatively from each camera's baseline edge map. An **unacknowledged-degraded** camera (sparse structure; no `degraded_acknowledged: true` in its baseline JSON) is **skipped** — its tampering detector would be permanently inoperative (confidence 0.0 on every frame). If all cameras are skipped, the run refuses to start (exit code 2). To run such a camera anyway (e.g. against test/synthetic data), pass `--allow-degraded-baseline`; a prominent `DEGRADED BASELINE ACKNOWLEDGED` warning is logged at startup. A baseline persisted via `--acknowledge-degraded` starts normally with the same prominent warning.

### 5. Runtime Behavior — What the Operator Sees

The three-part degraded-baseline safeguard (capture-time acknowledgment, startup-time refusal/override, runtime visibility) means an inoperative tampering detector is **never a silent failure**:

- **Runtime non-measurements.** While a camera's baseline is degraded, the tampering detector still runs on every frame but reports itself **unmeasurable** (`"status": "unavailable"`, `"reason": "degraded_baseline"`) — it structurally cannot compare against a dark/blurry/unstructured baseline. Those frames are excluded from tampering's temporal-confirmation statistics: they are non-measurements, not genuine "no tampering" negatives, so they never dilute or suppress real results.
- **One "unmeasurable:" banner line.** On the annotated snapshot banner, an unmeasurable detector appears on the familiar `unmeasurable:` line, e.g. `unmeasurable: tampering` — the **same** line already used for gate-skipped detectors (near-black or severely blurred frames). A business viewer sees one consistent message ("this detector couldn't tell us anything right now") regardless of the technical cause; the banner deliberately makes no visual distinction.
- **Technically precise logs & metrics.** The per-frame diagnostic log keeps the two causes distinguishable for engineers: `"status": "unavailable"` / `"reason": "degraded_baseline"` for a degraded baseline, versus `"status": "skipped"` / `"reason": "suppressed_by_gate"` for a gate skip. Periodic system metrics include per-detector `unavailable_counts` alongside the existing `skipped_counts` / `error_counts`.

---

## CLI Reference Summary

| Task | Command Syntax |
|---|---|
| **Baseline Capture** | `python -m pipeline.capture_baseline <camera_id> <video_path>` (clean) / `python -m pipeline.capture_baseline <camera_id> <video_path> --acknowledge-degraded` (degraded, last resort) |
| **Stream Execution** | `python main.py --camera <camera_id>=<video_path>` (normal; also correct for an acknowledged-degraded camera — no extra flag) |
| **Stream Execution (override)** | `python main.py --camera <camera_id>=<video_path> --allow-degraded-baseline` (unacknowledged-degraded camera, last-resort bypass) |
| **Stream Execution (multi-camera)** | `python main.py --camera <camera_id_1>=<video_path_1> --camera <camera_id_2>=<video_path_2>` |
| **Run Fast Tests** | `pytest tests/ -k "not tilt_detector"` |
| **Run Full Test Suite** | `pytest tests/ -v` |

---

## Verification & Automated Testing

To confirm the multi-fault decision engine, priority banner renderer, and frame log persistence (schema `v3`) are fully operational, run the test suite:

```powershell
pytest tests/ -v
```

**Note on `test_tilt_detector.py`:** The tilt detector uses the DISK model (GPU/CUDA-accelerated learned keypoint matching) for real inference against a video fixture, not a lightweight CPU calculation. This test is significantly slower than the rest of the suite — typically several minutes, and has been observed to take **up to 45+ minutes** under heavy GPU load or when running alongside other GPU-intensive processes. This is expected behavior, not a hang. To skip it during rapid iteration:

```powershell
pytest tests/ -k "not tilt_detector"
```

