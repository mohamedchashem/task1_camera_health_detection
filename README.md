# Camera Health Monitoring System

## Table of Contents

- [1. Project Overview & Architecture Summary](#1-project-overview--architecture-summary)
- [2. System Requirements & Prerequisites](#2-system-requirements--prerequisites)
- [3. Installation & Environment Setup](#3-installation--environment-setup)
- [4. Execution Reference](#4-execution-reference)
- [5. Testing & Validation](#5-testing--validation)
- [Appendix A: Repository Layout](#appendix-a-repository-layout)
- [Appendix B: Configuration Reference](#appendix-b-configuration-reference)
- [Appendix C: Detector Summary](#appendix-c-detector-summary)
- [Appendix D: Production Readiness Status](#appendix-d-production-readiness-status)

---

## 1. Project Overview & Architecture Summary

The Camera Health Monitoring System is a real-time computer vision pipeline for continuous fault detection across one or more camera feeds. The system operates as a single process with per-camera worker threads, bounded memory consumption, and fail-fast startup validation.

Four independent detectors evaluate each incoming frame:

- **Low-light detection** — identifies underexposed or degraded illumination conditions.
- **Tampering / obstruction detection** — identifies physical interference with the camera's field of view.
- **Blur / dirty-lens detection** — identifies loss of image sharpness consistent with lens contamination or defocus.
- **Tilt detection** — identifies unauthorized changes in camera orientation.

Detector outputs are fused by a per-camera `DecisionEngine`, which applies fault precedence rules, causal suppression logic, a spatial compactness guard (distinguishing genuine contiguous physical obstruction from scattered rotational edge-loss noise), and temporal confirmation before an event is considered confirmed. Confirmed fault episodes are persisted to a SQLite event store, alongside per-frame JSONL logs and periodic system metrics.

### 1.1 Architectural Overview

```
Camera 1 ─► Reader Sub-thread ─► Bounded Queue (FRAME_QUEUE_CAPACITY) ─► CameraWorker Thread ─► DecisionEngine
Camera 2 ─► Reader Sub-thread ─► Bounded Queue ────────────────────────► CameraWorker Thread ─► DecisionEngine
                                                                                    │
                                                  Main Thread: Metrics Loop + Cooperative Shutdown
                                                                                    ▼
                                          EventStore (SQLite, WAL) · FrameLogger (JSONL) · Annotated Frame Ring
```

### 1.2 Concurrency Model

- Each camera is managed by a dedicated `CameraWorker` thread (`main.py`). Each worker spawns a reader sub-thread that decodes frames into a bounded, per-camera queue (`queue.Queue(maxsize=FRAME_QUEUE_CAPACITY)`).
- **Live stream sources** apply drop-oldest backpressure: when the queue is full, the oldest queued frame is evicted to admit the newest, bounding memory usage under sustained load.
- **File-based sources** are lossless: the reader blocks for queue space rather than evicting frames, ensuring every frame of a local video file is processed exactly once.
- Frames dequeued beyond `MAX_PROCESSING_LAG_SECONDS` after capture are discarded as stale.
- The main thread runs a periodic metrics loop, writing per-camera counters to `system.jsonl`, and coordinates cooperative shutdown on `SIGINT`/`SIGTERM`, allowing workers to drain and flush within `SHUTDOWN_TIMEOUT_SECONDS`.

### 1.3 Decision Engine and Temporal Confirmation

The `DecisionEngine` (`pipeline/decision_engine.py`) consolidates the four per-frame detector outputs into a single fused decision per camera:

- **Fault precedence and suppression** — candidate faults are ranked according to `DECISION_PRECEDENCE` (`tampering > low_light > tilt > blur`). A `DECISION_SUPPRESSION_MAP` encodes causal relationships (for example, tampering can explain and suppress low-light, blur, and tilt symptoms), preventing symptomatic faults from being reported independently of their root cause. A suppressor requires a minimum confidence of `DECISION_SUPPRESSOR_MIN_CONFIDENCE` (0.20) to override a lower-precedence fault.
- **Spatial compactness guard** — tampering candidates are additionally validated against `TAMPERING_MIN_COMPACTNESS_RATIO` (0.60), the ratio of the largest contiguous structure-loss cluster to total structure loss. This distinguishes genuine, spatially contiguous physical obstructions from scattered edge-loss noise caused by camera rotation, preventing tilt events from being misclassified as tampering.
- **Temporal confirmation** — a single-frame candidate does not constitute a confirmed event. A `ConfirmationTracker` maintains a fixed-size sliding frame window per fault type (`CONFIRMATION_WINDOW_SIZE`, 30 frames); a fault is confirmed only when candidate frames constitute at least `CONFIRMATION_THRESHOLD` (0.50) of the observed frames within that window.
- **Fault isolation and backoff** — a detector exception is contained at the frame level without halting other detectors. A detector that exceeds its consecutive-error threshold is temporarily skipped and automatically retried after a configured backoff period.
- **Event-rate limiting** — a minimum gap interval prevents the same fault from re-confirming in rapid succession.

### 1.4 Persistence Layer

- **EventStore** — SQLite database operating in WAL mode with `synchronous=NORMAL`, schema-versioned via `PRAGMA user_version`. Writes are idempotent through a composite key, making event re-emission after restarts or stream reconnects a no-op. All queries are parameterized.
- **FrameLogger** — per-camera JSONL logs, size-rotated and retained for a configurable number of days.
- **Annotated frame ring buffer** — confirmed-fault snapshots are retained up to a bounded maximum count, with the oldest entries evicted first.
- **System metrics** — periodic per-camera throughput, latency, and drop-rate snapshots, written to a rotated JSONL file.

---

## 2. System Requirements & Prerequisites

### 2.1 Runtime Requirements

| Requirement | Specification |
|---|---|
| Python | 3.14.x (environment validated against 3.14.5) |
| Package manager | `pip` |
| Operating systems | Windows, macOS, Linux |
| GPU (optional) | NVIDIA GPU with CUDA-capable driver, required only for CUDA-accelerated execution |

### 2.2 Dependency Management

All Python package dependencies are pinned in `requirements.txt`. The default PyTorch dependency is a CPU-only build (`torch==2.13.0+cpu`) to guarantee installability across all machines and CI runners without a GPU-specific toolchain.

### 2.3 GPU Acceleration Prerequisites

GPU-accelerated execution (`--device cuda:0` or equivalent) requires the CPU-only PyTorch wheel to be replaced with a CUDA build matching the host driver version. This is an environment-level dependency substitution and requires no source code modification, as the compute device is resolved at runtime.

---

## 3. Installation & Environment Setup

### 3.1 Create and Activate a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.2 Install Base Dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Install CUDA-Enabled PyTorch (GPU Environments Only)

Identify the CUDA build corresponding to your installed driver from the official PyTorch wheel index (`https://download.pytorch.org/whl/torch/`), then install it in place of the default CPU wheel:

```bash
pip install --upgrade --force-reinstall torch==2.13.0 --index-url https://download.pytorch.org/whl/cuXXX
```

Replace `cuXXX` with the appropriate CUDA build identifier for the target environment.

### 3.4 Verify CUDA Availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

This command must return `True` before any `--device cuda:0` execution path is used. If it returns `False`, either the CUDA wheel installation was unsuccessful or the host driver is incompatible.

### 3.5 Test Fixture Provisioning

The repository does not include committed binary media assets. Synthetic test footage and detector baselines (`data/test_footage/*`, `data/baselines/*`) are excluded from version control and are materialized automatically on the first test run via `tests/conftest.py`. Fixture generation is deterministic, uses a fixed random seed, and will raise an exception on failure rather than allowing tests to silently skip. No manual setup action is required prior to running the test suite.

---

## 4. Execution Reference

### 4.1 Baseline Capture (Required Prior to First Run)

Each camera requires a baseline representing its normal operating condition, captured from a clean, fault-free segment prior to production use. `main.py` will not start for a camera lacking a baseline.

```bash
python -m pipeline.capture_baseline <camera_id> <rtsp://source or path/to/video.mp4>
```

This produces the following artifacts under `data/baselines/`:

| Artifact | Description |
|---|---|
| `<camera_id>.json` | Brightness and sharpness reference record |
| `<camera_id>.jpg` | Reference frame for tilt detection |
| `<camera_id>_edges.png` | Stable-edge baseline for tampering detection |

### 4.2 Running the Pipeline

Live RTSP sources, with credentials resolved from environment variables to avoid exposure in the command line, process list, or logs:

**Windows (PowerShell):**

```powershell
python main.py --camera cam1=rtsp://user:pass@host/stream ^
               --camera cam2=env:RTSP_URL
```

**macOS / Linux:**

```bash
python main.py --camera cam1=rtsp://user:pass@host/stream \
               --camera cam2=env:RTSP_URL
```

Local file source (for development and validation):

```bash
python main.py --camera cam1=data/test_footage/test_video.mp4
```

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | Clean shutdown |
| `2` | Usage or configuration error (invalid flags, missing baseline or source file, unrecognized `--config` key, unavailable CUDA device) |

### 4.3 Configuration Override Precedence

Configuration is resolved in the following order, from highest to lowest precedence:

1. Explicit command-line flags
2. `--config KEY=VALUE` overrides
3. Defaults defined in `config.py`

`--config` is repeatable and accepts any constant defined in `config.py`; values are automatically coerced to the appropriate type, and unrecognized keys cause startup to fail.

**Dedicated command-line flags:**

| Flag | Overrides |
|---|---|
| `--tilt-interval <sec>` | `TILT_SAMPLE_INTERVAL_SECONDS` |
| `--db <path>` | `EVENTS_DB_PATH` |
| `--frame-log <path>` | `FRAME_LOG_PATH` |
| `--system-log <path>` | `SYSTEM_LOG_PATH` |
| `--event-frames <path>` | `EVENT_FRAMES_DIR` |
| `--session-id <id>` | Event idempotency-key component |
| `--device <cuda\|cuda:N\|cpu>` | Compute device |
| `--allow-cpu-fallback` | `ALLOW_CPU_FALLBACK=True` |
| `--log-level <LEVEL>` | Console log level |

Device resolution precedence: `--device` > `TILT_DEVICE` > `DEFAULT_DEVICE` (default `"cuda"`). If the resolved CUDA device is unavailable, startup fails unless CPU fallback is explicitly enabled via `--allow-cpu-fallback` or `ALLOW_CPU_FALLBACK=true`.

**Example invocations:**

```bash
python main.py --camera cam1=rtsp://... --config FRAME_QUEUE_CAPACITY=5
python main.py --camera cam1=rtsp://... --tilt-interval 1.5 --device cuda:0 --log-level DEBUG
```

---

## 5. Testing & Validation

### 5.1 Running the Automated Test Suite

**GPU-accelerated execution:**

```bash
python -m pytest tests/ --device cuda:0
```

- The `--device` flag is applied to the tilt detector prior to test execution (`tests/conftest.py`).
- If CUDA is requested and unavailable, the run aborts immediately, consistent with the project's fail-fast device policy. There is no implicit CPU fallback during testing.

**CPU execution (no GPU required):**

```bash
python -m pytest tests/
```

> **CUDA requirement:** the DISK keypoint ground-truth tests in
> `tests/test_tilt_detector.py` configure the device as `cuda` at module
> scope, so they require a CUDA-capable device with a matching CUDA torch
> build (see section 2.3). On a CPU-only machine those tests fail; run
> them only on a GPU host.

### 5.2 Ground-Truth Validation

Synthetic test footage is generated with fault windows injected at known, fixed time intervals. Automated tests validate detector output against these ground-truth windows using blind grading, providing an objective pass/fail signal independent of manual visual inspection.

The `scripts/validate_*.py` tools are manual diagnostic utilities: they
score an arbitrary video file (e.g. real footage with physically staged
faults) against a camera's captured baseline and dump per-frame results
to CSV under `data/test_runs/` for manual review. They are NOT invoked
by the `pytest` suite. The automated suite grades the synthetic fixture
only, through `tests/` and `scripts/generate_test_fixtures.py`.

### 5.3 Test Coverage Scope

The current automated suite validates system behavior exclusively against deterministic, synthetically generated footage. Validation against real-world footage and physically staged fault scenarios is not yet part of the automated suite and remains an open item (see Appendix D).

---

## Appendix A: Repository Layout

```
config.py                     Central thresholds and paths; validated at startup via validate_config()
main.py                       Operational entry point (CLI argument parsing, per-camera worker orchestration)
requirements.txt              Pinned Python dependencies
detectors/                    Low-light, tampering, blur, and tilt detector implementations
pipeline/                     capture_baseline, decision_engine, event_store, frame_logger,
                               stream_reader, file_reader, annotate, paths
scripts/                      Test fixture generation and ground-truth validation utilities
tests/                        Unit and integration test suite; conftest.py (fixtures, --device flag)
data/                         Runtime data: baselines, logs, event frames (git-ignored)
```

## Appendix B: Configuration Reference

All configuration constants are defined in `config.py` and validated at startup by `validate_config()`.

| Area | Constant | Default | Description |
|---|---|---|---|
| Pipeline | `FRAME_QUEUE_CAPACITY` | 3 | Per-camera bounded queue depth; drop-oldest eviction policy on overflow |
| Pipeline | `MAX_PROCESSING_LAG_SECONDS` | 2.0 | Maximum queue residency before a live-stream frame is dropped as stale |
| Pipeline | `TILT_SAMPLE_INTERVAL_SECONDS` | 0.0 | Tilt detector sampling cadence; `0.0` evaluates every frame |
| Pipeline | `METRICS_LOG_INTERVAL_SECONDS` | 5.0 | System metrics snapshot cadence |
| Pipeline | `SHUTDOWN_TIMEOUT_SECONDS` | 5.0 | Cooperative shutdown deadline |
| Stream | `STREAM_RECONNECT_BASE_SECONDS` | 2.0 | Initial stream reconnect wait interval |
| Stream | `STREAM_RECONNECT_MAX_SECONDS` | 30.0 | Reconnect backoff ceiling |
| Stream | `STREAM_RECONNECT_BACKOFF_FACTOR` | 2.0 | Per-attempt backoff multiplier |
| Stream | `STREAM_MAX_CONSECUTIVE_FAILURES` | 15 | Failed reads before the connection is treated as dropped |
| Decision | `CONFIRMATION_WINDOW_SIZE` | 30 | Sliding window frame count for confirming persistent faults |
| Decision | `CONFIRMATION_THRESHOLD` | 0.50 | Fraction of positive frames within the window required to confirm a fault state |
| Decision | `DECISION_SUPPRESSOR_MIN_CONFIDENCE` | 0.20 | Minimum confidence required for a primary fault to suppress lower-precedence symptoms |
| Decision | `TAMPERING_MIN_COMPACTNESS_RATIO` | 0.60 | Minimum contiguous cluster ratio (largest cluster / total structure loss) required to validate physical occlusion vs. scattered rotation noise |
| Decision | `TILT_SHIFT_CONFIDENCE_CEILING_RATIO` | 0.15 | Median keypoint shift diagonal ratio required for maximum tilt confidence scaling |
| Decision | `DECISION_MIN_EVENT_GAP_SECONDS` | 10.0 | Minimum interval before the same fault type may re-confirm |
| Decision | `DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS` | 30 | Consecutive detector errors before temporary backoff |
| Decision | `DECISION_DETECTOR_ERROR_BACKOFF_SECONDS` | 5.0 | Detector backoff duration following repeated errors |
| Persistence | `FRAME_LOG_RETENTION_DAYS` | 7 | Frame log retention period, by age |
| Persistence | `FRAME_LOG_MAX_BYTES` | 10 MB | Frame log file rotation threshold |
| Persistence | `SYSTEM_LOG_MAX_BYTES` / `SYSTEM_LOG_BACKUP_COUNT` | 10 MB / 3 | Metrics log rotation policy |
| Persistence | `APP_LOG_MAX_BYTES` / `APP_LOG_BACKUP_COUNT` | 10 MB / 5 | Application log rotation policy |
| Persistence | `EVENT_FRAMES_MAX_TOTAL` | 200 | Annotated-frame ring buffer capacity |
| Baseline | `BASELINE_CAPTURE_SECONDS` | 3.0 | Duration of the baseline capture window |

**Design rationale:** The bounded frame queue with drop-oldest eviction limits memory consumption and bounds end-to-end latency under slow or degraded feeds. Reconnect backoff timers protect recovering streams from repeated connection storms. Log and event-frame retention limits prevent unbounded disk growth during extended deployments.

## Appendix C: Detector Summary

| Detector | Signal | Trigger Condition |
|---|---|---|
| Low-light | HSV V-channel dark-pixel ratio | Ratio exceeds the camera baseline by 50% or more |
| Tampering | Canny edge detection, gridded into connected clusters | A contiguous structure-loss cluster covers 15% or more of the baseline structure, and the largest cluster comprises at least `TAMPERING_MIN_COMPACTNESS_RATIO` (0.60) of total structure loss (spatial compactness guard, distinguishing physical occlusion from rotational edge-loss noise) |
| Blur | Variance of the Laplacian | Sharpness drops to 50% or less of the baseline value |
| Tilt | DISK feature extraction with SMNN matching and MAD outlier rejection | Median matched-keypoint shift equals or exceeds 10% of the frame diagonal |

## Appendix D: Production Readiness Status

The system has been validated end-to-end against deterministic synthetic fixtures using blind ground-truth grading. Validation against real-world footage with physically staged fault conditions, and evaluation against a broader production dataset, have not yet been completed. Until this validation is performed, the system should be considered **not production-ready** for client deployment.
