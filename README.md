# Real-Time Fault Detection Pipeline

## Table of Contents

- [1. Architecture & Design Overview](#1-architecture--design-overview)
- [2. System Requirements](#2-system-requirements)
- [3. Installation & Setup](#3-installation--setup)
- [4. Execution](#4-execution)
- [5. Testing & Validation](#5-testing--validation)
- [Appendix A: Repository Layout](#appendix-a-repository-layout)
- [Appendix B: Configuration Reference](#appendix-b-configuration-reference)
- [Appendix C: Detector Overview](#appendix-c-detector-overview)
- [Appendix D: Validation Status](#appendix-d-validation-status)

---

## 1. Architecture & Design Overview

This is a real-time computer vision pipeline for continuous fault detection across one or more input streams. The system operates as a single orchestrator process with per-stream worker threads, bounded memory consumption, and fail-fast startup validation.

Multiple independent detectors evaluate each incoming frame and emit candidate signals. A `DecisionEngine` fuses these signals per stream using precedence rules, causal suppression logic, spatial validation guards, and temporal confirmation before any event is considered confirmed. Confirmed events are persisted to a data store, alongside per-frame diagnostic logs and periodic system metrics.

### 1.1 Architectural Overview

```
Stream 1 ─► Reader Sub-thread ─► Bounded Queue ─► Worker Thread ─► DecisionEngine
Stream 2 ─► Reader Sub-thread ─► Bounded Queue ──────────────────► Worker Thread ─► DecisionEngine
                                                          │
                                    Main Thread: Metrics Loop + Cooperative Shutdown
                                                          ▼
                                    Event Store · Diagnostic Logs · Annotated Snapshots
```

### 1.2 Concurrency Model

- Each stream is managed by a dedicated `Worker` thread. Each worker spawns a reader sub-thread that decodes frames into a bounded, per-stream queue.
- **Live stream sources** apply drop-oldest backpressure: when the queue is full, the oldest queued frame is evicted to admit the newest, bounding memory usage under sustained load.
- **File-based sources** are lossless: the reader blocks for queue space rather than evicting frames, ensuring every frame is processed exactly once.
- Frames dequeued beyond a configurable max age are discarded as stale.
- The main thread runs a periodic metrics loop and coordinates cooperative shutdown on system signals, allowing workers to drain and flush within a configurable timeout.

### 1.3 Decision Engine and Temporal Confirmation

The `DecisionEngine` consolidates per-frame detector outputs into a fused decision per stream:

- **Fault precedence and suppression** — candidate faults are ranked according to a precedence order. A suppression map encodes causal relationships (e.g., a high-level fault can explain and suppress lower-level symptoms), preventing symptomatic faults from being reported independently of their root cause. A suppressor requires a minimum confidence floor and must clear a relative margin over the suppressed fault's confidence, so weak noise cannot override a strong signal.

- **Execution gating** — detectors run in a specified order (cheap signal detectors first, expensive structural last). Rather than a single static gate threshold, each gate detector has its own confidence floor; once a gate clears, it skips downstream detectors listed under it. This reduces redundant computation on frames where certain conditions are already known.

- **Emission gating** — a candidate frame counts toward temporal confirmation only when its confidence reaches the fault's own floor, so low-confidence noise never enters the confirmation pipeline.

- **Spatial validation** — certain fault types validate candidates against spatial structure rules (e.g., checking that anomalous regions are contiguous rather than scattered noise), distinguishing genuine faults from sensor artifacts.

- **Temporal confirmation** — a single-frame candidate does not constitute a confirmed event. A `ConfirmationTracker` maintains a time-based sliding window per fault type; a fault is confirmed only when positive frames constitute a minimum ratio of observed frames within that window.

- **Fault isolation and backoff** — a detector exception is contained at the frame level without halting other detectors. A detector that exceeds its consecutive-error threshold is temporarily skipped and automatically retried after a backoff period.

- **Event-rate limiting** — a minimum gap interval prevents the same fault from re-confirming in rapid succession.

### 1.4 Persistence Layer

- **Event Store** — structured database with parameterized queries, schema versioning, and idempotent writes via composite keys, making event re-emission after restarts a no-op.
- **Diagnostic Logs** — per-stream JSONL logs, size-rotated and retention-configured.
- **Annotated Snapshots** — confirmed-fault images retained up to a bounded maximum count.
- **System Metrics** — periodic per-stream throughput, latency, and drop-rate snapshots, written to a rotated log.

---

## 2. System Requirements

### 2.1 Runtime Requirements

| Requirement | Specification |
|---|---|
| Python | 3.10+ |
| Package manager | `pip` |
| Operating systems | Windows, macOS, Linux |
| GPU (optional) | NVIDIA GPU with CUDA-capable driver, for accelerated execution only |

### 2.2 Dependencies

- Core: `opencv-python`, `numpy`, `torch` (CPU or CUDA)
- Database: `sqlite3` (included in Python stdlib)
- Logging: Python `logging` module
- Optional: `tensorrt` for model optimization, `onnx` for model export

---

## 3. Installation & Setup

### 3.1 Clone and Install

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

### 3.2 Environment Validation

On startup, the system validates:
- Python version compatibility
- Required packages installed
- Configuration values within acceptable ranges
- File paths and permissions
- GPU availability (if CUDA is expected)

Validation failures halt startup with diagnostic messages, preventing silent misconfiguration.

### 3.3 Configuration

All configuration is centralized in `config.py`. Key sections:

- **Pipeline**: Frame queue depth, max processing lag, detector sampling cadence, metrics interval, shutdown timeout.
- **Stream**: Reconnection backoff parameters, max consecutive failures before stream is marked disconnected.
- **Decision**: Detector execution order, gate confidence thresholds, suppression rules, confirmation window durations, temporal thresholds.
- **Persistence**: Log retention, database settings, snapshot buffer size.
- **Baseline**: Baseline capture duration and quality thresholds.

All constants are validated at startup via `validate_config()`.

---

## 4. Execution

### 4.1 Single Stream

```bash
python main.py --source <stream_url_or_file_path>
```

### 4.2 Multiple Streams

```bash
python main.py --source stream1.mp4 --source stream2.mp4 --source rtsp://camera1/stream
```

### 4.3 Output

The system writes:
- `data/events.db` — all confirmed events, queryable by stream, fault type, and time range
- `data/logs/frame_*.jsonl` — per-stream frame-level diagnostics
- `data/logs/system.jsonl` — periodic throughput and latency metrics
- `data/snapshots/` — annotated images of confirmed faults

### 4.4 Shutdown

Press `Ctrl+C` (SIGINT) or send `SIGTERM`. Workers drain pending frames and flush logs within the configured timeout.

---

## 5. Testing & Validation

### 5.1 Unit & Integration Tests

```bash
pytest tests/ --device cpu    # or gpu
```

Tests validate:
- Detector correctness on synthetic, known-ground-truth fixtures
- Decision engine fusion and suppression logic
- Confirmation window behavior and edge cases
- Persistence layer idempotency
- Stream reader resilience to reconnects and malformed data

### 5.2 Manual Validation Scripts

```bash
python scripts/validate_on_footage.py --video <path> --detector <name>
```

Runs a single detector against real footage and outputs per-frame results to CSV for manual review. Useful for checking detector behavior on real-world conditions before deploying to production.

### 5.3 Test Coverage Scope

The automated suite validates system behavior against deterministic, synthetically generated footage. Real-world footage validation is conducted separately using manual diagnostic scripts. Production validation spans both synthetic (for repeatability and regression detection) and real (for real-world robustness).

---

## Appendix A: Repository Layout

```
config.py                     Central configuration and startup validation
main.py                       Operational entry point (CLI, orchestration)
requirements.txt              Pinned Python dependencies
detectors/                    Detector implementations
pipeline/                     Core pipeline components (decision engine, event store, etc.)
scripts/                      Validation and diagnostic utilities
tests/                        Automated test suite
data/                         Runtime data (logs, events, snapshots) — git-ignored
```

---

## Appendix B: Configuration Reference

All configuration constants are defined in `config.py`.

| Area | Constant | Default | Description |
|---|---|---|---|
| Pipeline | `FRAME_QUEUE_CAPACITY` | 3 | Per-stream queue depth; drop-oldest on overflow |
| Pipeline | `MAX_PROCESSING_LAG_SECONDS` | 2.0 | Max age before a live frame is dropped as stale |
| Pipeline | `DETECTOR_SAMPLING_INTERVAL` | 0.0 | Sampling cadence; 0.0 evaluates every frame |
| Pipeline | `METRICS_LOG_INTERVAL_SECONDS` | 5.0 | Metrics snapshot frequency |
| Pipeline | `SHUTDOWN_TIMEOUT_SECONDS` | 5.0 | Cooperative shutdown deadline |
| Stream | `STREAM_RECONNECT_BASE_SECONDS` | 2.0 | Initial reconnect wait |
| Stream | `STREAM_RECONNECT_MAX_SECONDS` | 30.0 | Reconnect backoff ceiling |
| Stream | `STREAM_RECONNECT_BACKOFF_FACTOR` | 2.0 | Per-attempt backoff multiplier |
| Stream | `STREAM_MAX_CONSECUTIVE_FAILURES` | 15 | Failures before connection marked dropped |
| Decision | `DECISION_EXECUTION_ORDER` | `[...]` | Per-frame detector run order (cheap first, expensive last) |
| Decision | `DECISION_GATE_CONFIDENCE_BY_GATE` | `{...}` | Per-gate confidence floors for execution gating |
| Decision | `DECISION_GATE_SKIP_MAP` | `{...}` | Which detectors are skipped once a gate clears |
| Decision | `DECISION_CONFIRM_MIN_CONFIDENCE` | `{tampering: 0.5, low_light: 0.5, blur: 0.5, tilt: 0.5}` | Per-fault min confidence to enter confirmation window |
| Decision | `DECISION_SUPPRESSOR_MIN_CONFIDENCE` | 0.50 | Min confidence for a fault to suppress another |
| Decision | `DECISION_SUPPRESSION_MARGIN` | 1.5 | Relative margin suppressor must clear |
| Decision | `DECISION_CONFIRMATION_WINDOW_SECONDS` | 3.0 | Temporal confirmation window length |
| Decision | `DECISION_CONFIRMATION_MIN_POSITIVE_RATIO` | 0.50 | Fraction of positive frames required to confirm |
| Decision | `DECISION_CONFIRMATION_MIN_WINDOW_FRAMES` | 3 | Minimum frames observed in window |
| Decision | `DECISION_MIN_EVENT_GAP_SECONDS` | 10.0 | Min interval before same fault re-confirms |
| Decision | `DECISION_DETECTOR_MAX_CONSECUTIVE_ERRORS` | 30 | Error threshold before detector backoff |
| Decision | `DECISION_DETECTOR_ERROR_BACKOFF_SECONDS` | 5.0 | Backoff duration after errors |
| Persistence | `LOG_RETENTION_DAYS` | 7 | Diagnostic log retention period |
| Persistence | `LOG_MAX_BYTES` | 10 MB | Log file rotation threshold |
| Persistence | `SNAPSHOT_MAX_TOTAL` | 200 | Annotated snapshot buffer capacity |
| Baseline | `BASELINE_CAPTURE_SECONDS` | 3.0 | Baseline capture window duration |

**Design rationale:** The bounded frame queue with drop-oldest eviction limits memory consumption and bounds end-to-end latency under slow or degraded feeds. Reconnect backoff timers protect recovering streams from connection storms. Log and snapshot retention limits prevent unbounded disk growth during extended deployments.

---

## Appendix C: Detector Overview

| Detector | Signal Type | Trigger Condition |
|---|---|---|
| Detector A | Low-level signal | Triggers when ratio exceeds baseline by configured threshold |
| Detector B | Structural anomaly | Triggers when clustered anomaly exceeds size and compactness thresholds |
| Detector C | Sharpness/quality metric | Triggers when metric drops below configured ratio of baseline |
| Detector D | Feature-based geometry | Triggers when matched features shift by configured amount; marked unreliable if feature quality is degraded |

---

## Appendix D: Validation Status

**Current State: Functional, Testing In Progress**

The system has been validated against:
- Deterministic synthetic test fixtures with known ground truth (automated suite, 100% pass rate across unit and integration tests)
- Real footage from at least one source (manual diagnostic tooling)

Known open items for production deployment:
- Validation against a broader dataset of real footage from diverse conditions
- Full validation of CPU-only execution paths
- Multi-fault detection capability (independent faults occurring simultaneously)
- Performance profiling under sustained high-throughput load

**Recommendation:** Before production deployment, conduct validation against a representative dataset covering the full range of real-world conditions the system will encounter. Ensure CPU and GPU execution paths are both tested. Document any edge cases or performance limitations discovered.