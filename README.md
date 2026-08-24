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

- **Fault precedence and multi-label suppression** — candidate faults are ranked according to `DECISION_PRECEDENCE`. Each ordered (suppressor, suppressed) pair carries an explicit relation class in `DECISION_SUPPRESSION_RULES`:
  - `always` — a deterministic physical link: the suppressor's signal fully explains the suppressed one, so suppression is unconditional once the margin check passes (e.g. tilt → blur: rotation resampling mechanically lowers Laplacian sharpness).
  - `conditional` — a link that exists only when the frame's own raw measurands confirm it is active: tampering → low_light and tampering → blur require the obstruction's lost-structure area to conservatively explain the observed dark/sharpness change (area conservation, with the `*_AREA_SLACK` tolerances); low_light → blur requires the frame to be near-black (`LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR`).
  - `independent` — no physical pathway, so the faults always co-survive and never suppress each other (e.g. tampering → tilt, low_light → tilt: structure loss or darkness cannot causally explain a geometric displacement). Pairs absent from `DECISION_SUPPRESSION_RULES` are independent by default.
  General rule: a lower-precedence fault survives alongside a higher-precedence one unless the pair is causally linked AND (for `conditional` pairs) the frame's own measurands confirm the link is active AND the suppressor clears its confidence floor (`DECISION_SUPPRESSOR_MIN_CONFIDENCE`) and a relative margin (`DECISION_SUPPRESSION_MARGIN`) over the suppressed fault's confidence, so weak noise cannot override a strong signal. The survivors are emitted as a multi-label fault list in precedence order (top survivor = `primary_fault`, the rest = `secondary_symptoms`).

- **Execution gating** — detectors run in a specified order (cheap signal detectors first, expensive structural last). Rather than a single static gate threshold, each gate detector has its own confidence floor; once a gate clears, it skips downstream detectors listed under it. This reduces redundant computation on frames where certain conditions are already known. Note the distinction between the two mechanisms: `DECISION_SUPPRESSION_RULES` encodes physical cause-and-effect between co-occurring faults, while `DECISION_GATE_SKIP_MAP` is execution gating — a decision about signal reliability (a detector's output is unmeasurable under adverse conditions, e.g. keypoint matching on near-black or severely blurred frames). The two can overlap in effect but not in meaning: a gate prevents an unreliable detector from running; a suppression rule removes a confirmed symptom of a higher-precedence root cause.

- **Emission gating** — a candidate frame counts toward temporal confirmation only when its confidence reaches the fault's own floor, so low-confidence noise never enters the confirmation pipeline.

- **Spatial validation** — certain fault types validate candidates against spatial structure rules (e.g., checking that anomalous regions are contiguous rather than scattered noise), distinguishing genuine faults from sensor artifacts.

- **Temporal confirmation** — a single-frame candidate does not constitute a confirmed event. A `ConfirmationTracker` maintains a time-based sliding window per fault type; a fault is confirmed only when positive frames constitute a minimum ratio of observed frames within that window.

- **Raw vs reportable confidence** — the engine preserves each detector's raw confidence verbatim in `DetectorObservation.raw_confidence` while the reportable `confidence` (the value used by decision paths, logs, and banners) is zeroed whenever the detector says the frame is not a valid candidate. A non-candidate therefore can never present a high confidence that reads as a contradiction (observed: tampering confidence 1.0 with `is_candidate=False`). Logs and banners display the reportable `confidence` and carry `raw_confidence` for debugging.

- **Fault isolation and backoff** — a detector exception is contained at the frame level without halting other detectors. A detector that exceeds its consecutive-error threshold is temporarily skipped and automatically retried after a backoff period.

- **Event-rate limiting** — a minimum gap interval prevents the same fault from re-confirming in rapid succession.

### 1.4 Persistence Layer

- **Event Store** — structured database with parameterized queries, schema versioning, and idempotent writes via composite keys, making event re-emission after restarts a no-op.
- **Diagnostic Logs** — per-stream JSONL logs (frame-log schema v3), size-rotated and retention-configured. Each frame record carries the full banner-fix fields: `confirmed_faults`, `faults` (current survivors), `suppressed_faults` (causally-suppressed only), `below_floor_faults`, `temporal_status`, and per-detector `raw_confidence` alongside the reportable `confidence`.
- **Annotated Snapshots** — confirmed-fault images retained up to a bounded maximum count.
- **System Metrics** — periodic per-stream throughput, latency, and drop-rate snapshots, written to a rotated log.

### 1.5 Annotated Snapshot Banner

Confirmed-fault snapshots render a five-section banner in which every fault type appears in exactly ONE section (the exclusivity invariant). Each section honestly means one thing:

1. `FAULT: <type> (conf=<peak>)` — **confirmed** faults currently in the temporal tracker, shown with their peak confidence. A confirmed fault whose detector was gate-skipped this exact frame carries a `[unmeasurable]` marker on its own FAULT line (never a second listing).
2. `pending: <type> (conf=<frame>)` — current fusion **survivors** not yet confirmed, shown with their frame confidence.
3. `suppressed: <type>[, <type>]` — candidates that cleared their own emission floor but were removed by a surviving higher-precedence fault via a predicate-gated causal relation (physical cause-and-effect only).
4. `too weak: <type>[, <type>]` — candidates **below their own emission floor**; they never reached fusion, so they are shown as too weak, never mislabeled "suppressed".
5. `unmeasurable: <type>[, <type>]` — detectors gate-skipped this frame (`status="skipped"`, `reason="suppressed_by_gate"`); their signal is not meaningful under the frame's conditions, so they are surfaced instead of silently vanishing.

The legacy `DECISION_SUPPRESSION_MAP` (the V1 flat map) is consumed ONLY by the legacy test-only single-primary path (`resolve_primary_fault` / `fuse_observations`); it is **never** used by the live snapshot rendering path, which sources all five sections from `DecisionFrame` fields (`confirmed_faults`, `faults`, `suppressed_faults`, `below_floor_faults`, and gate-skip status). The per-frame diagnostic log (schema v3) persists the same fields so any frame can be replayed offline through the fixed pipeline.

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
| Decision | `DECISION_SUPPRESSION_MAP` | `{...}` | Legacy single-primary suppression map: which candidates a primary fault causally explains. Consumed ONLY by the legacy test-only single-primary path (`resolve_primary_fault` / `fuse_observations`); never used by the live snapshot rendering path (which uses the predicate-gated `DECISION_SUPPRESSION_RULES` fields) |
| Decision | `DECISION_SUPPRESSION_RULES` | `{...}` | Per-pair relation classes for multi-label suppression (`always` / `conditional` / `independent`); pairs absent from the map are `independent` (never suppress) |
| Decision | `TAMPERING_LOW_LIGHT_AREA_SLACK` | 0.20 | Slack for the tampering→low_light area-conservation predicate (extent mismatch tolerated between obstruction area and dark region) |
| Decision | `TAMPERING_BLUR_AREA_SLACK` | 0.20 | Slack for the tampering→blur area-conservation predicate (extent mismatch tolerated between obstruction area and lost-sharpness area) |
| Decision | `LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR` | 0.80 | Confidence floor at which low_light is treated as near-black for the low_light→blur predicate (alias of `DECISION_GATE_CONFIDENCE`) |
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

**Predicate-parameter provenance:** The multi-fault predicate constants (`TAMPERING_LOW_LIGHT_AREA_SLACK`, `TAMPERING_BLUR_AREA_SLACK`, `LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR`) are camera-agnostic reasoned starting points, not derived from any specific footage, and have not yet been validated against a broader real-footage dataset — consistent with the open items listed in Appendix D (Validation Status).

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