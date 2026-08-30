# Camera Health Monitoring — Project Documentation

**Purpose of this document:** a technical history of the camera health monitoring system — what was tried for each component, what worked, what failed and why, and what replaced it — plus the current production state and known open issues. Written so the next engineer on this project understands why the current implementation looks the way it does, and doesn't re-attempt approaches that are already known dead ends.

---

## 1. System Overview

The system monitors camera health by running four independent fault detectors on each video frame, then passing their outputs through a decision engine that decides which faults are real and which are symptoms of another fault ("cross-triggering").

**The four detectors:**
| Detector | Fault it detects |
|---|---|
| `low_light` | Lights off / abnormal darkening relative to the camera's own baseline |
| `tampering` | Lens obstruction / physical blockage of the field of view |
| `blur` | Dirty/smudged lens or out-of-focus condition |
| `tilt` | Unauthorized change in camera orientation/angle |

**Core design principle (established early and held throughout the project):** every detector measures relative to a **per-camera baseline reference frame**, not a fixed global number. Fixed absolute thresholds were tried first for every detector and were rejected or replaced in every single case, because a threshold tuned for one camera's natural brightness/scene/texture misfires on a different camera. This is the single most repeated lesson across the whole project history — **do not reintroduce fixed absolute thresholds for any detector.**

**Known structural problem the project spent most of its later effort solving:** all four detectors read the same underlying pixel data, so a single real physical event triggers multiple detectors at once (e.g., a hand over the lens drops brightness *and* looks like tampering *and* can look like blur). This is called **cross-triggering** throughout the project and is the reason the decision engine exists.

**Decision engine evolution (three generations, in order):**
1. **Single-fault / precedence-only** — pick one winning fault per frame using a fixed priority order. Cannot represent two real faults happening at once.
2. **Multi-label with a static suppression map** — let multiple faults survive, but suppression between any two fault types is a single unconditional boolean ("always suppresses" / "never suppresses"). Cannot represent "sometimes causal, sometimes not."
3. **Multi-label with relation-class + physical-predicate fusion (current)** — each of the 6 possible fault-type pairs is classified as `always`, `conditional`, or `independent`, and `conditional` pairs are resolved per-frame by a physical predicate reading the detectors' own raw measurements (not just their confidence scores). This is the current production approach.

---

## 2. Current Production State

This section describes what is running today. Everything after this section is historical — how the system got here, and what was tried and discarded along the way.

### 2.1 Detector — Low Light
- **Method:** HSV V-channel dark-pixel ratio vs. per-camera baseline. Convert frame to HSV, compute the fraction of V-channel pixels below a fixed "dark" cutoff, compare that fraction against the same measurement on the baseline reference frame. Confidence is min-max scaled between "at baseline" (0) and "fully dark" (1).
- **Library:** OpenCV (`cv2.cvtColor` to HSV) + NumPy. No ML model.
- **Config:** `LOWLIGHT_DARK_PIXEL_THRESHOLD = 60`, `LOWLIGHT_BASELINE_DROP_RATIO = 0.5`
- **Why this design:** the HSV "V" channel is more stable than plain grayscale/RGB mean brightness under colored light sources (sodium streetlights, colored IR/LED). Measuring the *percentage of dark pixels* rather than a single mean is robust to one bright outlier (e.g., a reflection) skewing the average while the rest of the scene stays dark.
- **Known documented overlap (not a bug, a real cross-trigger):** the tampering fault window also triggers low_light candidates at up to ~61-80% depending on the run. Handled at the decision layer.

### 2.2 Detector — Tampering
- **Method:** Canny edge detection on baseline and current frame, divided into a grid of blocks (`TAMPERING_GRID_BLOCK_SIZE = 32`). Each meaningful block is scored by its **retention ratio** (current edge density / baseline edge density), and the frame's **ambient retention level** is estimated as the upper-tail quantile (0.90) of those ratios. A block is an obstruction block when its retention is at most a fixed fraction (0.35) of that ambient level — a **relative** comparison against the frame's own current state (the "Approach A" ambient-retention model, §4.7). Connected-component analysis then finds the largest single contiguous cluster of obstruction blocks, and a frame is a candidate only if that cluster is large enough (coverage), deep enough relative to ambient (outlier depth), and compact enough (not scattered).
- **Library:** OpenCV (Canny, `cv2.connectedComponentsWithStats`). No ML model.
- **Config:** `TAMPERING_GRID_BLOCK_SIZE = 32`, `TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY = 0.02`, `TAMPERING_BLOCK_DENSITY_DROP_RATIO = 0.5`, `TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO = 0.5`, `TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION = 0.15`, `TAMPERING_MIN_COMPACTNESS_RATIO = 0.60`, `TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION = 0.50`, `TAMPERING_AMBIENT_RETENTION_QUANTILE = 0.90`, `TAMPERING_OBSTRUCTION_DEPTH_RATIO = 0.35`, `TAMPERING_AMBIENT_MIN_RETENTION = 0.1`. (The legacy absolute ceiling `TAMPERING_MAX_GLOBAL_LOSS_FRACTION = 0.75` was **removed** by the Approach A redesign, §4.7.)
- **Why this design:** obstruction is a *localized* loss of structure; blur/low-light are *global* degradation. Measuring the largest contiguous cluster (not just total percentage lost) is what distinguishes "one physical object over the lens" from "the whole frame degraded." A prior version using per-pixel comparison had a ~69% false-positive rate on clean footage because fine repetitive texture (window blinds) shifts by a pixel or two between frames from ordinary compression noise — the block/grid approach absorbs that jitter. The relative ambient-retention comparison (replacing the absolute total-loss ceiling) keeps the obstruction a statistical outlier even when co-occurring blur collapses the frame's overall structure — the exact real case the old ceiling wrongly rejected.
- **Documented, accepted, unresolved limitation of this detector alone:** low-light and blur still trigger tampering candidates on their own fault windows, because a uniformly-degraded frame is technically "one contiguous cluster" too (see §3.2 for why the compactness fix couldn't fully solve this). This is documented in-code (`STRUCTURE_OVERLAP_FAULT_TYPES = {"low_light", "blur", "tilt"}`) and handled at the decision layer, not inside this detector. A second, **permanent** limitation is the fully-ambiguous case: when the frame's ambient retention collapses below `TAMPERING_AMBIENT_MIN_RETENTION` (near-total structure loss across the whole frame), the detector returns `reason="degraded_ambient"` and the decision engine treats it as a non-measurement (`unavailable`) — never a clean negative, and never a confirmed tampering event, because blur, extreme low-light, and a full-lens obstruction are all consistent with the same observation (§4.7, Approach C).
- **Confidence vs. candidacy (current):** `confidence` is the RAW obstruction-magnitude score — the largest obstruction cluster as a fraction of meaningful blocks, mapped over `[TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION, 1.0]` — and is always preserved on the detector result, even when `is_candidate` is `False`. Candidacy is gated separately by the Approach A checks (ambient above the degeneracy floor; obstruction blocks exist; largest cluster large enough; cluster a genuine depth outlier; cluster compact). Because the values are allowed to disagree, a `raw_confidence` field on the frame log preserves the true value while the *reported/logged* confidence is zeroed whenever `is_candidate` is `False`. See §4.5 for why the alternative "fold candidacy into confidence" fixes were rejected.

### 2.3 Detector — Blur
- **Method:** Variance of Laplacian (`cv2.Laplacian`) vs. per-camera baseline sharpness. Flag as a blur candidate when sharpness drops to ≤50% of the baseline value.
- **Library:** OpenCV. No ML model.
- **Config:** `BLUR_SHARPNESS_DROP_RATIO = 0.5`
- **Why this design:** a sharp image has abundant high-frequency edges → high Laplacian variance; a blurred/smudged-lens image has few → low variance. Comparing to a per-camera baseline (not a fixed threshold) avoids false positives on naturally low-detail scenes (a plain wall) that have low variance even in perfect focus. Chosen over Tenengrad and FFT high-frequency-energy alternatives — no evidence either outperforms Laplacian variance for this problem, and both add complexity/compute without a demonstrated benefit.
- **Known unresolved limitation at the detector level:** camera tilt/movement also produces motion blur, which reduces edge crispness the same way a dirty lens does. A whole-frame Laplacian-variance measurement cannot tell the two causes apart on its own. This is handled today by the decision engine's `tilt → blur = "always"` relation (§2.5), rather than inside the detector itself.

### 2.4 Detector — Tilt
- **Method:** DISK (a pretrained learned local-feature model, via Kornia, GPU-accelerated) extracts keypoints from the baseline and current frame. Keypoints are matched via mutual nearest-neighbor (SMNN) matching. Outliers are rejected using **Median Absolute Deviation (MAD)** on the matched-point displacement magnitudes — **no geometric model (no affine, no homography, no RANSAC) is fit at any point.** The tilt metric is the median displacement of the MAD-inlier points, normalized by the frame diagonal.
- **Library:** Kornia (`kornia.feature.DISK`, pretrained weights from the original DISK paper authors, EPFL cvlab-epfl/disk repo, `depth-save.pth`). GPU/CUDA required for acceptable speed.
- **Config:** `TILT_MAX_KEYPOINTS = 2048`, `TILT_MIN_RELIABLE_MATCHES = 10`, `TILT_MEDIAN_SHIFT_THRESHOLD_RATIO = 0.1`, `TILT_MAD_REJECTION_THRESHOLD = 3.0`, `TILT_MAD_EPSILON = 1e-6`, `TILT_MATCH_RATIO_THRESHOLD = 1.0`. Plus a **reliability floor** added after a critical bug (see §2.4.1): frames need ≥10 matched keypoints **and** a match ratio ≥0.050 of the smaller keypoint set, or tilt is skipped entirely as "unreliable" rather than scored.
- **Why DISK + MAD (and not a geometric model):** every geometric-model-based attempt tried before this (affine transform, homography, homography-filtered-by-RANSAC) proved numerically unstable on this project's sparse real-world keypoint matches — see the full chain of rejected tilt approaches in §3.4. MAD works because it is a robust spread statistic built from medians, so a handful of extreme outlier matches can't drag it around the way they drag a mean or a fitted geometric model; and because it works on the 1-D displacement-magnitude distribution rather than fitting an 8-DOF (homography) or 4-DOF (affine) model to 2-D points, it needs no minimum spatial distribution of keypoints to stay numerically well-behaved — which directly removes the root cause (ill-conditioned fits on sparse/clustered matches) that broke every earlier geometric attempt.

#### 2.4.1 Critical bug found and fixed in the current tilt detector
- **Bug:** severe optical blur caused false tilt signals. Under extreme defocus, DISK produces weak, ambiguous descriptors (match ratio <0.050); matching on these unreliable descriptors generated spurious displacement values that were misread as real tilt.
- **Fix:** added the minimum keypoint match-ratio reliability floor described above — frames below the floor are flagged unreliable and **skip tilt estimation entirely**, rather than being scored against known-bad data. Design rationale: *"sometimes not deciding is better than deciding on unreliable data."*
- **Verification:** 159+ unit/integration tests passing (100%), validated end-to-end on CUDA hardware against synthetic fixtures and real captured footage.

> **Resolved — value confirmed against the codebase:** the constant is `TILT_SHIFT_CONFIDENCE_CEILING_RATIO = 0.15` (defined in `config.py`, used in `detectors/tilt.py`). There is no `TILT_CONFIDENCE_CEILING_RATIO` anywhere in the current codebase. The `0.3` record was the earlier value: the constant was renamed and lowered from `0.3` to `0.15` because a 30%-of-diagonal ceiling compressed real camera rotations into the low half of the confidence scale (a clearly visible ~12%-of-diagonal rotation read as only ~0.4 confidence). The `0.3` value is historical only.

> **Resolved — pipeline integration status confirmed against the codebase:** the advanced tilt-hardening pipeline (sequential pre-filter tree, 3×3-grid spatial-distribution check, homography area/convexity/SVD condition-number gates, and Essential Matrix decomposition) is **not present anywhere in the current codebase — it was never merged.** `detectors/tilt.py` implements only the DISK + mutual-NN + MAD + median-displacement approach described above, and no code anywhere in the repository calls `cv2.findEssentialMat`, `cv2.solvePnP`, or `cv2.findHomography`. The hardening pass was designed on top of the DISK extractor but was superseded by the MAD-based measurement, which removes the root cause (ill-conditioned geometric fits on sparse/clustered matches) without fitting any geometric model. Its sub-checks remain documented as design history in §3.4, not as live behavior.

### 2.5 Decision Engine — Multi-Fault Fusion (current production approach)

**Approach: Relation-class + physical-predicate fusion**, config name `DECISION_SUPPRESSION_RULES`.

Each of the 6 possible fault-type pairs is classified into exactly one of three relation classes:

| Relation class | Meaning | Pairs |
|---|---|---|
| `always` | Deterministic physical link — the first fault always explains the second | `tilt → blur` |
| `conditional` | A link exists only under a per-frame measurable physical condition | `tampering → low_light`, `tampering → blur`, `low_light → blur` |
| `independent` (absent from the map) | No physical pathway exists at all — never suppress | `tampering → tilt`, `low_light → tilt` |

`conditional` pairs are gated by physical predicate functions that read each detector's **own raw measurands** (not just its confidence score):
- **`_area_conserved`** — does an obstruction's measured pixel coverage plausibly account for the measured darkness/blur increase? (Used for `tampering → low_light` and `tampering → blur`.)
- **`_is_near_black`** — is the scene dark enough that blur's own signal is noise, not real focus data? (Used for `low_light → blur`.)

A candidate fault survives fusion into `decision.faults` unless a surviving higher-precedence fault causally suppresses it per its relation class and predicate, **and** clears the relative confidence margin.

**Config:** `DECISION_PRECEDENCE = tampering > low_light > tilt > blur`; `DECISION_CONFIRM_MIN_CONFIDENCE = 0.5` (per-fault emission floor); `DECISION_SUPPRESSOR_MIN_CONFIDENCE = 0.5`; `DECISION_SUPPRESSION_MARGIN = 1.5` (suppressor_confidence × margin ≥ suppressed_confidence); `TAMPERING_LOW_LIGHT_AREA_SLACK = 0.2`; `TAMPERING_BLUR_AREA_SLACK = 0.2`; `LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR = 0.8` (aliased to `DECISION_GATE_CONFIDENCE`, not a separately-tuned number).

**Why this design won over every prior fusion attempt (see §4.2 for the full chain):** every relation class and predicate is derived from actual physical mechanism (rotation resamples pixels and reduces sharpness; a static occlusion cannot move keypoints; darkness cannot move the camera), and every predicate reads quantities the detectors already compute relative to their own camera baseline. This makes the logic **camera- and clip-independent by construction**, not reverse-engineered from any one test video's numbers — an explicit, repeatedly-enforced project rule (two footage-tuned shortcuts were proposed along the way and both were rejected for violating it — see §4.2).

**Verification:** 191/191 tests passing at initial completion (up from a 167 pre-upgrade baseline), including dedicated pair-level tests for all 6 fault combinations; later banner-fix work (§2.7) brought the full suite to 234/234.

> **Honest limitation, documented in-project:** this mechanism is structurally complete and correct, and is verified against real bug evidence — but the specific numeric constants (`DECISION_CONFIRM_MIN_CONFIDENCE = 0.5`, `DECISION_SUPPRESSION_MARGIN = 1.5`, the two area-conservation slack values, the near-black floor) are **reasoned, camera-agnostic starting points, not yet derived from a labeled real-footage calibration dataset.** That calibration work is a separate, undone item — see §5.1.

### 2.6 Decision Engine — Execution Gating (unchanged by design, current)

To avoid running the expensive/unreliable DISK tilt model on frames already known to be badly degraded, a hard execution-gate skip is kept exactly as it was: a detector does not run at all when a stronger co-occurring signal makes its reading unreliable.

**Config:** `DECISION_GATE_CONFIDENCE = 0.8`; `DECISION_GATE_CONFIDENCE_BY_GATE = {"blur": 0.9}` (blur needs a higher bar than the default 0.8 before it's trusted enough to skip tilt — this specifically fixes the false-tilt-under-blur bug); `DECISION_GATE_SKIP_MAP = {"low_light": ("tampering", "tilt"), "blur": ("tilt",)}`.

The only recent change to this layer was **reporting transparency**, not the gating behavior itself: a gate-skipped detector is now explicitly labeled `"unmeasurable"` in output instead of silently vanishing. Two alternatives to this (running the detector anyway with a soft confidence discount; running it anyway with a "less trustworthy" flag consumed by fusion) were both explicitly rejected — see §4.3 for why.

### 2.7 Decision Engine — Snapshot Banner / Reporting Layer (current production approach)

**Approach:** single source of truth for the rendered snapshot banner, sourced entirely from the fusion engine's own `DecisionFrame` fields — never from the legacy V1 flat suppression map. Five mutually-exclusive categories, strict priority order:

`confirmed` (temporally-confirmed, at peak confidence) → `pending` (this frame's live fusion survivors, not yet confirmed) → `suppressed` (cleared its own floor but causally explained by a stronger fault) → `too weak` (never cleared its own confidence floor) → `unmeasurable` (detector was execution-gate-skipped this frame).

A confirmed fault whose detector happens to be gate-skipped on this exact frame gets an inline `[unmeasurable]` marker rather than being listed twice.

**Bug this fixed:** the active list and the suppressed list used to be computed from two independent, mutually-inconsistent rule systems (new fusion rules vs. the old V1 map), and the render code never checked whether they agreed — producing snapshots where the same fault type appeared simultaneously as an active `"FAULT:"` line and a `"suppressed:"` line (e.g., `"FAULT: blur (conf=0.75)"` and `"suppressed: blur"` together). Reproduced from preserved real frame logs on ≥3 documented cases across two different videos/cameras.

**Verification:** replay tests against the original preserved real bug-evidence frames confirmed both original contradiction cases now render cleanly. Full suite 234/234 passing, including a general exclusivity-invariant regression test and end-to-end tests through the real `CameraWorker` + real `DecisionEngine` (not mocked).

**Snapshot timing:** a related fix ensures two faults with genuinely overlapping confirmed windows actually appear together in at least one rendered snapshot. The chosen approach keeps the existing trigger (save a snapshot on any fault's `CONFIRMED` transition, unchanged) but renders the **full current set of confirmed faults** from the temporal confirmation tracker's own already-maintained state, plus a separate "pending" line for the current frame's not-yet-confirmed survivors. Two alternatives (periodic time-based snapshots; a fixed "N seconds" overlap window) were rejected — see §4.4.

**Frame log schema:** bumped to `_SCHEMA_VERSION = 3` to persist the complete multi-label state (`below_floor_faults`, `confirmed_faults` with peak confidences, `raw_confidence`) to JSONL logs.

### 2.8 Current Full Configuration Reference

> Where two documented values conflict for the same constant name, the conflict is listed below with a **resolved** note — every such conflict was checked against the current codebase, and the definitive live value is stated.

**Low light:** `LOWLIGHT_DARK_PIXEL_THRESHOLD = 60`, `LOWLIGHT_BASELINE_DROP_RATIO = 0.5`

**Tampering:** `TAMPERING_GRID_BLOCK_SIZE = 32`, `TAMPERING_MIN_BASELINE_BLOCK_EDGE_DENSITY = 0.02`, `TAMPERING_BLOCK_DENSITY_DROP_RATIO = 0.5`, `TAMPERING_BASELINE_EDGE_PERSISTENCE_RATIO = 0.5`, `TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION = 0.15`, `TAMPERING_MIN_COMPACTNESS_RATIO = 0.60`, `TAMPERING_MIN_MEANINGFUL_BLOCK_FRACTION = 0.50`, `TAMPERING_AMBIENT_RETENTION_QUANTILE = 0.90`, `TAMPERING_OBSTRUCTION_DEPTH_RATIO = 0.35`, `TAMPERING_AMBIENT_MIN_RETENTION = 0.1` (the last three are the Approach A/C constants; the legacy `TAMPERING_MAX_GLOBAL_LOSS_FRACTION = 0.75` was removed)

**Blur:** `BLUR_SHARPNESS_DROP_RATIO = 0.5`

**Tilt:** `TILT_MAX_KEYPOINTS = 2048`, `TILT_MIN_RELIABLE_MATCHES = 10`, `TILT_MIN_MATCH_RATIO = 0.05`, `TILT_MIN_INLIER_RATIO = 0.5`, `TILT_MEDIAN_SHIFT_THRESHOLD_RATIO = 0.1`, `TILT_SHIFT_CONFIDENCE_CEILING_RATIO = 0.15`, `TILT_MAD_REJECTION_THRESHOLD = 3.0`, `TILT_MAD_EPSILON = 1e-6`, `TILT_MATCH_RATIO_THRESHOLD = 1.0`. The reliability floor (`TILT_MIN_RELIABLE_MATCHES` = ≥10 matches **and** `TILT_MIN_MATCH_RATIO` = match ratio ≥0.050 of the smaller keypoint set) and the inlier-ratio floor (`TILT_MIN_INLIER_RATIO` = ≥0.5 of matches must survive MAD rejection) are the two volume guards that make a frame "unreliable" (§2.4).

**Decision engine / fusion:** `DECISION_PRECEDENCE = tampering > low_light > tilt > blur`; `DECISION_SUPPRESSION_RULES` (relation classes, see §2.5); `DECISION_CONFIRM_MIN_CONFIDENCE = 0.5`; `DECISION_SUPPRESSOR_MIN_CONFIDENCE = 0.5`; `DECISION_SUPPRESSION_MARGIN = 1.5`; `TAMPERING_LOW_LIGHT_AREA_SLACK = 0.2`; `TAMPERING_BLUR_AREA_SLACK = 0.2`; `LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR = 0.8` (= `DECISION_GATE_CONFIDENCE`); `DECISION_GATE_CONFIDENCE = 0.8`; `DECISION_GATE_CONFIDENCE_BY_GATE = {"blur": 0.9}`; `DECISION_GATE_SKIP_MAP = {"low_light": ("tampering","tilt"), "blur": ("tilt",)}`; `_SCHEMA_VERSION = 3`.

> **Resolved — checked against the codebase:** the earlier/alternate decision-engine records are **not current**. The live values are: `DECISION_CONFIRM_MIN_CONFIDENCE = {"tampering": 0.5, "low_light": 0.5, "blur": 0.5, "tilt": 0.5}` (a per-fault dict, not a single `0.80`); `DECISION_GATE_CONFIDENCE_BY_GATE = {"blur": 0.9}` (no `low_light` or `default` entries); `DECISION_EXECUTION_ORDER = ("low_light", "blur", "tampering", "tilt")` (the alternate `["low_light", "tampering", "blur", "tilt"]` ordering is not current); `DECISION_SUPPRESSION_MARGIN = 1.5` (multiplicative, not the additive `0.05`).

The separate, older, standalone **persistence tracker** (`PERSISTENCE_SECONDS = 2.0`, `required_frames = fps * persistence_seconds`) does **not** exist in the current codebase — it was superseded by the time-based confirmation window (`DECISION_CONFIRMATION_WINDOW_SECONDS = 3.0` / `DECISION_CONFIRMATION_MIN_POSITIVE_RATIO = 0.50` / `DECISION_CONFIRMATION_MIN_WINDOW_FRAMES = 3`, all present in `config.py`, described in §4.6).

---

## 3. Full Trial-and-Error History, by Component

The rest of this document is the "how we got here." Each subsection lists every distinct approach tried for that component, in the order one replaced another, including the ones that never made it to production. **Read this before proposing a "new" idea for any of these detectors — many obvious-looking ideas below were already tried and have a documented, specific reason they didn't work.**

### 3.1 Low-Light Detector — History

| # | Approach | Status | Result |
|---|---|---|---|
| 1 | Fixed global threshold on mean grayscale brightness (`LOW_LIGHT_BRIGHTNESS_THRESHOLD = 50.0`) | Rejected/Replaced | Worked on the one test video it was tuned on (normal frames 150-190 vs. real lights-off 24.04, cleanly separated). But rejected on design grounds: cameras differ in natural brightness (dim indoor vs. bright outdoor), so one fixed number requires manual per-camera tuning, fighting the project's baseline-relative / no-magic-numbers goal. It also could not distinguish "lights off" from "lens covered" (tampering), since both lower raw brightness — this ambiguity directly motivated moving to a baseline-relative design. |
| 2 | Baseline-relative mean brightness (`Option A`: percentage drop from baseline mean) | Rejected | Still sensitive to a single large dark/bright object entering the frame skewing the whole-frame mean, since the composition problem is only made relative, not fixed. |
| 3 | Percentile brightness vs. baseline (`Option C`: e.g. median or 10th percentile) | Rejected | A reasonable middle ground (resistant to outliers, cheaper than a full histogram) but carries less information for confidence scoring than the histogram approach; not chosen. |
| 4 | **HSV V-channel dark-pixel ratio vs. baseline (`Option B`) — current approach** | Kept-working | See §2.1. Chosen for: robustness to colored lighting (HSV V-channel vs. plain grayscale), robustness to a single bright outlier (percentage of dark pixels vs. a single mean), and a graded (not boolean) confidence signal useful to the fusion layer. Ground-truth validation: low-light window candidate_rate ≈0.80, mean_confidence ≈0.79 (re-measured 0.744/0.673/0.812 across different test runs); clean footage max_confidence ≈0.04-0.14, never crossed the candidate threshold. Automated pytest 3/3 passing (`MIN_TRUE_POSITIVE_CANDIDATE_RATE=0.7`, `MAX_UNRELATED_FALSE_POSITIVE_RATE=0.05`). |

### 3.2 Tampering Detector — History

| # | Approach | Status | Result |
|---|---|---|---|
| 1 | Frame-to-frame differencing, fixed change-percent threshold (`TAMPERING_CHANGE_PERCENT_THRESHOLD = 40.0`) | Replaced | Only caught the moment of transition (start/end of obstruction) — on real footage, only 9/1643 frames flagged, appearing as brief 1-2 frame spikes, not a continuous window. Once an obstruction became the new steady state, frame-to-frame difference dropped back near zero even though the camera was still blocked. |
| 2 | Raw pixel-difference against a fixed baseline frame | Rejected (design stage, never built) | Conflates "lens blocked" with "lighting changed" or "something moved through frame" — considered the weakest of the researched tamper-detection signals. |
| 3 | Histogram correlation (Bhattacharyya distance) vs. baseline | Rejected (design stage, never built) | A textured obstruction (patterned cloth) can produce a histogram not obviously different from a busy real scene; treated only as a supporting signal, not standalone. |
| 4 | **TamperingMonitor** — stable reference frame that only updates during calm periods, compared via frame differencing (`TAMPERING_STABLE_PERCENT_THRESHOLD = 5.0`) | Replaced (was kept-working for a time) | Fixed the sustained-obstruction problem — fault count rose from 9 to 298 frames, correctly capturing both real obstruction events for their full duration. Eventually superseded by the edge-based approach below as the project moved from raw-pixel comparison to structure-loss measurement. |
| 5 | Canny edge detection + per-pixel edge-disappearance ratio vs. baseline (`TAMPERING_EDGE_DISAPPEARANCE_THRESHOLD = 0.5`, 3×3 dilation tolerance) | Replaced | 69% false-positive rate on genuinely clean footage. Root cause: fine repetitive texture (window blinds) produces edges that shift by a pixel or two frame-to-frame from ordinary video-compression noise even with zero real scene change; the small dilation tolerance couldn't absorb this jitter. Confirmed with a debug tool comparing the baseline's own source footage against itself. |
| 6 | Block-based (grid) edge-density comparison vs. baseline | Replaced | Fixed the noise-floor problem — clean-footage false positives dropped from 69% to 1.92%, real tampering still detected (~62.5% candidate rate). But low-light (80.3%) and blur (74.4%) still triggered at high rates, because both genuinely weaken/destroy Canny's edge signal — a real structural overlap, not measurement noise. |
| 7 | **Block-based edge density + connected-component clustering (current approach)** | Kept-working, with a documented unresolved partial limitation | See §2.2. Requires the largest contiguous cluster of lost blocks to be both large enough and compact. Real tampering detection remained correct (~63% candidate rate, correct segment boundaries) but the added clustering filter did **not** meaningfully reduce low-light (80.3%) or blur (76.7%) overlap, because a *uniformly*-degraded frame is just as spatially "contiguous" as one real localized object — contiguity distinguishes scattered-vs-clustered, not "one object's footprint" vs. "the whole frame degraded." A decision was made **not** to iterate further on this detector in isolation; the overlap is documented (`STRUCTURE_OVERLAP_FAULT_TYPES`) and handled at the decision engine, where it is now resolved via the `_area_conserved` physical predicate (§2.5). |
| 8 | **Relative ambient-retention model (Approach A) — replaces the absolute total-loss ceiling (current)** | Kept-working | The legacy detector also capped the TOTAL fraction of lost structure at an absolute ceiling (`TAMPERING_MAX_GLOBAL_LOSS_FRACTION = 0.75`). Real cam_08 footage exposed the bug: a large real obstruction (~83% of meaningful blocks) co-occurring with severe blur wiped out the remaining structure, total loss reached ~0.83, and the ceiling wrongly rejected the frame — `is_candidate=False` — even though the obstruction was one compact, physically meaningful cluster. The fix replaces the absolute ceiling with a **relative** comparison: each meaningful block is scored by its retention ratio (current/baseline edge density); the frame's own ambient level is the upper-tail quantile (0.90) of those ratios; an obstruction block is one whose retention is at most 0.35× the ambient level. A real obstruction stays a deep outlier even when blur collapses the ambient level; pure global degradation flags nothing (every block sits *at* ambient). The ratios are dimensionless and within-frame, so the model is not footage-tuned. Confidence keeps the legacy magnitude scale. See §4.7. |
| 9 | **Degraded-ambient non-measurement + contamination resolution (Approach C) — engine-side handling (current)** | Kept-working | The fully-ambiguous case remains: if a frame's ambient retention collapses below the degeneracy floor (`TAMPERING_AMBIENT_MIN_RETENTION = 0.1`), the relative comparison is meaningless and the detector must NOT confidently confirm. `evaluate()` returns `reason="degraded_ambient"` (never a candidate); the decision engine packages it as a per-frame non-measurement (status `unavailable`, excluded from the temporal tracker's observed set) and resolves it against the same frame's blur/low_light (severe ⇒ `degraded_ambient_explained`; otherwise `degraded_ambient_unexplained`). Both outcomes deliberately never confirm tampering and never read as a clean "no tampering" negative. See §4.7. |

### 3.3 Blur Detector — History

| # | Approach | Status | Result |
|---|---|---|---|
| 1 | Laplacian variance, fixed threshold (`BLUR_VARIANCE_THRESHOLD = 100.0`) | Replaced | On real footage, 286/1643 frames flagged, with false-positive windows exactly matching the real hand-cover and lights-off events. A dark or obstructed frame also has very little edge detail — not because the lens is dirty, but because there's nothing visible to measure. Sharpness-only measurement can't distinguish "no detail because too dark to see" from "no detail because the lens is smudged." |
| 2 | Brightness-gated Laplacian variance (`detect_dirty_lens_refined`) — skip sharpness judgment if brightness is below a minimum (`min_brightness_for_check = 60.0`, chosen just above the low-light threshold as a safety margin) | Replaced | False-flag count dropped from 286 to 119; lights-off overlap eliminated entirely; hand-cover overlap reduced from a full ~2.8s window to a brief ~0.3s sliver. But camera-tilt overlap (~45.3s-46.0s) was completely unaffected, since tilt-induced motion blur is a genuine sharpness drop unrelated to lighting. |
| 3 | Motion-gating via frame differencing (`detect_dirty_lens_refined_v2`, proposed `max_motion_for_check = 15.0`) | **Never implemented** — deferred | Drafted but deliberately not built: the proposed `15.0` threshold had no grounding in any validated project data (unlike the brightness threshold, which reused an already-validated constant), and building it risked overfitting a guessed number to one test video. Work stopped here rather than accept an unvalidated number. |
| 4 | Tenengrad (Sobel-gradient sharpness) | Rejected (design stage) | No evidence it meaningfully outperforms Laplacian variance for this problem. |
| 5 | FFT high-frequency energy analysis | Rejected (design stage) | Meaningfully heavier compute than Laplacian variance (FFT vs. one convolution+variance) with no demonstrated benefit. |
| 6 | **Variance of Laplacian, baseline-relative (`BLUR_SHARPNESS_DROP_RATIO = 0.5`) — current approach** | Kept-working | See §2.3. Ground truth: blur window candidate_rate ≈0.76, clean footage ≈0.03. Documented overlaps: tampering ≈0.71-0.81, low_light ≈0.81, tilt ≈0.26-0.51 (all handled at the decision layer, not treated as detector-level false positives). The tilt/motion-blur overlap that motion-gating (#3) was meant to fix remains a documented gap at the individual-detector level — it is instead handled today by the decision engine's `tilt → blur = "always"` relation (§2.5), which was the eventual resolution. |

### 3.4 Tilt Detector — History (the most heavily-iterated component in the project)

This detector went through more failed attempts than any other component. The short version: **every classical/geometric approach failed on this project's real, low-texture, sparse-match footage.** The eventual fix was to abandon geometric model-fitting entirely.

| # | Approach | Status | Result |
|---|---|---|---|
| 1 | ORB (hand-crafted keypoints) + `cv2.estimateAffinePartial2D` + RANSAC, fixed angle threshold (`ANGLE_CHANGE_THRESHOLD_DEGREES = 15.0`) | Rejected | Wildly inconsistent rotation estimates on real continuous footage (e.g., -36.95°, -113.98°, -156.09°, 94.00° across nearby frames), reliable-match counts as low as 5-8 out of 120-140 (vs. 356/379 on a clean simulated pair). Also crashed outright (`cv2.error`) when a frame's ORB descriptors returned `None`. A later integration run falsely flagged 450/500 frames (expected ~1). Root cause: ORB's hand-crafted corner features have too few detectable points on low-texture real-world regions to support a reliable RANSAC fit. Crash-prevention patches (a None-descriptor safety check, a try/except wrapper) stopped the crash but did not fix the underlying unreliability. |
| 2 | Optical flow (dense pixel motion, proposed as a classical alternative to ORB) | Rejected before implementation | Reasoned (not tested) that optical flow is still a classical technique measuring pixel movement and would likely struggle on the same low-texture frames that broke ORB. |
| 3 | End-to-end learned rotation/pose regression network (single model, direct angle output) | Rejected (design stage) | No reputable off-the-shelf pretrained model exists that performs "camera-mount tilt angle in degrees" as its trained task; training a custom model was judged disproportionate for one detector. |
| 4 | Hybrid: classical ORB as default, learned model (XFeat) as fallback only when ORB fails | Rejected (design stage) | Since ORB was already known to fail on this project's real footage, "try ORB first" would just delay building the actual working solution. |
| 5 | XFeat via Kornia's top-level convenience function (`kornia.feature.match_xfeat`) | Rejected | `AttributeError: module 'kornia.feature' has no attribute 'match_xfeat'` — the function exists in Kornia's latest/dev docs but not in the installed release (Kornia 0.8.3). |
| 6 | XFeat via Kornia's class-based API (`kornia.feature.XFeat` + `match_smnn`) | Rejected | Returned **0 keypoints** on every frame, including the baseline matched against itself, and even on pure random-noise tensors using Kornia's own official doctest example parameters. Model weights were confirmed loaded and non-zero — diagnosed as a genuine bug/immaturity in Kornia 0.8.3's XFeat integration (added in the very latest patch release at the time), not something further project-side debugging was likely to fix. |
| 7 | **DISK via Kornia (feature extraction only)** | Kept-working as the extractor | Loaded correctly and found 1500-2000+ real keypoints per frame on GPU — the first working keypoint extractor after both XFeat variants failed. The downstream transform/scoring method was iterated on next (see below). |
| 8 | DISK + affine transform (`cv2.estimateAffinePartial2D`) for rotation angle, threshold `ANGLE_CHANGE_THRESHOLD_DEGREES = 15.0` (also recorded as `TILT_ANGLE_THRESHOLD_DEGREES` in one record) | Replaced | `is_candidate` was `False` for **every single frame in the entire video, including the real tilt window** — max angle measured anywhere in the tilt window was 13.89°, just under the 15° threshold. Two compounding root causes: (1) the threshold was set slightly high relative to the true measured max; (2) more fundamentally, an affine transform can only represent in-plane rotation (roll) and cannot represent the perspective change from pitch/yaw (tilting up/down or turning sideways) — the mathematically wrong model for a camera rotating in place. Also, real match counts dropped ~10x during the actual tilt event (from ~1342 to 75-103), which a too-strict match-ratio threshold (0.9) was discarding. |
| 9 | DISK + homography (`cv2.findHomography`, RANSAC) + corner-displacement measurement, normalized by frame diagonal (`TILT_CORNER_SHIFT_THRESHOLD_RATIO = 0.1`) | Replaced | Correctly captured the real tilt window (candidate_rate ≈0.86, matching the true tilt segment) after loosening the match-ratio threshold to 1.0. BUT tampering (≈0.59-0.60) and blur (≈0.51-0.61) candidate rates were also high, with some corner-shift values reaching **30-43x** the threshold — larger than even the real tilt event's own max (~28x). Root cause: a homography fit on sparse/spatially-clustered matches (common during tampering/blur, where few reliable correspondences exist) can be numerically ill-conditioned; evaluated at the 4 frame corners (far from the small clustered patch the fit was based on), this produces enormous, physically meaningless "extrapolation explosion" displacement values — noise misread as an extreme framing change. |
| 10 | DISK + homography + post-fit validation gate (SVD condition-number check `κ(H) > 100` rejected, convexity check, area-ratio check `0.5`-`1.5`) | Rejected | Correctly rejected the clearly-degenerate false-positive fits — but **also rejected the real tilt frames**, which had condition numbers just as extreme (in the millions) as the fault-driven false positives. This proved the "correct-looking" results from the prior homography version were numerically accidental, not from a genuinely stable fit: DISK produced only 75-103 raw matches during the real tilt event — too sparse/clustered to produce a stable 8-DOF homography under any circumstance, real tilt or not. This finding led to abandoning homography as the *measurement* entirely. |
| 11 | DISK + RANSAC used *only* as an outlier filter on matched points (homography matrix itself discarded) + median displacement of RANSAC-inlier points | Rejected | Clean, tampering, and blur frames were correctly quiet — a genuine improvement. But real tilt frames failed outright with "Only 6 RANSAC inliers (need 10)," despite 75-103 raw matches being available. Root cause: RANSAC's internal outlier test is itself based on fitting the same already-proven-degenerate homography model, so it was still silently discarding real, correct tilt matches as "outliers" — the same root problem, just manifesting as a match-count rejection instead of a wild displacement number. |
| 12 | **DISK + Median Absolute Deviation (MAD) outlier rejection on displacement magnitudes, no geometric model anywhere — current approach** | Kept-working | See §2.4 for full description and the critical-bug fix layered on top of it later. Single-frame diagnostic: 103 raw matches at a real-tilt timestamp produced median displacement 202.2px, MAD 62.74, correctly retaining 92/103 as inliers (vs. only 6/103 surviving the prior RANSAC approach on the *same* data). Full ground-truth validation across all four fault windows plus clean footage passed (`MIN_TRUE_POSITIVE_CANDIDATE_RATE=0.7`, `MAX_UNRELATED_FALSE_POSITIVE_RATE=0.05`), combined 4-detector suite 12/12 tests passing. |
| 13 | Essential Matrix (E) decomposition with camera intrinsics (`cv2.findEssentialMat`) for true calibrated pitch/roll/yaw | Rejected as not actionable in one design pass; confirmed **never implemented** | Requires camera intrinsic parameters (focal length, principal point) that the project has no calibration step to obtain — flagged as "aspirationally correct but not something we can build right now without changing the project's scope." The later, more detailed design (the "tilt-hardening pipeline") was also **never merged** — no Essential-Matrix/homography code exists anywhere in the current codebase (confirmed, §2.4.1). |
| 14 | PnP (Perspective-n-Point) | Rejected as not actionable | Requires known 3D-to-2D correspondences (a real 3D scene model or calibration object) that the project has neither of and is not scoped to obtain for arbitrary customer cameras. |
| 15 | Vanishing-point / horizon-based absolute tilt estimate | Rejected | Proposed as a way to measure tilt absolutely (from a single frame) rather than relative to a baseline, avoiding "bad reference frame" problems. Ultimately rejected: high complexity and vulnerable to failure in unstructured/low-feature scenes (blank walls, sky) since it depends on visible parallel-line structure, and it's not suited to a fixed-geometry security camera that already has a known baseline frame. |

**A second tilt-hardening pass**, layered on top of the DISK-based measurement to further reduce false positives from tampering/blur (measured at the time: tampering false-positive rate 59%, blur false-positive rate 61%, with corner-shift values 30-43x threshold vs. real tilt's ~28x max):
- **Homography inlier-ratio check alone** — rejected: sparse matches localized in a small patch can still achieve a high inlier ratio (e.g., 10/12 = 83%) while remaining geometrically degenerate; inlier ratio doesn't account for spatial distribution.
- **3×3 grid spatial-distribution check** (require matches spread across ≥3-4 of 9 grid sectors) — kept-working: prevents ill-conditioned fits caused by matches tightly clustered in one small region.
- **Homography area & convexity validation** (reject if transformed corner-quad area ratio is `<0.5` or `>1.5` of the original) — kept-working: true tilt causes subtle perspective shifts; degenerate homographies stretch/invert/distort the frame quad beyond physical limits.
- **Keypoint drop-count / feature-density check** (DISK) — kept-working, used as an early-exit signal: a >70% keypoint drop vs. baseline indicates lens obstruction, letting tampering be ruled out before geometric tilt processing runs.
- **Sequential pre-filter decision tree** (blur check → tampering check → spatial-spread check → geometric tilt evaluation only if the frame passes all three) — kept-working: ensures the geometric transform only evaluates frames the system already believes are healthy.

*(Confirmed against the codebase — this hardening pass and the Essential-Matrix decomposition step were **never merged**. The current `detectors/tilt.py` is the DISK+MAD approach only; these sub-checks are design history, not live behavior — see §2.4.1.)*

---

## 4. Decision Engine — Full Trial-and-Error History

### 4.1 Single-Fault Era (superseded)
- **Independent per-detector thresholding (earliest state)** — each of the 4 detectors ran completely independently with no cross-checking. Real-footage testing surfaced the project's most important early finding: a single physical event (e.g., full hand-cover) reliably triggers multiple detectors at once (hand-cover → low_light + tampering; lights-off → low_light + tampering; plastic-wrap/blur → blur + tampering; camera tilt → tampering + blur).
- **Proposed combined/fusion classifier** (pattern-match across all 4 detector readings to infer one most-likely cause) — **rejected before implementation**, in favor of narrower, single-component fixes (e.g., blur's brightness-gating) instead of building a general classifier at that time.
- **Precedence-ranked single-primary fusion** (`DECISION_PRECEDENCE` + a static boolean `DECISION_SUPPRESSION_MAP`, one winning fault per frame) — became the first *actually implemented* decision layer. Retained today only as a legacy, test-only code path (12+ existing tests depend on it); confirmed **runtime-dead in production** since the multi-label upgrade. By construction it can only ever report one fault per frame, so it could not represent two genuinely independent simultaneous faults (e.g., tilted AND obstructed at once) — this limitation is what forced the multi-label upgrade below.

### 4.2 Multi-Fault Fusion — Rejected Intermediate Attempts
- **Naive multi-label collection** (gather every detector clearing its own confidence floor, apply only execution-gate skips, no suppression at all) — an early draft attempt. **Rejected:** would report a fault (e.g. low_light) alongside a fault it causally explains (e.g. tampering causing the apparent darkness) as if both were independent — produces "noise, not the fix."
- **Static-map multi-label filter** (`resolve_active_faults()` — keep the existing flat boolean `DECISION_SUPPRESSION_MAP`, but *filter* rather than pick-one-winner) — mechanism was sound and correct (byte-identical output to the single-fault version on single-fault frames, verified via regression tests), but **insufficient in scope**: under the existing map, every one of the 6 fault pairs was covered by *some* suppression edge, so true multi-label co-occurrence only ever emerged as a side effect of a margin check failing — not from genuine physical independence. The requirement was for all 6 pairs to be able to co-occur *generally* when physically true.
- **Footage-tuned emission-floor lowering** (lower `low_light`'s confidence floor from 0.5 to ~0.25 because one specific demo clip measured 0.29-0.36) — **rejected.** The number was reverse-engineered to fit one clip, not derived from any general principle; it would silently make the whole system more sensitive to low_light noise on every other camera globally, since the constant is shared. Rejected per an explicit project rule against any fix that only works for the specific demo footage.
- **Global margin/threshold loosening** (lower `DECISION_SUPPRESSION_MARGIN` toward 1.0, or raise `DECISION_SUPPRESSOR_MIN_CONFIDENCE` toward 0.8) — **rejected.** Analysis showed it wouldn't even solve the target cases (doesn't help tilt+low_light, which is blocked by the emission floor, not the margin; unreliable for tampering+tilt, since real tampering confidence 0.6-1.0 usually still clears even a loosened margin) while broadly weakening suppression everywhere else and touching arithmetic hardcoded into ≥4 existing tests — high blast radius for low payoff.
- **Static suppression-map edge deletion only** (remove just the `tampering→tilt` and `low_light→tilt` edges, treating them as always-independent) — **kept-working for the two pairs it targeted** (these two have zero physical causal link — a static obstruction can't move the camera, and darkness can't move the camera, so unconditional "never suppress" is correct and general), but **insufficient alone**: the remaining 4 pairs have a real causal link *sometimes but not always* depending on the actual scene each frame, and a boolean edge (present or absent) cannot express that conditionality. This fix was preserved as the "independent" relation class inside the final system (§2.5), not discarded.
- **Reclassify `tampering→low_light` as independent unless obstruction coverage exceeds an unvalidated 80%, plus vague instructions to "tune" detector floors** — **rejected.** The diagnosis behind this proposal was factually wrong (claimed detectors were "firing correctly but being suppressed," when database evidence showed tampering was never firing at all — blocked by a degraded-baseline guard, unrelated to suppression). The proposed 80% number had no derivation (same unjustified-magic-number pattern already rejected once); "tune detector floors to real-world test conditions" is functionally identical to hardcoding thresholds to pass one test video.
- **Relation-class + physical-predicate system → current production approach.** See §2.5.

### 4.3 Execution-Gating Fix — Rejected Alternatives
When the false-tilt-under-blur bug (§2.4.1) was traced partly to a single static gate confidence floor applied uniformly to all gate detectors (unable to express that blur needed its own, higher floor than low_light before it could justify skipping tilt), three fixes were considered:
- **Option 3-b — soft confidence discount** (run the gated detector anyway, multiply confidence by a reliability factor) — **rejected:** defeats the entire purpose of the gate (the expensive DISK model would run on every frame regardless), and a hand-picked discount curve risked reintroducing the exact false-tilt spikes the gate exists to prevent.
- **Option 3-c — run anyway + attach a "reliability/unmeasurable" flag for fusion to weigh** — **rejected (deferred):** still pays the full compute cost of running the gated detector, and adds real complexity to fusion (a new flag type it must consume) for a benefit judged not clearly necessary given a simpler alternative existed.
- **Option 3-a (chosen) — keep the hard skip exactly as-is; only add reporting transparency** (`"unmeasurable"` label). See §2.6. Chosen because it's the only option that fixes the actual complaint (silent, unexplained omission) at zero cost to the existing, working performance/robustness tradeoff.
- Separately, before per-gate floors existed, a **single static `DECISION_GATE_CONFIDENCE`** threshold shared by all gate detectors was the original (now-replaced) design; it could not express that blur needed a different (0.9) floor than low_light (0.8) to justify skipping tilt, which is why `DECISION_GATE_CONFIDENCE_BY_GATE` was introduced.

### 4.4 Snapshot Timing Fix — Rejected Alternatives
To guarantee overlapping confirmed faults actually appear together in a rendered snapshot:
- **Option 1-b — periodic time-interval snapshots**, independent of confirmation events — **rejected:** adds ongoing, unbounded extra I/O/storage cost and competes with genuine confirmation-transition snapshots for a limited ring-buffer eviction budget.
- **Option 1-c — supplemental snapshot when a fault confirms within N seconds of another's active window** — **rejected:** the existing `DECISION_MIN_EVENT_GAP_SECONDS` mechanism can delay a second fault's confirmation past any reasonable fixed "N seconds," so this could still miss the exact overlap case it targets — a known, documented gap in the approach itself.
- **Option 1-a (chosen)** — keep the existing confirmation-transition trigger, but render the full current confirmed-fault set from the tracker's own state. See §2.7.

### 4.5 Tampering Confidence/Candidacy Logging Fix — Rejected Alternatives
When a real synthetic test frame showed tampering confidence=1.0 simultaneously with `is_candidate=False` (because coexisting blur pushed total structure loss over its ceiling even though the obstruction itself was large):
- **Option 4-1 — fold the `is_candidate` guards into the confidence computation itself** (hard zero, or a continuous penalty) — **rejected:** confidence is also the fusion suppression weight, so a hard 0/1 multiplier would silently change real suppression behavior (tampering could no longer ever suppress blur on high-total-loss frames) — a detection-behavior change disguised as a display fix. A soft penalty curve would reintroduce hand-picked, footage-tuning-risk values.
- **Option 4-2 — one canonical decision path** (compute the "localized" guard first; force confidence to 0.0 immediately if it fails) — **rejected:** loses all severity information for "large loss but not localized" frames, making them indistinguishable in logs from frames with genuinely zero tampering signal — judged too destructive to diagnostic value for a fix whose actual goal was just an honest log.
- **Option 4-3 (chosen)** — keep confidence and `is_candidate` as independently-meaningful values; add a `raw_confidence` field that always preserves the true value; zero the *reported* confidence only when `is_candidate` is `False`. See §2.2. Verified inert everywhere except the log/display path by auditing every real read site of `DetectorObservation.confidence` in the codebase (fusion, gating, confirmation tracker, banner candidate list) and confirming all of them already short-circuit on `is_candidate` before reading confidence.

### 4.6 Confirmation-Window Mechanism — History
- **Time-based sliding window** (`DECISION_CONFIRMATION_WINDOW_SECONDS = 3.0`, `DECISION_CONFIRMATION_MIN_POSITIVE_RATIO = 0.50`, `DECISION_CONFIRMATION_MIN_WINDOW_FRAMES = 3`) — tried first.
- **Frame-count-based window** (`CONFIRMATION_WINDOW_SIZE = 30` frames, `CONFIRMATION_THRESHOLD = 0.50`) — tried as an alternative basis, then **reverted** back to the time-based version (same numeric values as originally).
- The reverted time-based version is the one currently kept-working. The specific reason the frame-count version was abandoned in favor of reverting is not documented and should be confirmed with the team if it matters for future work.
- Separately, in an earlier project phase before this decision-engine confirmation window existed, a **standalone persistence tracker** used `PERSISTENCE_SECONDS = 2.0` (`required_frames = fps × persistence_seconds`). The 2-second window was chosen specifically because it's shorter than the real ~7-8 second gap between distinct fault events in the test footage (avoiding "bleed" between consecutive faults) while still being long enough that brief 1-frame noise spikes don't accumulate to a false confirmation. Verified against the codebase: this standalone tracker was **superseded** — `PERSISTENCE_SECONDS` exists nowhere in the current code, and the 3.0-second time-based window in `config.py` is the live confirmation mechanism (see §2.8).
- **Per-frame candidate confidence-floor gating into the confirmation window** (`DECISION_CONFIRM_MIN_CONFIDENCE = 0.80` in this specific usage — note this conflicts with the `0.5` "per-fault emission floor" value used elsewhere for a same-named constant; verified against the codebase, the `0.80` record was superseded, see the resolved note in §2.8) — an additional noise filter requiring a minimum per-frame confidence before a candidate is even allowed to enter the confirmation window at all.

### 4.7 Tampering Candidacy Redesign — Approach A (Ambient-Retention Model) and Approach C (Contamination Handling)

The complete fix for the tampering candidacy bug, implemented across subtasks 6-9 of the two-part
effort. This section records the decision, the alternatives considered, and the honest limits.

**The real bug (cam_08, `tampering_meaningful_block_fraction` ≈ 0.686):** a large physical
obstruction (~83% of the frame's meaningful blocks, one compact contiguous cluster) co-occurred
with severe blur (confidence ~0.915-0.955 on the real frames). The legacy detector measured total
structure loss ~0.83, which exceeded its absolute ceiling `TAMPERING_MAX_GLOBAL_LOSS_FRACTION =
0.75`, so it returned `is_candidate=False` even though the obstruction was a textbook
tampering signature. The absolute ceiling — meant to reject "the whole frame degraded" — could
not tell "one big obstruction, rest of frame blurred" (legitimate tampering) from "whole frame
blurred" (not tampering), because both can produce total loss above 0.75.

**Approach A (chosen): relative ambient-retention obstruction model.**
Replace the absolute ceiling with a comparison to the frame's OWN ambient retention level:

- retention ratio per meaningful block = current edge density / baseline edge density;
- ambient level = upper-tail quantile (0.90) of retention ratios. The obstruction sits in the
  lower tail, so the ambient estimate stays at the true ambient level while obstruction coverage
  stays below (1 − quantile) ≈ 90% of the frame;
- obstruction block = retention ≤ 0.35 × ambient level;
- candidacy additionally requires the largest obstruction cluster to pass the existing size and
  compactness gates and to be a genuine depth outlier relative to ambient.

Why it is general rather than footage-tuned: the two ratios are dimensionless and compare blocks
WITHIN a single frame, so they cannot drift with camera exposure, resolution, or scene texture;
the same constants apply identically to any camera/baseline window. Why it fixes the bug: with
co-occurring blur the ambient level itself collapses, but the obstruction still sits ~0.35×
below it, so it stays a statistical outlier and the frame becomes a candidate. Pure global
degradation flags nothing (every block is AT ambient). Confidence intentionally keeps the legacy
magnitude scale (largest cluster / meaningful blocks, mapped over
`[TAMPERING_MIN_CONTIGUOUS_BLOCK_FRACTION, 1.0]`), so emission floors and decision-layer
suppression behavior are unchanged.

Alternatives considered and rejected for the detector-level gate:
- **Keep the absolute ceiling but lower it** — **rejected.** Any fixed number has the same
  structural flaw; it would have to be re-tuned per obstruction size and per blur severity, and a
  lower number makes the "whole frame degraded" false-confirmation worse.
- **Detect obstruction on a blur-independent signal (e.g., texture vs. brightness)** — **rejected
  as a fix for this bug.** It would be a new detector, not a candidacy fix, and the project's
  established architecture keeps cross-fault ambiguity at the decision layer, not inside one
  detector.

**Approach C (chosen): degraded-ambient non-measurement + contamination resolution.**
The relative comparison is only meaningful while the frame retains structure somewhere. When the
frame's ambient retention collapses below the degeneracy floor `TAMPERING_AMBIENT_MIN_RETENTION =
0.1`, the frame is information-theoretically ambiguous for tampering: near-total structure loss
across the whole frame is consistent with a full-lens obstruction, extreme blur, or extreme
low-light. Handling:

- the detector returns `reason="degraded_ambient"` (never a candidate);
- the decision engine packages it as a per-frame **non-measurement** (status `unavailable`,
  excluded from the temporal confirmation tracker's observed set) — so it never dilutes
  `window_positive_rate` as a false negative, and never confirms;
- the engine resolves the collapse against the same frame's blur/low_light: severe (blur ≥ 0.9,
  its existing gate floor; low_light ≥ 0.8, the near-black floor) ⇒ `degraded_ambient_explained`
  ("contamination genuinely wiped the structure out"); otherwise `degraded_ambient_unexplained`.
  The distinction is diagnostic only; both are non-measurements.


**Permanent, honest limitation (not a bug to be fixed later):** a near-total structure collapse
across the whole frame still cannot be CONFIRMED as tampering by any structure-loss-based
approach — with essentially no structure left anywhere there is nothing to measure, and the
observation is consistent with several faults. The system's honest behavior is to surface the
frame as unmeasurable rather than guess. This is inherent to the signal, not a deficiency in this
implementation.

**Calibration status (consistent with the project's documentation standard):** the three new
constants — `TAMPERING_AMBIENT_RETENTION_QUANTILE` (0.90), `TAMPERING_OBSTRUCTION_DEPTH_RATIO`
(0.35), and `TAMPERING_AMBIENT_MIN_RETENTION` (0.1) — are reasoned, empirical starting points,
chosen to match the known real cam_08 bug case and the physics, but they are **not yet validated
against a labeled dataset**. The separate emission-floor calibration question (whether
`DECISION_CONFIRM_MIN_CONFIDENCE = 0.5` is the right per-fault confirmation threshold, see §2.8's
resolved note) is a **distinct, still-open item**, unaffected by this fix.

**Real-footage re-validation (subtask 10, `tools/revalidate_cam08_tampering.py`):** re-running
`tampering.evaluate()` directly on the real cam_08 frames —
- bug case t ≈ 23.5-23.7 s (frames 570-573): `is_candidate=True` (was False), confidence
  0.667-0.681, blur confidence 0.915-0.955, meaningful block fraction 0.686; the legacy model
  would still reject these frames (total loss 0.83 > 0.75);
- normal localized-obstruction case t ≈ 36.0-37.0 s: candidate on 15/15 frames, identical to the
  legacy model (15/15) — regression unchanged;
- whole-clip candidate windows: t = 22.12-24.40 s (n=56, conf 0.06-0.76), t = 24.56-24.64 s
  (n=3, conf 0.59-0.64), t = 24.73-28.48 s (n=92, conf 0.08-0.32), t = 35.83-38.51 s (n=66,
  conf 0.03-0.33); no `degraded_ambient`/`degraded_baseline` non-measurements occurred in this
  clip (the un-obstructed regions always retained measurable structure).


---

## 5. Known Open Issues / Honest Limitations (as of the latest verified state)

1. **Calibration of the fusion-layer and tampering constants is not done.** `DECISION_CONFIRM_MIN_CONFIDENCE = 0.5`, `DECISION_SUPPRESSION_MARGIN = 1.5`, `TAMPERING_LOW_LIGHT_AREA_SLACK = 0.2`, `TAMPERING_BLUR_AREA_SLACK = 0.2`, `LOW_LIGHT_BLUR_NEAR_BLACK_FLOOR = 0.8`, and the Approach A/C constants (`TAMPERING_AMBIENT_RETENTION_QUANTILE = 0.90`, `TAMPERING_OBSTRUCTION_DEPTH_RATIO = 0.35`, `TAMPERING_AMBIENT_MIN_RETENTION = 0.1`) are reasoned, camera-agnostic starting points, **not** derived from a labeled real-footage dataset. A rough estimate for doing this properly: on the order of hundreds of positive/negative frames and tens of confirmed events per fault class per camera, k-folded **across cameras** (not just across frames of one clip). This was deliberately deferred rather than closed out with a footage-tuned shortcut.
2. ~~Reconciliation needed between the DISK+MAD tilt detector (§2.4) and the separate, more advanced tilt-hardening pipeline~~ — **RESOLVED (verified against the codebase):** the hardening pipeline (spatial-spread check, homography area/convexity/condition-number gates, Essential Matrix decomposition, sequential pre-filter tree) was **never merged**. `detectors/tilt.py` is purely DISK + mutual-NN + MAD + median displacement, and the repository contains no `cv2.findEssentialMat` / `cv2.solvePnP` / `cv2.findHomography` code anywhere. The design remains documented as rejected/superseded history in §3.4.
3. ~~Several numeric config constants are recorded with conflicting values under the same name~~ — **RESOLVED (verified against the codebase):** every conflicting-value open item from §2.8 and §2.4.1 was checked against `config.py`, and the definitive live values are stated there. Summary: `TILT_SHIFT_CONFIDENCE_CEILING_RATIO = 0.15` (the `0.3` record was the earlier value, §2.4.1); `DECISION_CONFIRM_MIN_CONFIDENCE = {"tampering": 0.5, "low_light": 0.5, "blur": 0.5, "tilt": 0.5}`; `DECISION_GATE_CONFIDENCE_BY_GATE = {"blur": 0.9}`; `DECISION_EXECUTION_ORDER = ("low_light", "blur", "tampering", "tilt")`; `DECISION_SUPPRESSION_MARGIN = 1.5` (multiplicative). All `0.80` / `0.05` / extra-entry records were superseded.
4. **The reason the frame-count-based confirmation window (§4.6) was tried and then reverted back to time-based is not documented.**
5. **Tampering vs. blur/low-light structural overlap is a known, accepted, permanent limitation at the individual-detector level** (§3.2, item 7) — a uniformly degraded frame is indistinguishable from a real localized obstruction using contiguity alone. This is intentionally handled only at the decision-engine layer (`_area_conserved` predicate), not inside the tampering detector itself.
6. **The fully-ambiguous case is a permanent limitation of any structure-loss approach, not a bug to be fixed later:** a near-total structure collapse across the whole frame (ambient retention below `TAMPERING_AMBIENT_MIN_RETENTION = 0.1`) cannot be *confirmed* as tampering — blur, extreme low-light, and a full-lens obstruction are all consistent with the same observation. The system deliberately surfaces such frames as non-measurements (`degraded_ambient`, status `unavailable`) rather than guessing (§4.7).
7. **The emission-floor calibration question remains a distinct, still-open item:** whether `DECISION_CONFIRM_MIN_CONFIDENCE = 0.5` (the verified current per-fault floor, see §2.8) is the right per-fault confirmation threshold has not been resolved by the tampering fix, which does not change any emission floor. (The earlier-recorded `0.80` value for the same-named constant was verified to be a superseded alternate record.)

---

## 6. Testing & Validation Methodology

- **Ground-truth real-footage validation:** a real recorded test video with known fault windows (approximate real timestamps for each fault type) was used repeatedly to compute `candidate_rate` (fraction of frames in a fault's true window that the detector flagged) and false-positive rate (candidate rate on clean, no-fault footage).
- **Standard automated test acceptance bar used per-detector:** `MIN_TRUE_POSITIVE_CANDIDATE_RATE = 0.7` (detector must catch ≥70% of frames in its own real fault window) and `MAX_UNRELATED_FALSE_POSITIVE_RATE = 0.05` (≤5% false positives on clean footage). Cross-detector overlap (e.g., tampering also triggering low_light) was **explicitly exempted** from the false-positive assertion via a documented `STRUCTURE_OVERLAP_FAULT_TYPES` set, rather than being hidden or ignored.
- **Single-frame / small-sample diagnostics:** before committing to a full ground-truth run, several tilt-detector iterations were first spot-checked on a handful of hand-picked frames (one per fault zone plus one clean) to catch obviously broken behavior cheaply.
- **Regression/replay testing against preserved real evidence:** the banner-rendering bug fix (§2.7) was verified specifically by replaying the *actual* preserved frames that first exposed the contradiction, not just new synthetic cases — considered stronger proof than a newly written test alone.
- **Full-suite regression tracking as a running scoreboard:** the project tracked a single "full suite passing" count across the whole codebase at each milestone as a coarse regression signal. Historical series: 12/12 (early 4-detector suite) → 43/43 → 103/107/114/129/140 (various decision-engine milestones) → 163/164 → 191/191 → 234/234 (post banner-fix; the "standard suite" = all tests except the 3 slow tilt-detector tests) → **259/259 (after the two-part effort's subtasks 1-5, baseline-observability)** → **290/290 (after subtasks 6-9, tampering candidacy + contamination; standard suite)** — the **complete** suite including the 3 slow tilt tests stands at **293/293** at this point → **293/293 (final, after subtask 10's real-footage re-validation and documentation — no code changed)**. A single pre-existing unrelated failure noted at the 163/164 milestone does not appear resolved or explained anywhere else in the record and should be checked.
- **End-to-end tests through real components, not mocks:** several of the later fixes (banner rendering, snapshot timing) were explicitly verified with an end-to-end test through the real `CameraWorker` + real `DecisionEngine`, not a mocked pipeline, to guard against fixes that pass in isolation but not in the real call path.

---

## 7. Glossary of Components (Quick Reference)

- **`low_light`** — HSV V-channel dark-pixel ratio vs. baseline.
- **`tampering`** — Canny + block-grid edge density + relative ambient-retention model (Approach A) + connected-component compactness vs. baseline.
- **`blur`** — Variance of Laplacian vs. baseline sharpness.
- **`tilt`** — DISK (Kornia) keypoints + mutual-NN matching + MAD outlier rejection + median displacement normalized by frame diagonal.
- **`decision_engine` (fusion)** — Relation-class (`always`/`conditional`/`independent`) + physical-predicate suppression across all 6 fault pairs (§2.5).
- **`decision_engine` (gating)** — Hard execution-skip of unreliable detectors with `"unmeasurable"` reporting transparency (§2.6).
- **`decision_engine` (banner/reporting)** — Single source of truth from `DecisionFrame`, 5 mutually-exclusive categories (`confirmed` > `pending` > `suppressed` > `too weak` > `unmeasurable`), frame-log schema v3 (§2.7).

Full per-component config values live in §2.8 (current configuration reference); design rationale and history are in §3 (detectors) and §4 (decision engine).
