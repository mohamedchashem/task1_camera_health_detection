# CAMERA HEALTH MONITORING — PRODUCTION RULES

## 1. PROJECT CONTEXT

This is a real approved camera health monitoring system intended to become a working production feature and be presented to a real client.

It is NOT a demo, learning project, proof of concept, or disposable prototype.

All decisions must be evaluated against:
- real camera footage
- realistic operating conditions
- reliability
- maintainability
- resource usage
- production integration
- realistic failure modes

---

## 2. SYSTEM SCOPE

The system contains four camera-fault detectors:

1. Low-light
2. Tampering / obstruction
3. Blur
4. Tilt

The detectors feed into a higher-level decision layer.

The system is also expected to include:
- SQLite event logging
- CI
- integration testing
- broader real-world dataset validation

Keep responsibilities separated between:

detectors → decision layer → event logging

Do not move responsibility between components without discussing the architectural reason.

---

## 3. REAL-WORLD COMPUTER VISION VALIDATION

Computer-vision features must be evaluated against realistic footage.

Consider:
- real camera footage
- lighting variation
- motion blur
- low texture
- compression artifacts
- occlusion
- camera movement
- viewpoint changes
- environmental variation
- false positives
- false negatives
- temporal instability
- threshold sensitivity

Prefer physically staged real faults over synthetic fault generation when validating fault detection.

Where appropriate, use blind evaluation:

The detector must not know when the fault occurs.

After detection, compare results against independently established ground truth.

Do not create tests that tell the detector the expected answer.

---

## 4. BASELINE

The system uses a camera-specific baseline reference representing normal operation.

Whenever baseline-dependent logic is changed, consider:

- how the baseline is captured
- whether it represents normal operation
- exposure differences
- environmental differences
- scene changes
- baseline drift
- absolute vs relative thresholds
- what happens if the baseline is poor

Do not assume one baseline automatically represents every legitimate operating condition.

---

## 5. LOW-LIGHT DETECTOR

Current approach:

HSV → V-channel → dark-pixel fraction → comparison with baseline → confidence.

The detector should consider:
- camera exposure
- automatic exposure
- scene-dependent brightness
- lighting variation
- threshold selection
- false positives
- false negatives
- temporal stability

Do not change thresholds merely to improve a small test sample.

If the approach needs significant redesign, research current production alternatives before implementation.

---

## 6. TAMPERING / OBSTRUCTION DETECTOR

Current approach:

Canny edges → grid blocks → edge density → structure loss → connected components → localized obstruction decision.

The intended physical signature is localized loss of visible structure caused by obstruction.

Known limitation:

Blur and low-light can also reduce edge density.

Do NOT overfit the tampering detector simply to hide this ambiguity.

When appropriate, evaluate whether cross-triggering should instead be resolved by the higher-level decision layer using information from multiple detectors.

---

## 7. BLUR DETECTOR

Current approach:

Variance of Laplacian → compare current sharpness against camera baseline → flag when sharpness falls significantly.

Before changing this approach, consider:

- motion blur
- lens obstruction/smudging
- scene texture
- compression
- camera focus behavior
- autofocus
- exposure
- threshold calibration
- temporal stability
- false positives
- false negatives

If a materially different algorithm is proposed, research current production approaches before implementation.

---

## 8. TILT DETECTOR

Current approach (DISK + MAD — no geometric model):

Baseline frame
→ DISK learned local features
→ current-frame features
→ mutual nearest-neighbor matching
→ MAD (median absolute deviation) outlier rejection on displacement magnitudes
→ median displacement of inlier matches
→ normalization by frame diagonal
→ tilt decision

**No geometric model (no affine, no homography, no RANSAC) is fit at any point.** Every geometric-model-based attempt tried before this (affine transform, homography, homography-filtered-by-RANSAC) proved numerically unstable on this project's sparse/clustered real-world matches — fitted condition numbers reached the millions even on genuine tilt frames. MAD works because it is a robust spread statistic built from medians and operates on the 1-D displacement-magnitude distribution, so it needs no minimum spatial distribution of keypoints to stay numerically well-behaved.

### DISK

DISK is a learned local feature detector/descriptor.

It identifies visually distinctive local points and produces descriptors that can be matched between images.

The implementation uses DISK through Kornia and may use GPU acceleration.

Do not assume GPU availability.

Provide a sensible fallback when appropriate.

### Feature matching

Feature matching identifies corresponding visual features between the baseline and current frame.

### Mutual nearest-neighbor matching

A correspondence is retained when the nearest-neighbor relationship is consistent in both directions.

This helps reduce unreliable matches.

### MAD outlier rejection

Outlier rejection uses median absolute deviation (MAD) on the matched-point displacement magnitudes (the "X84 rule"; threshold `TILT_MAD_REJECTION_THRESHOLD` = 3.0 MADs, near-zero-MAD epsilon floor `TILT_MAD_EPSILON`). Because MAD is built from medians, a handful of extreme outlier matches cannot drag it around the way they drag a mean or a fitted geometric model. RANSAC was tried for this role and rejected: RANSAC's internal outlier test fits the same already-proven-degenerate homography model, so it silently discarded real tilt matches along with the bad ones.

### Reliability guards (explicit failure behavior)

A frame's tilt estimate is only trusted when it passes two volume guards; otherwise tilt is skipped as "unreliable" rather than scored:

- **Match-volume guard** — at least `TILT_MIN_RELIABLE_MATCHES` (10) matched pairs AND the matches are at least `TILT_MIN_MATCH_RATIO` (0.05) of the smaller keypoint set.
- **Inlier-ratio guard** — at least `TILT_MIN_RELIABLE_MATCHES` inliers AND inliers are at least `TILT_MIN_INLIER_RATIO` (0.5) of the matches after MAD rejection.

Sparse or low-quality matches therefore never produce a large false tilt measurement.

### Median displacement and normalization

The tilt metric is the median displacement of the MAD-inlier matches, normalized by the frame diagonal so the measurement is less dependent on image resolution.

A frame is a tilt candidate when its median shift ratio reaches `TILT_MEDIAN_SHIFT_THRESHOLD_RATIO` (0.1). Confidence is a linear severity map that reaches 1.0 at `TILT_SHIFT_CONFIDENCE_CEILING_RATIO` (0.15) of the frame diagonal.

### Rejected geometric-model approaches (do not re-attempt)

The following were tried and rejected against this project's real footage; see Documentation.md §3.4 for the full trial-and-error record:

- ORB + affine/RANSAC — wildly inconsistent rotation estimates and outright crashes on low-texture frames.
- DISK + affine transform — can only represent in-plane roll, not pitch/yaw; fixed angle threshold misfired.
- DISK + homography + RANSAC + corner displacement — numerically ill-conditioned on sparse/clustered matches; false-positive corner shifts 30-43x threshold.
- DISK + homography + post-fit validation gates (SVD condition-number, warped-corner convexity, area-ratio) — also rejected genuine tilt frames, whose condition numbers were just as extreme.
- Essential Matrix decomposition (`cv2.findEssentialMat`) / PnP — require camera intrinsics / 3D-to-2D correspondences the project has no calibration step to obtain; never implemented.
- Vanishing-point / horizon-based absolute tilt — rejected as too fragile in low-structure scenes.

---

## 9. DECISION LAYER

Individual detectors should not necessarily make the final fault classification independently.

Cross-triggering is expected to exist.

Examples:

- low-light can reduce edge density
- blur can reduce edge density
- obstruction can remove visible structure
- tilt can change feature relationships

When modifying a detector, evaluate how its output interacts with the other detectors.

Do not solve cross-detector ambiguity by blindly adding complexity to an individual detector.

Research and compare approaches if the decision-layer architecture changes materially.

---

## 10. TESTING

Where appropriate, maintain:

- unit tests
- regression tests
- realistic footage validation
- blind fault detection
- independently confirmed ground truth
- false-positive evaluation
- false-negative evaluation
- edge-case testing
- failure-path testing
- integration testing

Passing unit tests does NOT mean a detector is production-ready.

Real-world behavior must also be evaluated.

---

## 11. THRESHOLDS

Do not arbitrarily choose thresholds.

For every important threshold, determine:

- why it exists
- whether it is absolute or relative
- how it was selected
- whether it requires calibration
- what happens near the boundary
- false-positive/false-negative tradeoff

If a threshold is empirical, state that clearly.

Do not pretend an arbitrary value is scientifically derived.

---

## 12. PRODUCTION READINESS

Before calling the system production-ready, evaluate:

- detector reliability
- cross-triggering
- decision-layer behavior
- event logging
- SQLite failure behavior
- configuration management
- CPU/GPU resource usage
- GPU fallback
- error handling
- logging/observability
- CI
- integration testing
- broader real-world dataset validation
- regression coverage
- deployment assumptions

If major validation remains incomplete, explicitly state:

"Not yet production-ready."

---

## 13. CHANGE WORKFLOW

For every new detector, algorithm change, threshold change, architectural change, database change, or integration change:

INSPECT
→ RESEARCH IF NEEDED
→ COMPARE OPTIONS
→ RECOMMEND
→ STOP
→ WAIT FOR USER DECISION
→ IMPLEMENT ONLY APPROVED CHANGE
→ TEST
→ REPORT
→ STOP

Never automatically implement the next stage.

Never silently redesign the architecture.

Never expand the approved scope without asking.