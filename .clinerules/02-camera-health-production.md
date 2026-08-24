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

Current approach:

Baseline frame
→ DISK learned local features
→ current-frame features
→ mutual nearest-neighbor matching
→ RANSAC
→ homography
→ corner displacement
→ normalization by frame diagonal
→ tilt decision

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

### RANSAC

RANSAC (Random Sample Consensus) estimates a geometric model while rejecting outlier matches.

Do not assume every feature match is correct.

### Homography

A homography is a projective transformation represented by a matrix that maps points from one image plane to another under an appropriate planar/projective relationship.

Do not assume that every estimated homography is physically meaningful merely because a mathematical solution exists.

### Corner displacement

Apply the estimated homography to the four frame corners.

Measure how far those corners move.

### Normalization

Normalize corner displacement using the frame diagonal so the measurement is less dependent on image resolution.

The resulting metric is used to detect significant camera movement affecting roll, pitch, or yaw.

### Post-fit validation

Validate the estimated homography before trusting the displacement.

Current validation work includes:
- SVD condition number
- warped-corner convexity
- rejection of degenerate/unstable estimates

When matches are sparse or low-quality, the system must have explicit failure behavior.

Never allow numerically unstable geometry to silently produce a large false tilt measurement.

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