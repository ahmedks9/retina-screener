# Problem Statement — AI-Assisted Retinal Disease Screening

## Input
A single fundus (retinal) image of one eye (left or right), typically
sourced from the ODIR-5K dataset, RGB color, variable resolution and
quality (subject to the artifact taxonomy documented in Phase 1.2:
illumination, focus, media opacity, field-of-view, and
acquisition/compression variation).

## Output
An 8-dimensional vector of independent probabilities, one per disease
category, each between 0 and 1:

    [N, D, G, C, A, H, M, O]

Where:
    N = Normal                          H = Hypertensive Retinopathy
    D = Diabetic Retinopathy            M = Myopia (pathological)
    G = Glaucoma                        O = Other
    C = Cataract
    A = Age-related Macular Degeneration (AMD)

More than one position can be active simultaneously (e.g., a patient
can be both D=1 and C=1), reflecting real co-occurring conditions.

## Task Type
Multi-label image classification (sigmoid output per class, not
softmax) — chosen because ODIR-5K's patient records show real
co-occurring diseases (Step 1.3.2), and collapsing this to multi-class
would discard clinically meaningful information.

## Unit of Prediction
Per-eye (single image in -> single label vector out), not per-patient
pair. Chosen for a simpler, more debuggable baseline pipeline and
~2x more training samples (7,000 vs. 3,500) (Step 1.4.1). Per-eye
labels will be derived from the free-text diagnostic keywords column
during Chapter 2, since ODIR-5K's official one-hot vector is
patient-level (combining both eyes).

## What Success Looks Like
- Primary metrics: per-class AUC-ROC and F1-score (macro-averaged
  across the 8 classes), not raw accuracy -- because accuracy is
  misleading under class imbalance and doesn't reflect screening
  priorities (Step 1.1.2, Step 1.3.3).
- We prioritize sensitivity/recall over precision where the two
  trade off, since in a screening context a missed disease (false
  negative) is clinically worse than a false alarm (false positive).
- Reference benchmarks from published work (Step 1.3.3): AUC in the
  90-96%+ range and F1 in the 85-94% range are realistic strong
  results for full 8-class multi-label ODIR-5K classification. Scores
  above 98% typically come from simplified 5-class setups and are not
  a fair comparison point for our full 8-class task.
- Secondary deliverable: Grad-CAM visualizations (Chapter 7) showing
  the model attends to clinically plausible regions (e.g., optic disc
  for Glaucoma, macula for AMD), supporting interpretability and trust.

## Known Constraints
- Hardware: single GTX 1650, 4GB VRAM -- influences batch size,
  possibly requiring gradient accumulation (Chapter 5).
- Dataset origin: multi-hospital, multi-camera -- requires
  normalization to avoid shortcut learning tied to acquisition
  equipment rather than disease signal (Step 1.2.2).
