# OptiHealth EiV contrastive-loss case study

Use this reference as empirical prior, not as a universal benchmark.

## Protocol

- RTX 5070, bf16, batch 102
- Subject-disjoint raw-tensor split: 4,962 train / 1,119 validation / 944 test
- Scientific pilot: seed 17, 500 steps, 51,000 samples
- Test stayed locked because no pilot passed
- Collapse gate: median native cosine >0.90 or native effective rank <5
- Population normalization disabled
- Present HR zeros were corrected to missing in a frozen experiment-only mask

Primary artifacts:

- `/root/optiHealth-EiV/outputs/experiments/contrastive_losses_20260727/FINAL_REPORT.md`
- `/root/optiHealth-EiV/outputs/experiments/contrastive_losses_20260727/aggregate/pilot_summary.json`
- `/root/optiHealth-EiV/outputs/experiments/native_rbf_kta_vicreg_20260727/aggregate/pilot_summary.json`

## Results

| Objective | Configuration | HR R² | Recall@5 | Effective rank | Median cosine |
|---|---|---:|---:|---:|---:|
| Untrained | seed-matched | 0.96008 | 0.01746 | 2.662 | 0.9975 |
| DCL package | fixed | 0.95260 | 0.10723 | 4.300 | 0.1234 |
| DCL + RBF-HSIC | weight .1, BW .5 | 0.96306 | 0.11222 | 4.544 | 0.1852 |
| projected SoftCCA | decorrelation .5 | 0.96298 | 0.01247 | 2.249 | 0.9998 |
| projected VICReg | covariance .5 | 0.95540 | 0.00499 | 2.356 | 0.9887 |
| native RBF-KTA + VICReg | regularization .3 | 0.94465 | 0.11970 | 4.910 | 0.1336 |

No row passed the rank gate.

## What the results established

1. The HR Ridge probe was saturated by random high-dimensional features.
   Trained-vs-untrained R² differences were too small for confident model
   selection.
2. DCL learned useful cross-modal geometry despite low spectral rank:
   Recall@5 rose 0.01746→0.10723 and median retrieval rank improved 201→39.
3. The best HSIC pilot improved DCL's R², MAE, Recall@5, and rank, but held-out
   HSIC decreased 0.00657→0.00552. It was the best of nine one-seed candidates,
   so the mechanism and reproducibility remain unconfirmed.
4. SoftCCA and VICReg regularized learned 128×2,560 loss heads while native
   embeddings collapsed. Loss-head success did not imply usable geometry.
5. Direct native KTA plus variance/covariance fixed directional collapse and
   achieved the best Recall@5, alignment gap, and rank, but rank 4.91 remained
   below both the original 5 and committed follow-up 6 thresholds.
6. Increasing native regularization from .3 to 1 and 3 reduced rank and HR R².
   More anti-collapse weight was not monotonically better.
7. Adaptive-bandwidth RBF-HSIC was high for collapsed untrained/projected
   representations and low for useful DCL geometry. It measured shared
   low-dimensional structure, not representation utility.

## Data-path findings worth carrying forward

- Before fixing the sampler, about 99% of audited stratified batches repeated a
  raw row. Sampling without replacement within adequately sized pools removed
  this confound.
- After the fix, 25/48 audited batches still contained caption-equivalent
  positives, with mean 0.667 extra equivalent captions and maximum 3. These
  are false negatives for one-positive DCL.
- Cache precomputation must not consume augmentation RNG. Reset model RNG after
  cache population so cold and warm cache runs initialize identically.
- Exact HR zero correction was sufficient in the campaign domains; no present
  nonzero HR fell outside 30–220 bpm.
- Merge-time population normalization was unnecessary for HR and would have
  fitted auxiliary-channel statistics using held-out subjects.

## Reusable implementation

- Losses: `/root/optiHealth-EiV/src/training/losses.py`
- Phase-A integration: `/root/optiHealth-EiV/src/training/pipeline.py`
- Deterministic data path: `/root/optiHealth-EiV/src/training/datasets.py`
- Campaign harness: `/root/optiHealth-EiV/src/experiments/contrastive_loss_campaign.py`
- CLI: `/root/optiHealth-EiV/scripts/run_contrastive_loss_campaign.py`

Reusable components include unbiased RBF-HSIC, projected SoftCCA/VICReg,
duplicate-aware native RBF-KTA, exact batch-Gram covariance, deterministic
subject splits, advancing private augmentation RNG, strict stratified batches,
duplicate-aware retrieval, fixed evaluation indices, and artifact hashing.

## Best next hypothesis

Test duplicate-aware multi-positive DCL before adding another objective family.
The intervention directly removes a measured false-negative mechanism.

Suggested order:

1. incumbent DCL;
2. multi-positive DCL using exact caption-equivalence masks;
3. only if anisotropy persists, add one weak batch-Gram spectral
   rank/entropy regularizer.

Use paired validation seeds and retrieval/effective rank as primary evidence.
Keep HR R² secondary with shuffled-label and low-capacity-probe controls.
