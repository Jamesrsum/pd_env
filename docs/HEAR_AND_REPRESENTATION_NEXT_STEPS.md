# HeAR and representation-learning development protocol

Status: all HeAR jobs completed; see [completed results](HEAR_RESULTS.md).
Final test remains sealed. Improvements
below are hypotheses, not claimed results. Code and aggregate summaries belong
in Git; recordings, embeddings, individual predictions and weights remain private.

## Evidence and priorities

The completed frozen development screen selected Whisper Large-v3 AUROC .8951,
Whisper Small .8878, AST .8867 and WavLM Large .8822. These are maxima selected
from many pipelines on validation, not unbiased performance estimates. Age-only
AUROC .9330 and age+device .9337 make incremental and common-support analyses
essential. A change in pooled accuracy alone will not establish better disease
signal. AUROC, AUPRC, log loss, Brier score and achieved specificity are primary
diagnostics; accuracy depends on prevalence and threshold.

Older AST full adaptation (.8695) did not beat its frozen head control (.8767).
Its learning-rate and checkpoint controls were inadequate for a strong conclusion.
The corrected AST experiment is running: warm-up 27612213, followed by branches
27612214_[0-2]. Synthetic and real GPU smoke tests passed in the previous stage.
Do not call early warm-up metrics final results or start another AST sweep yet.

Three research agents reviewed the HeAR protocol, contrastive design and
downstream evaluation. Their recommendations are incorporated here. Sequence:

1. Establish HeAR frozen/native frozen-head/last-two/full baselines.
2. Compare classifiers and simple fusion on exactly the same people.
3. Select one encoder/unfreezing policy, then compare BCE, diagnosis SupCon and
   acoustic auxiliary prediction with identical controls.
4. Consider cross-model alignment only if simple fusion demonstrates useful
   complementary information. Confirm narrow finalists across three seeds.
5. Lock the representation, classifier, calibration and threshold before the test.

The available-model CPU head panel can run while HeAR access is being resolved;
it does not replace the HeAR experiments.

## 1. HeAR protocol and access

Access is now approved. The official PyTorch distribution is `google/hear-pytorch`, pinned in
`configs/pretrained_models.lock.json` to
`f791cd42437c3e268c8ac84707e3508900f65f1a`. The 1.213 GB download is complete;
the weight SHA256 is `d44d355816ee4315f67d7810da274409e9b1a6570325fc5ba9ae27555fd81723`.
Earlier HTTP 403 failures were genuine gated-account restrictions. After the
user obtained access, the pinned downloader succeeded using configured local
credentials. No credential is saved in project files or submitted job commands.
The older `google/hear` home cache was documentation-only; this usable PyTorch
checkpoint is under `$MPOWER_MODEL_ROOT/checkpoints/hear_pytorch`.

Implementation: `src/models/hear.py`, `scripts/run_hear_protocol.py`,
`configs/hear_protocol.yaml`, and `scripts/submit_hear_protocol.py`.
Official Apache-licensed frontend code is preserved with its license and hashes
in `src/models/hear_vendor/`, pinned to upstream revision
`3cba3ba843350a9166227cf56e2a4a58399640d3`. This code license is distinct from
the model-weight terms.

Input and pooling:

- Reuse the immutable mono16k primary waveform, first at most 160000 real
  samples. Reject spans shorter than two seconds. No pitch shift/time stretch.
- Split into complete 32000-sample windows using the shared endpoint-anchored
  coverage rule. Ten seconds gives five nonoverlapping windows. Other lengths
  may require measured overlap; do not fabricate padded tails.
- Apply the official mel-PCEN frontend separately to a clone of each window.
  Upstream subtracts minima in place and uses global extrema: preprocessing a
  waveform batch together would introduce batch dependence. Concatenate the
  resulting `[1,192,128]` tensors for encoder inference.
- Use finite 512D `pooler_output`. Primary frozen representation is the mean
  across windows. Mean plus population SD (1024D) is a separate candidate.
- During fine-tuning, average window embeddings inside autograd, apply the same
  LayerNorm+linear diagnosis head, and calculate one record-level BCE loss.

Freeze/control design:

- Shared ten-epoch frozen-backbone head warm-up; branch from its best checkpoint.
- Three 20-epoch branches: frozen backbone, last two transformer blocks, full
  backbone. Keep head, initialization, visit order, head LR and evaluation equal.
- AdamW; starting head LR 1e-4, backbone LR 5e-6, separate cosine floors 1e-5
  and 1e-6. These are a starting protocol, not established best HeAR settings.
- Sample one visit per training participant per epoch. Evaluate original initial
  validation visits. Never replace a failed initial visit with a later recording.
- Freeze dropout/statistics in frozen blocks. Audit changed trainable parameters,
  unchanged frozen parameters, nonzero gradients and phase-labelled selection.
- Resume includes optimizer, Python/NumPy/CPU/CUDA RNG. Cache identity includes
  waveform/source hashes, pinned model, frontend/adapter code, versions and span.

The actual checkpoint loads completely under Torch 2.15.0+cu132 and Transformers
4.57.6: no missing, unexpected or mismatched keys. It is a 24-block ViT with
1024D hidden states and a pretrained linear 1024→512 pooler. The inspected block
path is `encoder.layer`, now recorded in config. Last-two tuning updates blocks
22/23 (25,192,448 parameters); full tuning updates 303,199,744 backbone parameters.
Set `MPOWER_HEAR_BLOCK_PATH=encoder.layer` for the submitter's explicit preflight.

With normal project environment variables set:

```bash
python scripts/download_pretrained_models.py --models hear_pytorch --workers 1
python scripts/run_hear_protocol.py --stage readiness
export MPOWER_HEAR_BLOCK_PATH=encoder.layer
python scripts/submit_hear_protocol.py
```

Submission verifies assets and the block sequence before allocating GPUs. A
small GPU smoke job tests pretrained batch/singleton embedding agreement, a
serialized checkpoint round trip, frozen extraction and all adaptation policies. Full
frozen extraction follows only after success, then shared warm-up and branches.
At most two GPU jobs run simultaneously. The cluster's five-day GPU partition
limit is retained; jobs checkpoint rather than assume unlimited wall time.
Frozen extraction also fits weighted LR and small MLP heads on mean/mean+SD.
Aggregate metrics and private per-person predictions are written outside Git.

Current snapshot: `$MPOWER_BENCHMARK_ROOT/hear_protocol/20261116T011601Z`.
Jobs: smoke 27612307; frozen extraction/heads 27612308; shared warm-up 27612309;
frozen branch 27612310; last-two 27612311; full 27612312; CPU downstream panel
including HeAR 27612313. Full stages depend on successful smoke/extraction;
the full adaptation branch follows the frozen branch to cap concurrency at two.
The downstream panel begins after frozen features and baseline heads complete.
See `docs/results/hear_execution.json` for verified execution status; submission
alone does not establish completed training or scientific performance.

The first smoke (27612286) correctly failed batch/singleton invariance: maximum
embedding difference .00190115. Its six dependent jobs were cancelled before
execution. Explicitly disabling TF32 for both CUDA matmuls and cuDNN convolutions
resolved the failure without weakening the tolerance. The replacement smoke's
pretrained `[2,512]` embedding and serialized checkpoint round-trip checks pass
on an A100 80GB. All HeAR stages use this full-float32 policy; no AMP is enabled.

The numerical retry (27612294) completed all three training policies and their
parameter-change checks, but review found that its frozen-adaptation completion
file overwrote frozen-extraction provenance. Its pending full jobs were cancelled.
Extraction now stays in `runs/frozen/`, the shared checkpoint in `runs/warmup/`,
and adaptation policies in `runs/adaptation/{frozen,last2,full}/`. A final smoke
audit requires preserved feature provenance, identical branch initialization,
head updates in every arm, backbone updates only in adapted arms, and unchanged
frozen parameters. Full stages cannot start until this final audit succeeds.

Final smoke 27612307 completed successfully in 4m15s, including the artifact and
all parameter-change audits, on an A100-SXM4 40GB. Full frozen job 27612308 is
running; warm-up, adaptation and downstream jobs are dependency-pending. The
21 HeAR/research-protocol tests and five controlled-adaptation tests passed.
No full-cohort HeAR accuracy or fine-tuning outcome is available at this point.

Official references: [PyTorch example](https://github.com/Google-Health/hear/blob/master/notebooks/quick_start_with_hugging_face_pytorch.ipynb),
[frontend source](https://github.com/Google-Health/hear/blob/3cba3ba843350a9166227cf56e2a4a58399640d3/python/data_processing/audio_utils.py),
[model access](https://huggingface.co/google/hear-pytorch),
[model card](https://developers.google.com/health-ai-developer-foundations/hear/model-card).

## 2. Downstream experiments implemented and submitted

`scripts/run_downstream_panel.py` uses canonical initial recordings, one per
person, with participant/duplicate-component-disjoint three-fold training CV.
It refuses test rows and incompatible source-manifest signatures. Features are
loaded using the original source/derived waveform/model-signature cache keys;
missing caches abort, rather than silently changing the comparison cohort.

This panel also trains on one canonical initial recording per participant.
Earlier probes used repeated training visits with equal-person weights, so
their selected maxima are context rather than directly matched head controls.
Compare classifier choices within this panel. A later repeat-visit comparison
can fit PCA on initial training visits, then fit weighted heads on all visits.

Predeclared cached features: matched-two-second Whisper Small block 6 mean/SD
and WavLM Large block 12 mean/SD, each averaged across windows. Experiments:

| Experiment | Fitted comparison |
|---|---|
| Each encoder | LR, PCA+LR, LinearSVC |
| Early fusion | Concatenate source blocks, then fold-fitted scaling/PCA/head |
| Late fusion | Equal mean of predeclared LR C=.1 probabilities |
| OOF stacking | Fit LR C=.1 on training-only OOF source probabilities |
| Age+device | Fixed cubic age basis and training-vocabulary device indicators |
| Age+device+audio | Same cohort and CV, adding Whisper or both audio sources |

LR C grid [.01,.1,1,10]; PCA sizes 32/128; LinearSVC C [.01,.1,1]. Imputation,
scaling, PCA and heads are fitted inside each training fold. PCA uses one
recording/person, preserving equal-person contributions. SVM probabilities use
a sigmoid fitted on grouped training OOF decision scores. No validation rows fit
calibration or stacking. Device vocabulary is defined on training only; it is not
a numerical distribution-fitted transform. All head artifacts remain private.

Report AUC/AP/calibration loss and a paired duplicate-component bootstrap of
the incremental audio AUROC. Bootstrap intervals remain exploratory after
extensive validation selection. Add pre-specified age/sex/device strata and
age-overlap sensitivity analyses before choosing a scientific finalist; these
are not implemented by the initial panel.

CPU smoke 27612238 completed successfully (18 seconds); full job 27612239 was
submitted with an after-success dependency. Smoke metrics from its tiny cohort
are implementation checks, not scientific estimates. Snapshot/output root:
`$MPOWER_BENCHMARK_ROOT/downstream_panel/20261116T004453Z`.

Set `MPOWER_HEAR_ROOT` to a completed HeAR protocol root to include HeAR's 512D
frozen mean in a subsequent panel, with matching manifest/signature validation.
The already submitted snapshot compares the two available encoders only.
Add broad acoustic features after same-span preparation/coverage auditing;
older features from a different duration are not interchangeable.

The confirmed specificity bug is fixed in `development.operating_point`: both
threshold selection and probability comparison use float64, preserving the
`nextafter` exclusion of tied controls. Achieved specificity and control count
are saved. Historical outputs are not relabelled. Thresholds chosen on validation
are labelled exploratory; the final-test operating point must be locked separately.

References: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html),
[SVC complexity](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html),
[calibration](https://scikit-learn.org/stable/modules/calibration.html).

## 3. Exact learning objectives and controls

`src/training/research_objectives.py` implements differentiable BCE/SupCon/
grouped masked-Huber composition, the diagnosis projector, participant-balanced
visit batches and observed-only initial-training acoustic target scaling. These
components have synthetic gradient/masking tests. They are not yet wired into
a production encoder training job; no contrastive gain or real-data execution
is claimed. Integrate only after selecting a baseline encoder/policy, then test
the real GPU batch's memory and parameter-change audits before a full run.

| Arm | Loss | Learned signal |
|---|---|---|
| A | BCE | Diagnosis reference |
| B | BCE + .1 SupCon | Same-diagnosis geometry across distinct people |
| C | BCE + .1 grouped masked Huber | Predict measured phonatory/acoustic attributes |

All arms use the same best BCE warm-up checkpoint, unfreezing policy, participant
order, diagnosis head, LR schedules and compute budget. Instantiate the same
auxiliary modules for identical initial state; inactive heads receive no loss.
Run one seed for debugging/screening, then three seeds for finalists.

**SupCon definition.** Each anchor's positives are different participants with
the same diagnosis; negatives have the opposite diagnosis. Exclude self, same
participant and same duplicate component from the denominator. Different visits
do not create independent positives. Use actual batches of 32 distinct people,
approximately 16 PD/16 controls, with at least two eligible people per class.
Gradient accumulation does not enlarge the contrastive denominator. Projection:
D→128→GELU→32; L2 normalization; temperature .07. BCE stays on the encoder
representation; discard the projection head for standard downstream extraction.
Balanced batches use importance-weighted BCE to recover the participant-level
training diagnosis prior. Invalid batches fail explicitly. Audit class/duplicate
composition and majority-class coverage across epochs.

Diagnosis SupCon can also reinforce age/device confounding, because these
covariates correlate with diagnosis. Probe learned embeddings for age/device,
report same-age/common-support performance and incremental value, and consider
train-only nuisance-balanced sampling only with sufficient support in both
diagnostic groups. Do not claim that diagnosis contrast removes confounding.

A [2025 PD study](https://www.mdpi.com/2306-5354/12/7/728) tested Wav2Vec2/HuBERT
with SupCon on NeuroVoz. It motivates this arm but does not establish a gain for
HeAR or mPower. Underlying objective: [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362).

**Acoustic auxiliary definition.** This is regression, not contrastive learning.
Recompute a pre-registered representative panel on exactly the encoder's first
≤10-second source span. Proposed groups:

1. Pitch/voicing: F0 mean/variability, pitch-valid fraction.
2. Perturbation/noise: representative jitter, shimmer and HNR.
3. Spectral dynamics: centroid, bandwidth, rolloff, flatness, flux mean/SD.
4. Timbre: MFCC1–13 mean/SD and selected nonredundant eGeMAPS descriptors.

Current extraction supports Praat, 88 eGeMAPSv02 functionals and spectral/MFCC
features; the older alignment preparer uses only ten Praat targets. CPP is not
implemented. Version the new schema and same-span extractor before running it.
Ignore undefined targets; never train against an imputed target. Fit scaling,
coverage filtering and any correlation reduction only on initial training visits.
Average observed errors within each group, then across groups, so MFCC count
does not dominate. Save per-target coverage, MAE and R², alongside diagnosis
metrics. Compare acoustic-only, audio-only, concatenation and late fusion before
interpreting auxiliary supervision.

## 4. Conditional cross-model alignment and prototype repairs

Pair the exact same recording and sample interval across HeAR/Whisper/WavLM.
The positive teaches agreement between encoders, not disease truth. Start with
frozen projectors and compare to no alignment, untrained projections, early
concatenation and OOF late fusion. Only move gradients into a student encoder
after the frozen pilot beats these controls across seeds.

Negative-free [VICReg](https://arxiv.org/abs/2105.04906) is a possible pilot with
32D projections, ≥64 distinct people/batch and explicit variance/effective-rank
collapse checks. Symmetric paired [InfoNCE/CLIP](https://proceedings.mlr.press/v139/radford21a.html)
is an alternative with separate row/column identity masking. Neither is proven
to improve this dataset. Do not run both as an unrestricted model sweep.

Keep the existing narrow alignment prototype held until these issues are fixed:

- Replace recording-ID-only pair caches with source/waveform/span/model/schema/
  extractor/config hashes and verify signatures on reuse.
- Preserve the canonical initial-visit flag after filtering; audit exclusions.
- Clone one identical projector initialization into true/shuffled/untrained arms.
- Shuffle with participant/component-excluding pairing; use the shuffled column
  identities when masking negatives, and balance sampling by participant.
- Restore RNG on resume and reject mismatched configuration/provenance.
- Refit all transforms inside training CV when claiming full-pipeline CV.

These repairs are identified work, not completed repairs to the old prototype.
The new participant-aware objectives and downstream panel do not use it.

## Decision rules and remaining work

Wait for corrected AST branches, the CPU panel and HeAR smoke/full baselines.
Compare matched frozen and adapted heads before increasing complexity. Broaden
the downstream panel to HeAR and same-span acoustic features, then implement
the selected three-arm encoder objective runner with real GPU resource checks.
Only shortlist improvements with consistent paired-person evidence, sensible
calibration and age/device incremental value. Report negative results as such.
The sealed test is not part of debugging, objective learning or selection.
