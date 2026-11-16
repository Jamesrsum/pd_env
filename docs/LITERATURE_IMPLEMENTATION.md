# Source-audited literature comparisons

The requested single literature reviewer examined 13 primary full PDFs and
extracted full text privately. Bibliography, PDF hashes/page counts, methodology
cards, source-code references, missing details and adjacent-field evidence are
in [the review](literature/REVIEW.md). The durable single-agent workflow is
[documented here](literature/AGENT_PROTOCOL.md). Full papers/text and participant
data stay outside Git. `scripts/archive_literature.py --verify-only` independently
verified every archived PDF checksum.

## What is implemented

| Family | Methods | Source and interpretation |
|---|---|---|
| Strong HeAR initialization | LP-head-only, LP→full tuning, LP→LoRA | LP-FT, ICLR2022; transferred to HeAR/mPower |
| Direct PD contrastive mechanism | HuBERT-base CE, detached-classifier CE + encoder SupCon | Bioengineering2025 NeuroVoz paper; explicitly adapted to sustained vowels |
| Adjacent small-data models | TabICL2025, TabICLv2 2026, matched logistic regression | Local tabular foundation inference on cached audio embeddings |

These are implementations of published mechanisms and controlled adaptations
to our cohort. Different endpoints/splits and underspecified paper settings
prevent a claim of exact reproduction of another paper's numerical SOTA.

**Prior fine-tuning was not LoRA.** Earlier AST/HeAR last-block runs updated the
selected original weights; full runs updated their original backbones. The new
LoRA branch freezes every original HeAR weight and trains only Q/V rank8,
alpha16 low-rank adapters plus the classifier. It updates786,432 backbone adapter
parameters, approximately0.26% of303,199,744 original parameters. The standard
`(alpha/r) B A` update starts with zero B/random A, so initial predictions match.

## HeAR LP-FT/LoRA controls

All three branches start from the completed, converged standardized frozen
logistic probe, not the much weaker earlier LayerNorm/linear warm-up. The fitted
training mean/scale remain fixed buffers in the differentiable head. Initial
real-waveform predictions are checked against the original sklearn pipeline
before any training; they agree within the stated float32 tolerance.

Each arm uses the same initialization, one visit/person/epoch, canonical initial
validation endpoint, AdamW, head LR1e-4 and20epochs. Full backbone LR5e-6;
LoRA LR1e-4; frozen backbone has no gradient. Rank/optimizer settings transferred
to HeAR are declared hypotheses, not proven PD-optimal parameters. Encoder
gradient checkpointing controls memory. Float32 is used without TF32 or AMP.

The converged LP is an explicit **epoch-zero candidate**. If adaptation never
beats it, reports say `selected_stage=lp_initialization`; they cannot silently
label an unchanged initial checkpoint as a fine-tuning improvement. Actual
parameter-update audits are reported independently of checkpoint selection.

## HuBERT CE versus the paper's SupCon mechanism

The separately pinned `facebook/hubert-base-ls960` checkpoint is768D/12blocks,
not the existing1024D HuBERT Large. All native tensors load completely before
latent augmentation is disabled. A loader regression initially excluded the
unused masking tensor when disabling augmentation before construction; this was
fixed by loading the native architecture first, then setting probabilities to0.
No unexpected encoder key is silently ignored.

Transformers also deep-copies the supplied configuration during loading. The
disabled masking/layerdrop settings are therefore applied to the loaded
`backbone.config` and asserted at runtime, not to the input config copy. The
first production speech attempt was cancelled and preserved after this check;
the corrected family uses a new isolated directory and does not resume it.

Both arms use the same mean sequence pooling and768→64/ReLU/dropout.2→2 head,
AdamW2e-5, betas(.9,.999), weight-decay.01,100epochs and actual32-person batches.
CNN weights are frozen in both arms as a declared project choice. Gradient
checkpointing keeps each actual batch in the contrastive graph; microbatch
accumulation is not substituted for the contrastive denominator.

The contrastive arm stops the classification gradient at the encoder. CE trains
the head on detached features; SupCon trains the encoder and32D projector.
Both loss terms use coefficient1 and SupCon averages anchors; the paper does
not report a loss coefficient, so this is an explicit scaling convention.
Positives share diagnosis across different people; self/person/duplicate pairs
are masked. Temperature.07 follows the paper. The CE-only control uses the same
class-balanced visit batches and training-prevalence importance weights.

Material deviations/gaps are recorded in `configs/literature_benchmarks.yaml`:
mPower vowels instead of Spanish repeated sentences; up to5 real seconds instead
of the paper's padding/cropping/filtering; frozen CNN; a declared single-linear32
projector because depth is missing; no unspecified low-pass filter; and no
invented hard-negative mining formula. The paper's best reported hard-negative
scaling cannot be reproduced exactly from its available description. This is
therefore labelled **unscaled detached-SupCon adaptation**, not exact SOTA.

## Local TabICL and label-efficiency comparisons

Both checkpoints are ungated, BSD-3-Clause and hash-verified under the private
model root. The2025 v1 uses32 estimators; the2026 v2 uses8. Library2.2.0 and
einops.8.1 are fixed; original checkpoint names/revision are explicit so updated
defaults cannot silently swap the method. Inference sets a local model path,
`allow_auto_download=False`, no external inference API, float32 and no FA3.

One canonical initial training visit/person is used. StandardScaler/PCA fit
only the labelled training context; LR regularization uses training-only grouped
CV with transform refitting inside folds. Foundation-model settings are fixed
before validation, without fitting a validation-label context. PCA dimension is
at most128 and decreases by a predeclared training-size rule for small budgets.

Full-label comparisons use HeAR, Whisper Small and WavLM Large cached features.
HeAR additionally uses50/100/500-label subsets, with a fixed, stratified training
sampling seed and the same validation people. This is a one-seed label-budget
simulation, not a claim that pretraining used only those few labelled people.
There are18 fixed encoder/budget/classifier comparisons. No foundation-model
pretraining is performed on participant data.

## Verification and Slurm submission

Thirty-seven targeted checks passed across the11 new method checks and26
existing HeAR/controlled-adaptation checks. They cover LoRA initial identity,
adapter updates without base updates, checkpoint reload, exact standardized-LR
transfer, detached gradient routing, true contrastive batches, complete RNG/
optimizer resume, epoch-zero selection, split guards, local-only TabICL context,
native masking-parameter loading, and scheduler dependencies.

Final immutable snapshot:
`$MPOWER_BENCHMARK_ROOT/literature_methods/20261116T091246Z`.

| Stage | Job | Verified status at submission audit |
|---|---|---|
| LP-FT/LoRA real GPU smoke |27613592|Completed, all three arms audited|
| Corrected HuBERT CE/SupCon real GPU smoke |27614603|Completed, actual config and32-person batches verified|
| Both local TabICL smoke checks |27613594|Completed|
| Full TabICL/label-budget panel |27613595|Completed,18 comparisons|
| Corrected HuBERT CE/SupCon production array |27614604_[0-1]|First arm running; second queued|
| HeAR LP frozen/full/LoRA array |27613597_[0-2]|First arm running; full/LoRA queued|

At most two suite GPU jobs run simultaneously. Production arrays run one task
per family at a time. Smoke gates protect full jobs; a failed tabular production
job does not permanently block independent HeAR arms. An earlier HuBERT smoke
failed the strict loading check and its production dependencies were cancelled;
no failed run is presented as a completed method.

Corrected speech outputs are under
`speech_retry_20261116T092846Z/` inside the suite root above. HeAR and TabICL
keep their original valid directories. The superseded speech array27613596
was cancelled; its partial metrics are not part of the planned comparison.

The final test remains sealed. Smoke metrics are execution checks on tiny
cohorts and are not scientific results. Full runs will write `DONE.json`, phase-
labelled validation metrics/predictions, history, configuration and parameter
audits under private `runs/` and `tabular_runs/`. No accuracy improvement is
claimed until those full comparisons finish and are analyzed.

## First completed adjacent-field results

The18-comparison TabICL panel completed in1m57s. All rows below use the same
729 canonical initial validation participants and3,397 initial training visits,
with train-fitted PCA128 and the same features within each encoder comparison.

| Frozen encoder | PCA + LR | TabICL2025 | TabICLv2 2026 |
|---|---:|---:|---:|
| HeAR |.8522|.8577|.8599|
| Whisper Small |.8684|.8572|.8662|
| WavLM Large |.8718|.8743|**.8811**|

These are AUROC values. WavLM+TabICLv2 improves over its matched LR pipeline
by+.00934, with exploratory paired component-bootstrap95% interval
[-.00340,.02315]. This is promising, but the interval includes zero and the
models share a repeatedly used validation cohort. It is not proof of a new
independent-test SOTA result. The raw HeAR mean+LR pipeline from the prior run
remains.8700; the table above additionally applies PCA and initial-visit-only
training, so it is a different pipeline.

At50 HeAR labels, LR/TabICLv1/TabICLv2 AUROCs are.6049/.6439/.6542;
at100 labels they are.6710/.6872/.6323; at500 they are.7916/.7881/.7986.
No classifier dominates all budgets. These are single, non-nested stratified
subsets, and PCA dimension changes with budget (16/33/128/128). Compare methods
within a budget; do not interpret the curve as a pure causal effect of label
count or a stable few-shot guarantee.

The [aggregate table](results/literature_tabular/metrics.csv) and
[paired intervals](results/literature_tabular/summary.json) contain all results.
`scripts/analyze_tabular_literature.py` verifies identical prediction identities
and recomputes AUROCs before exporting only aggregates. HeAR LP-FT/LoRA and
HuBERT CE/SupCon training outcomes remain pending.

## Re-running

Install `requirements-literature.txt`, configure the existing private benchmark,
model and development-snapshot environment variables, and set:

```bash
export MPOWER_HEAR_SOURCE=/private/path/to/completed/hear_protocol/run
export MPOWER_FROZEN_ROOT=/private/path/to/completed/protocol_experiment
python scripts/submit_literature_benchmarks.py
```

The locked HuBERT and TabICL model inventories must already be staged under
`MPOWER_MODEL_ROOT/checkpoints`. Public model weights, trained derivatives,
participant embeddings and predictions must not be committed. The source
review explains why authors' mPower-trained AST weights and unavailable
VoiceFM weights are not substituted for independent, auditable controls.

Download the existing verified locks with:

```bash
python scripts/download_pretrained_models.py --inventory configs/literature_models.json --lock configs/literature_models.lock.json --public-only --output-root "$MPOWER_MODEL_ROOT"
python scripts/download_pretrained_models.py --inventory configs/tabicl_models.json --lock configs/tabicl_models.lock.json --public-only --output-root "$MPOWER_MODEL_ROOT"
```
