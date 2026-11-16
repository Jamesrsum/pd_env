# Research-agent audit and controlled AST rerun

**Completion update:** all corrected branches finished. Frozen/last2/full
AUROCs were.8715/.8671/.8647; no matched fine-tuning improvement was observed.
See the [completed methods/results notebook](../notebooks/02_All_Models_Methods_and_Results.ipynb).
The submission/startup notes below are historical.

Three independent research agents reviewed the executed training code,
checkpoints, primary paper and authors' implementation. The previous runs are
preserved under private adaptation_20261114; no prior result was overwritten.

## What the paper establishes

Tougui et al. report subject-wise mean-prediction AST AUROC 0.5471 from scratch
versus 0.9589 using AudioSet pretraining plus fine-tuning. They do not report a
final frozen-pretrained AST control. That supports transfer learning, but is
not evidence that unfreezing must improve our frozen-pretrained baseline.
Their cohort, repeated-record aggregation and five-seed protocol differ from
our initial-assessment development endpoint.
[Paper, Tables 3–4](https://pmc.ncbi.nlm.nih.gov/articles/PMC11614913/).

The authors' notebook uses native AST filterbanks, a LayerNorm+Linear classifier
and restoration of a best warm-up checkpoint. Its AST path does not resize the
spectrogram to 224 pixels. The paper/table and notebook contain discrepancies;
this is an audited follow-up, not an exact reproduction.
[Authors' notebook](https://github.com/Jetliqs/PDVAT/blob/main/mmpd-voicedl.ipynb).

## Actual old-run findings

AST really was fine-tuned. Direct tensor comparisons with the pinned checkpoint
found that last-two-block training changed blocks 10/11 while preserving lower
layers; full training changed patch projection, earlier transformer blocks and
final norm. The frozen control's sampled backbone tensors were unchanged.

All AST arms had identical first-ten-epoch trajectories. Their best warm-up
AUROC was 0.86852. Full adaptation improved to 0.86952, but continued head-only
training achieved 0.87665. The comparison confounded unfreezing with a head LR
reduction from 1e-3 to 1e-5 in the adapted arms. Adaptation also started from
the last rather than best warm-up. The previous global best checkpoint could
come from warm-up while DONE named only the requested fine-tuning policy.

The broader 'all worse' conclusion is also incorrect. Relative to their own
warm-up, last-four-block training improved wav2vec2 Base 0.6560→0.7353,
WavLM Base+ 0.6615→0.8313 and WavLM Large 0.6799→0.8205. Their earlier cached
frozen probes used different representations, scaling and classifiers and
are not matched before/after controls. ResNet full improved over its direct
frozen control. Robust last-two-block adaptation deteriorated and needs a
controlled baseline before architecture-level conclusions.

## Framework corrections

The new controlled runner uses native LayerNorm+Linear heads consistently for
all paired arms, a single best warm-up checkpoint, identical branch head LRs,
identical participant/visit schedules and separate backbone LRs. Optimizers
reset identically at branching; no branch inherits an unfair momentum state.
Warm-up, best branch and selected policy are labelled explicitly. Frozen and
trainable state fingerprints, nonzero backbone gradients and actual LRs are
recorded for production runs. Test rows are rejected.

Full optimizer, phase/config identity and Python/NumPy/CPU/CUDA RNG state are
saved. Mid-adaptation interruption/resume reproduces uninterrupted CPU training
exactly in the deterministic synthetic test. CUDA RNG state is supported;
that does not independently establish cross-hardware bitwise reproducibility.

Native AST microbatches of four accumulate to 32 distinct participants per
optimizer step. Batch/singleton logits and gradients agree in the synthetic
zero-dropout AST check. AST uses LayerNorm and this checkpoint has zero
hidden/attention dropout, so prior singleton accumulation was principally a
throughput issue, not by itself a learning bug. Waveform models retain
singleton forwards until equivalent batching tests pass.

Default research schedule: SGD/Nesterov, 10 common warm-up epochs at head LR
1e-3; then 20 paired epochs with identical head cosine schedule 1e-3→1e-5 and
adapted-backbone schedule 1e-5→1e-6. This tests decoupled rates; it is not a claim
of an optimal clinical recipe. A configurable Adam alternative supports the
official AST betas (0.95,0.999); changing optimizer/weight decay requires a fresh
matched frozen control. The current run is one seed, not a definitive multiseed
comparison.
[Official AST training](https://github.com/YuanGongND/ast/blob/master/src/traintest.py).

Implementation: src/training/controlled_adaptation.py,
scripts/run_controlled_adaptation.py, configs/ast_controlled_finetuning.yaml.
Four unit tests passed: paired initialization/state changes, actual mid-run
resume, sealed-test refusal and native minibatch/singleton gradient agreement.

## Submitted jobs

New immutable code root: controlled_ast/20261116T002439Z under
MPOWER_BENCHMARK_ROOT. A bounded GPU smoke validation gates the common full
warm-up, which gates three branches (frozen, last2, full), at most two GPUs
concurrently. The original dataset snapshot and held-out test remain unchanged.

- Smoke: 27612212.
- Full common warm-up: 27612213.
- Paired branch array: 27612214, tasks 0–2.

Submission is not completion and no improved AUROC is yet claimed. Use the
separate warm-up and branch metrics instead of labelling an earlier frozen
checkpoint as a fine-tuned result. Age-only AUROC remains a necessary confounding
control; a higher pooled voice AUROC alone would not establish added value.

GPU smoke validation completed successfully: frozen, last2 and full branches
started with identical backbone/head fingerprints. Requested adapted backbones
changed, the frozen backbone stayed unchanged, all heads trained, and frozen
parameters remained unchanged. These smoke scores are not scientific results.
The common full-data warm-up has started; the three paired branches depend on
its successful completion.
