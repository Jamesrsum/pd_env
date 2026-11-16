# Full-PDF methods audit: PD and adjacent small-data learning

Reviewed 2026-11-16 by the single literature-review agent requested for this
project. Primary PDFs and full `pdftotext -layout` extractions are retained under
`$MPOWER_BENCHMARK_ROOT/literature_review/`, outside Git. The public method cards
contain bibliographic metadata, hashes, page references and original summaries;
they do not redistribute full papers, patient data, or authors' code.

The current mPower endpoint is sustained phonation, participant/component-
disjoint development data, one canonical initial validation assessment, and a
sealed final test. Published numbers below use different cohorts/endpoints and
cannot be compared numerically with our AUROC as if they were the same benchmark.
No single reviewed paper establishes a universal best model for this endpoint.

## Methods suitable for implementation now

| Priority | Evidence | Executable comparison | Fidelity |
|---|---|---|---|
| 1 | LP-FT, ICLR 2022 | Converged standardized HeAR linear probe, then matched head-only/full updates | Transfer of an established vision method to health audio |
| 2 | LoRA, ICLR 2022 | Same LP initialization; freeze backbone and update query/value low-rank matrices plus head | Original mechanism; HeAR/rank/optimizer combination is a project adaptation |
| 3 | PD SupCon, Bioengineering 2025 | HuBERT-base CE versus SupCon with detached classifier input | Preserves the paper's gradient mechanism; mPower phonation differs from NeuroVoz sentences |
| 4 | TabICL, ICML 2025 | Local pretrained classifier on fold-fitted PCA128 cached embeddings; matched LR control | Adjacent tabular method, not established audio SOTA |

Start with one debugging/screening seed and confirm a narrow finalist across
three seeds. Preserve identical visit schedules, people, input spans and head
initialization within matched comparisons. Actual contrastive batches must
contain the intended people simultaneously; accumulation of gradients alone
does not enlarge the contrastive denominator.

## Direct Parkinson's evidence

**Tougui et al., Scientific Reports 2024.** The mPower AST paper uses sustained
phonation, a selected medication/multimodal-overlap cohort, and both subject-wise
and record-wise splits. Its subject-wise mean-of-record-predictions AST AUROC is
.9589, with no final frozen-pretrained AST control. Methods pp14–17: 16kHz,
repeat shorter recordings to 10s, 128-bin log-Mel, 25ms Hamming/10ms hop; AST
keeps its native grid. BCE, batch32, SGD momentum.9/Nesterov/weight-decay.01;
10 frozen-head epochs at .001, then20 transformer-tuning epochs at 1e-5 with
cosine floor1e-6, five runs. Our initial-visit endpoint and no-repeat waveform
policy differ. An authors' mPower-trained checkpoint exists, but unresolved
participant overlap makes direct testing on our held-out split inappropriate.
[Paper](https://www.nature.com/articles/s41598-024-81824-x),
[authors' notebook](https://github.com/Jetliqs/PDVAT).

**Malekroodi et al., Bioengineering 2025.** Full PDF pp3–8 and17–19 establishes
112 NeuroVoz speakers (54PD/58HC), 1,695 recordings of16 repeated sentences;
these experiments do not use the database's sustained vowels. Preprocess at16kHz,
trim ends, pad/crop5s, and augment training with low-pass filtering. Mean sequence
pooling feeds768→64/ReLU/dropout.2→2 classification; projection size32 and
temperature.07. AdamW2e-5, betas(.9,.999), weight-decay.01, batch32, up to100
epochs; speaker-disjoint80/20 holdout and five-fold training CV. Crucially,
classification gradients stop before the encoder: SupCon updates the encoder/
projector, while CE trains the classifier on detached features. Hard-negative
scaling1.5 is reported, but its mining/application formula, projection topology,
scheduler and convolution-freezing policy remain underspecified. Code is
available on request. Main text reports F1 .9000 Wav2Vec2/.8899 HuBERT.
[Paper](https://www.mdpi.com/2306-5354/12/7/728).

This warrants an explicitly **unscaled, adapted detached-SupCon** arm, not a
claim to exact reproduction of the paper's best number. The listed Hugging Face
models are not perfectly mapped to variants in the text; `facebook/hubert-base-ls960`
unambiguously matches the12-layer768D HuBERT description. Existing HuBERT-large
is a different model. The inaccessible supplementary attachment is recorded as
a reproducibility gap, not silently filled with guessed settings.

## Adjacent methods with direct implementation lessons

**LP-FT, Kumar et al., ICLR 2022.** Vision experiments show that converged
linear-probe initialization can reduce feature distortion during later tuning.
Appendix B pp47–48 includes ImageNet batch128, five LP plus five FT epochs,
LP rates [.01,.03,.1], subsequent FT rates [1e-5,3e-5,1e-4]. Other datasets use
different budgets. This supports the initialization principle, not copying those
image learning rates as proven audio optima. A trained standardized LR can be
folded into a raw-embedding linear head, or its frozen scaler can remain in the
differentiable graph. Preserve initial logits before any encoder update.
[Full paper](https://arxiv.org/abs/2202.10054),
[official code](https://github.com/AnanyaKumar/transfer_learning).

**LoRA, Hu et al., ICLR 2022.** Sections4 and7 and Appendix Table9 specify
frozen base weights with additive `(alpha/r) B A`, random A and zero B.
Query/value targeting and rank8/alpha16 occur in RoBERTa-large experiments;
rank8/alpha8 is used for RoBERTa-base. Head/adapter optimizers and rates remain
task-specific. Initial HeAR rank8/alpha16/dropout0 therefore has a real mechanism
and rank precedent, but is not a published PD optimum. Test unchanged initial
logits, nonzero adapter gradients, unchanged base parameters, checkpoint reload
and merged/unmerged agreement. LoRA is distinct from last-block or full tuning.
[Paper](https://arxiv.org/abs/2106.09685),
[MIT implementation](https://github.com/microsoft/LoRA).

**LungAdapter, Interspeech 2024.** On ~5.5h ICBHI lung sounds, official split
has79 training/49 testing patients. Frozen AST adds a serial attention adapter
(width16; depthwise3/5/7 convolutions) and parallel residual MLP adapter
(bottleneck ratio.15), 2.48M trainable parameters. PDF p3:16kHz,8s,128-Mel,
25ms Hamming/10ms hop, normalization(-4.27,4.57); weighted CE, Adam5e-4,
cosine, batch8,50epochs, five runs. Four-class score .6240 is comparable to
the cited .6237 reference; binary score is .6953. This motivates parameter-
efficient adaptation in limited clinical audio, not replacing HeAR with a lung
diagnostic model. Official code/license were not verified.
[Paper](https://www.isca-archive.org/interspeech_2024/xiao24_interspeech.pdf).

**SG-SCL, ICASSP 2024.** The full equation matters: positives are grouped by
the **same device**, with adversarial gradient reversal and stopped target
gradients; it is not ordinary same-diagnosis/different-device SupCon. AST uses
16kHz8s log-Mel, projector768→768 MLPs, tau.06, batch8,50epochs and five seeds.
Official script uses Adam5e-5, weight-decay1e-6, cosine and a moving-average
update. It reports ICBHI score.6171 versus.5955 CE. A mPower device-adversarial
pilot would require common support across diagnosis/device groups and explicit
person/duplicate masking. Do not implement a simplified loss under this name.
[Paper](https://arxiv.org/abs/2312.09603),
[official code](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning).

**CBA-Whisper, Interspeech 2025.** This is dysarthric ASR, not PD detection.
Whisper-large-v2 AdaLoRA starts rank12 and targets4; AdamW1e-3,50 warm-up
steps, batch6, up to2epochs on four GPUs. Transcript-based curriculum,
loudness/VAD preprocessing and text postprocessing contribute to WER10.51.
The '<one-fifth' statement refers to53.48h of official train data; additional
41.23h development data,6.92h UA-Speech and.13h TORGO were also used. Its
improvements do not imply that normalization or transcript filtering preserves
PD phonation biomarkers. Use PEFT as inspiration; do not import its ASR pipeline
or WER number as a diagnosis benchmark.
[Paper](https://www.isca-archive.org/interspeech_2025/tan25b_interspeech.pdf),
[official code](https://github.com/tan90xx/CBA-Whisper).

## New model families and evidence boundaries

**HeAR, Baur et al., 2024.** The original paper evaluates frozen embeddings
with cross-validated ridge/logistic probes, not full fine-tuning. Grouping follows
individual/clip/site where available. Appendix F tests6.25–100% label fractions.
Its313M-scale pretraining corpus provides low-label transfer; this is not training
a303M model from scratch on a few clinical participants. The paper motivates
retaining our strong frozen probe and a label-efficiency curve as essential
controls. [Full paper and appendices](https://arxiv.org/abs/2403.02522).

**VoiceFM, Elemento et al., medRxiv May2026.** This non-peer-reviewed study
uses846 Bridge2AI participants and138 later enrollees. Whisper-large-v2 last4
blocks feed256D vectors aligned to a44-feature clinical TabTransformer. Targets
include diagnoses, demographics and questionnaires, rather than acoustic metrics.
InfoNCE uses learned temperature plus category-BCE.1 and age-MSE.05;
AdamW backbone1e-5/head1e-4, cosine10%warmup, batch32/accumulation2, up to200
epochs/patience25. Its mPower fine-tune reports AUROC.87 under another cohort
and mean-visit endpoint; sustained-vowel NeuroVoz is weaker than sentences.
The MIT repository is code-only; pretrained weights require separate access.
Replacing clinical metadata with ten Praat metrics would be a distinct method.
[Full paper](https://www.medrxiv.org/content/10.64898/2026.05.28.26354346v1.full.pdf),
[official repository](https://github.com/oelemento/VoiceFM-public).

**TabPFNv2 and TabICL.** Small-data tabular foundation models are a useful
additional downstream family: initial-visit cached embeddings create a table
of3,397 training participants. Nature2025 TabPFNv2 targets <=10k rows/500
features; its weight license adds attribution conditions. ICML2025 TabICL has
permissive BSD-3-Clause weights and local sklearn-style inference. Compare a
version-pinned TabICL on train-fold PCA128 to a matched LR pipeline. Current
TabICL defaults point to newer2026 weights, so the original2025 checkpoint and
its original32-member ensemble must be explicit for historical reproduction.
No remote inference, validation-label context, or whole-cohort PCA is allowed.
[TabPFN paper](https://www.nature.com/articles/s41586-024-08328-6),
[TabICL paper](https://proceedings.mlr.press/v267/qu25d.html),
[official API and weights](https://github.com/soda-inria/tabicl).

**Speech-emotion adaptation.** The reviewed WavLM-Large paper uses learned
layer aggregation, time pooling, a small head, frozen CNN frontend, AdamW1e-5,
weight-decay.01, batch32 and20epochs with5s training crops. Its pooling/text/
gender ablations motivate small controlled readouts, but its emotion labels and
semantic tasks differ from sustained phonation. No evidence here justifies
downloading another large audio decoder ahead of the methods above.
[Paper](https://arxiv.org/abs/2405.04485).

## Reproduction requirements

Record dataset selection, sample unit, source span, augmentation, split keys,
head initialization, loss gradient routes, actual contrastive batch size, trainable
names and checkpoint selection. Separate exact numerical reproduction on the
original dataset from method adaptation to mPower. Keep controls for age/device,
missingness and participant identity. Report AUROC, average precision, calibration
and locked-threshold operating points; a published WER/F1/ICBHI score is not an
interchangeable target. No reviewed source guarantees improvement on our cohort.
