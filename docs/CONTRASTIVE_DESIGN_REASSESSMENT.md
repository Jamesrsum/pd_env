# Reassessment: what should contrastive learning learn?

November 14, 2026. User requested a fresh research-agent analysis before launching contrastive work. All three agents were rerun independently. **No contrastive preparation or training job has been submitted.** The narrow ten-Praat-feature prototype remains an unexecuted ablation, not the accepted production design. Independent AST/ResNet50 and partial wav2vec2/WavLM supervised experiments are unaffected.

## Correction to the initial implementation

Recomputing features serves a temporal-alignment requirement, not feature selection. It is unnecessary to recompute an audited feature when its source waveform and exact interval already match the encoder input. Conversely, when the encoder sees a cropped window, ALL targeted acoustic descriptors must correspond to that window. The implementation prototype selected ten Praat descriptors; that was neither a comprehensive acoustic view nor an evidence-based optimal teacher.

The existing extractor already provides Praat pitch/perturbation/HNR, 88 eGeMAPSv02 functionals and spectral/MFCC summaries. Reducing that collection to jitter/shimmer would throw away potentially complementary structure. Adding every feature indiscriminately is also unsuitable: correlated perturbation variants can dominate an auxiliary loss, and device-dependent features can strengthen shortcuts.

## Three scientific questions, three controlled comparisons

| Question | Recommended first comparison | What success would mean |
|---|---|---|
| Does a discriminative representation improve PD prediction? | CE/BCE vs the same encoder/head plus diagnosis-supervised contrastive loss | Better unseen-participant development metrics under matched capacity and confounding controls |
| Does the representation capture interpretable phonatory properties? | Audio-only vs masked, group-balanced acoustic regression; compare fusion, InfoNCE and regression+InfoNCE | Descriptor prediction skill and downstream value beyond simple fusion, not merely pair retrieval |
| Are two pretrained encoders complementary? | Whisper, WavLM, concatenation and late fusion; then VICReg/Barlow/InfoNCE under the same budget | Alignment beats complementary fusion without collapsing or amplifying nuisance signals |

There is no verified source showing that a single one of these is universally best for mPower sustained vowels. The agents' different priorities reflect different objectives, not contradictory performance evidence. We should not preselect a complex contrastive method merely because it is newer.

## Literature-supported options and limits

[Malekroodi et al., 2025](https://www.mdpi.com/2306-5354/12/7/728) provides direct PD precedent for Wav2Vec2/HuBERT and supervised contrastive learning on NeuroVoz. It is not evidence for Whisper–Praat alignment or our mPower cohort. A CE/BCE-only versus CE/BCE-plus-SupCon comparison is the closest discriminative follow-up. [SupCon](https://arxiv.org/abs/2004.11362) defines multiple label-positive examples; participant-balanced batches should include both diagnoses and several people per class. Diagnosis positives can compress clinically meaningful heterogeneity, so this is an ablation rather than a replacement for all baselines.

[CLIP](https://proceedings.mlr.press/v139/radford21a.html) supports symmetric paired-view InfoNCE as a construction. Same-recording Whisper/acoustic pairing does not supply independent measurements: both are deterministic functions of one waveform. Instance matching can preferentially encode identity, age, device or recording conditions. Similar phonation from different people need not be biologically negative; false-negative pressure is a substantive concern.

[VICReg](https://arxiv.org/abs/2105.04906) combines invariance with variance and covariance terms, avoiding explicit negatives while controlling collapse. [Barlow Twins](https://arxiv.org/abs/2103.03230) is a redundancy-reduction alternative. [BYOL-A](https://arxiv.org/abs/2103.06695) is audio-specific evidence for negative-free representation learning. These justify controlled speech experiments, not claims of superior PD performance. Negative-free objectives can still align nuisance information, and forcing two representations to agree can discard useful branch-specific information.

[data2vec](https://arxiv.org/abs/2202.03555) supports latent prediction/self-distillation as another methodological precedent. A fixed Whisper teacher and partially adapted WavLM student would be our cross-model modification, not a reproduction of data2vec. Compare that student with identical supervised partial adaptation without distillation.

## Broader acoustic targets, grouped by scientific role

| Group | Candidate descriptors | Qualification |
|---|---|---|
| Pitch and phonation stability | F0 mean/SD/CV, valid-pitch fraction, representative jitter and shimmer | Do not overweight correlated variants; tracking failures remain missing |
| Harmonic and cepstral organization | HNR; CPP/CPPS after validation | CPP is not yet implemented/validated in this repo |
| Spectral shape | Alpha ratio, Hammarberg index, slopes, centroid/bandwidth/rolloff/flatness/contrast | Strong microphone/channel sensitivity is possible |
| Cepstral/timbre | Compact MFCC groups and eGeMAPS descriptors | Useful representations need not map uniquely to physiology |
| Within-vowel dynamics | Spectral flux, F0/intensity variation | Acquisition artifacts and physiological variation can overlap |
| Vocal-tract estimates | Reliable formant locations/bandwidths, if tracking validates | No vowel-space, connected-speech rate or articulation claims from one sustained vowel |

[eGeMAPS](https://doi.org/10.1109/TAFFC.2015.2457417) supplies standardized descriptors across several of these roles; preserve its names and definitions. Do not interpret voicing-region counts as syllable rate. Praat has an established [CPPS command](https://www.fon.hum.uva.nl/praat/manual/PowerCepstrogram__Get_CPPS___.html), but its smoothing, pitch search and trend-fitting settings require explicit validation. [Šimek and Rusz](https://pubmed.ncbi.nlm.nih.gov/34972306/) motivate studying cepstral prominence while examining ambient-noise effects. Additional targets are hypotheses, not automatically better biomarkers.

## Preferred interpretable objective

For training-only, standardized acoustic targets, define a group-balanced masked Huber auxiliary loss:

`L = L_PD + lambda_reg * mean_over_observed_groups(masked_Huber_group) + lambda_align * L_alignment`.

Within a feature group, average only genuinely observed entries. Skip groups with no observed targets. Imputed values may enter a tabular encoder with a missingness mask, but must NOT become ground-truth targets in regression. Fit feature selection, scaling, imputation and optional collinearity reduction only on training participants. Use a compact predeclared panel and a broader eGeMAPS/spectral panel as separate arms. Do not allow many near-equivalent jitter measurements to outweigh an entire harmonic/spectral group.

Report per-feature and per-group MAE/R² on unseen participants with coverage and target variance. An audio-only deployment uses the learned acoustic encoder without running Praat at inference. A fusion deployment requires both branches; report their costs and coverage separately. Auxiliary regression is not itself contrastive learning and must be named accurately.

## Preferred cross-encoder experiment

Begin with frozen Whisper and WavLM representations for identical intervals, then train projection heads. Compare individual encoders, concatenation and prediction-level fusion BEFORE alignment. Test VICReg first as a negative-free candidate, with same-budget masked InfoNCE and Barlow alternatives. Use modest projection dimension and batches of distinct participants; measure actual batch size. Gradient accumulation does not enlarge the sample set used for covariance or InfoNCE negatives.

Only if alignment shows value over fusion should we partially unfreeze an encoder or add teacher–student distillation. Training projectors on cached Whisper vectors does not update Whisper itself; a claim of Whisper fine-tuning requires differentiable waveform/frontend forwards, verified encoder changes and new feature provenance.

## Required validity controls

- Keep validation and test participants out of alignment pretraining even when their labels are ignored. Keep the test sealed through design and selection.
- Keep each participant and identified duplicate component in one split. Avoid using other visits by the same person as negatives. Different visits are not automatically positives because medication and physiology can change.
- Use the exact same evaluation participants when comparing candidate methods; separately report the larger audio-only cohort and pair-availability exclusions by diagnosis, age and device.
- Do not declare pitch shift, time stretch, aggressive noise, gain or spectral masking invariant to the measured targets. Re-extract changed targets or make augmentation a separate, validated arm.
- Track representation variance, effective rank, covariance, retrieval and target-prediction error. Lower contrastive loss or better recording retrieval does not establish disease prediction improvement.
- Compare against age-only/age-device controls, age/gender-matched analyses and incremental value over demographics. Current age-only AUROC (0.9330) exceeds every audio model; this is the central validity issue.
- Use paired participant bootstrap intervals for comparative performance; never resample visits as independent people. Report AUROC, AUPRC, log loss/calibration and the predeclared operating point.

## Revised experiment matrix

1. Broad acoustic-only, Whisper-only, WavLM-only, concatenation and late fusion.
2. CE/BCE versus CE/BCE+SupCon for a fixed encoder, sampling scheme and classifier.
3. Masked grouped regression, instance InfoNCE, and their combination; include original audio-only and fusion endpoints.
4. Whisper–WavLM VICReg, Barlow and masked InfoNCE with equal budgets.
5. Shuffled-training-pair, missingness/quality-only, and demographic/device probes.
6. Later: partially adapted encoder with and without auxiliary regression/distillation, only after simpler comparisons justify it.

HeAR can enter the encoder comparisons only after complete weights and runtime checks. No numeric result should appear for any unexecuted arm. This document supersedes the assumption that the narrow-Praat contrastive prototype is ready for a production run.

## Independent supervised execution status

Slurm setup job 27594680 verified ImageNet ResNet50 and isolated TorchVision. GPU smoke 27594728 completed with exit 0 in 3:10; all six architectures updated intended backbone parameters (AST, ResNet50, two wav2vec2 and two WavLM variants). Six synthetic adaptation tests passed, including test-row rejection, participant sampling, real weight updates and resume. The initial array 27594729 was cancelled after detecting ignored dependency environment settings; its premature summary is not a completed experiment. The helper now accepts explicit dependency/array arguments. After checking completed smoke evidence, supervised array **27595280** was submitted (14 tasks, maximum two concurrent GPUs), with summary **27595295** depending explicitly on it. No contrastive job was submitted. These job submissions do not imply completed training or new predictive results.
