# Literature review and next-stage experiment design

**Completion update (November16):** the original supervised runs, corrected
AST/HeAR controls and later LP-FT/LoRA/HuBERT-SupCon experiments have finished.
See the [executed model report](../notebooks/02_All_Models_Methods_and_Results.ipynb).
Statements below about missing weights, unrun adaptation and absent scores
describe the November14 review snapshot. Whisper/acoustic paired alignment
remains untrained and is distinct from the completed HuBERT diagnosis-SupCon arm.

Evidence checked November 14, 2026. Contrastive recommendation updated after a second three-agent review: see [design reassessment](CONTRASTIVE_DESIGN_REASSESSMENT.md). The narrow-Praat alignment prototype is on hold. Three research agents independently reviewed AST/ResNet50, HeAR/partial speech-encoder adaptation, and contrastive alignment. Completed results, proposed experiments, and tested building blocks are distinguished below. No fine-tuning, contrastive training or held-out evaluation was launched for this review.

## Existing results and what happened to wav2vec2/WavLM

All nine backbones completed frozen development experiments: 18 model/context combinations, 364 pipelines, a common 53,122-recording / 4,126-participant development snapshot, 729 validation participants, and zero extraction failures. The encoders stayed frozen; logistic/MLP heads were trained. Thus wav2vec2 and WavLM were included, but were not fine-tuned. Their lower frozen scores do not establish that adaptation will be ineffective.

See [all completed models](FROZEN_DEVELOPMENT_RESULTS.md), [PNG comparison](results/all_models/all_models_validation.png), [SVG comparison](results/all_models/all_models_validation.svg), and [all 364 aggregate pipeline results](results/all_models/all_364_development_pipelines.csv). The technical-primary arm uses 729 validation participants; the pitch-sensitivity arm uses 725, so their curves have different eligibility. The age-only reference applies to the primary 729-person cohort. Rankings select configurations on validation and are optimistic as estimates of generalization. Test labels remain unopened. No new method has a score yet; [status table](results/all_models/model_status.csv) records missing results explicitly.

Age-only AUROC is 0.9330, exceeding the best audio model's 0.8951. Any claim that fine-tuning finds disease-specific vocal cues therefore needs demographic controls, matched/restricted analyses, and incremental predictive value beyond demographics. An improved pooled AUROC alone does not answer that question.

## Literature and implications

[Tougui et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11614913/) compares AudioSet-initialized AST and ImageNet-initialized CNNs on a differently selected mPower cohort. Its subject-wise, mean-prediction AST/ResNet50 AUCs are 0.9589/0.9232. The method uses 16-kHz audio, 128-bin Mel features (25-ms window, 10-ms hop), and 224×224 three-channel inputs for ResNet50. The schedule is 10 head-only epochs at 1e-3, then 20 adaptation epochs with cosine decay toward 1e-6; AST starts adaptation at 1e-5 and CNNs at 1e-4. SGD uses momentum 0.9, Nesterov, weight decay 0.01, batch 32, and BCE. The narrative preserves AST's time–frequency frontend, while Table 9 lists image-resizing steps: exact replication needs clarification. Cohort restrictions, repeated short audio, aggregation and split differences preclude a direct score comparison. We propose a Tougui-inspired adaptation, retaining participant independence rather than its record-wise development alternative.

The original [AST paper](https://sls.csail.mit.edu/archives/root/publications/2021/YuanGong_Interspeech-2021.pdf) motivates pretrained spectrogram transformers; it does not establish PD-specific optimal settings. Preserve the pinned checkpoint's frontend and positional-grid conventions. ImageNet [ResNet](https://arxiv.org/abs/1512.03385) supplies a CNN comparison; ImageNet image recognition is pretraining, not an acoustic validation claim.

[wav2vec 2.0](https://arxiv.org/abs/2006.11477) learns waveform representations through self-supervision. [WavLM](https://arxiv.org/abs/2110.13900) targets a broad range of speech tasks, including information related to speakers. Their success in speech does not automatically imply disease specificity. Partial adaptation is a hypothesis we can test while keeping lower representations fixed.

[HeAR](https://arxiv.org/abs/2403.02522) is a health-acoustic representation model, primarily evaluated using frozen embeddings. It is a useful domain-pretraining comparator, not an established Parkinson classifier. The [official TensorFlow model card](https://huggingface.co/google/hear/blob/main/README.md) specifies 16-kHz mono, 32,000-sample windows and 512-dimensional output. The [PyTorch model](https://huggingface.co/google/hear-pytorch) is a separate distribution and requires its official preprocessing; AST filterbanks cannot be substituted. Runtime compatibility must be verified before scheduling extraction.

[CLIP](https://proceedings.mlr.press/v139/radford21a.html) supplies a general paired-view contrastive construction. Our two views would be acoustic measurements and Whisper encoder features rather than text/images. [Supervised contrastive learning](https://arxiv.org/abs/2004.11362) defines class-based positives; that is a different experiment from matching recordings. Neither source validates the proposed Parkinson application. [False-negative cancellation](https://openaccess.thecvf.com/content/WACV2022/papers/Huynh_Boosting_Contrastive_Self-Supervised_Learning_With_False_Negative_Cancellation_WACV_2022_paper.pdf) motivates checking whether apparently negative pairs actually share identity or duplicated content.

A closer disease-specific precedent is [Malekroodi et al., 2025](https://www.mdpi.com/2306-5354/12/7/728), which investigates supervised contrastive learning with Wav2Vec2/HuBERT and compares handcrafted representations on NeuroVoz. It supports including a diagnosis-supervised contrastive ablation, but does not validate Whisper–Praat pairing, mPower vowel analysis, or our expected performance. Our cross-view objective is a distinct hypothesis.

## HeAR cache audit and implementation

Observed local Hugging Face repository: `google/hear`, revision `9b2eb2853c426676255cc6ac5804b7f1fe8e563f`. The inspected snapshot has only `.gitattributes` (1,819 bytes) and README (11,895 bytes). No `saved_model.pb` or variables are present. This is an incomplete asset download, not an inference failure. TensorFlow and torchvision are absent from the current benchmark environment; PyTorch is available. No credentials were read or printed, and no additional checkpoint download was started.

Use `MPOWER_HEAR_CHECKPOINT` for an explicitly configured complete local snapshot. Verify required files and hashes, then load offline in an isolated TensorFlow environment using `tf.saved_model.load`; inspect its signature before binding `serving_default(x=...)` and the output key. Legacy Hugging Face Keras-loading helpers have separate compatibility requirements; do not destabilize the working PyTorch benchmark environment to satisfy them.

First add HeAR frozen embeddings to the identical matched-2-second development windows. Average windows within recording and retain the current initial-recording participant endpoint. Fit heads/scalers on training only. Store revision, file hashes, preprocessing signature, shape, dtype, device and timing. Require synthetic shape tests, real-window smoke inference, cache reload equality and single-vs-batch independence before full extraction. The official [PyTorch frontend source](https://raw.githubusercontent.com/Google-Health/hear/master/python/data_processing/audio_utils.py) deserves a batch-composition check because its normalization uses batch-global extrema; preprocessing individual clips is a conservative alternative. Keep research audio local.

## AST and ResNet50 adaptation (proposed)

Implement a separate differentiable runner. The frozen runner intentionally uses `inference_mode()`, disables gradients and reuses embeddings, so it cannot be converted into fine-tuning by changing one configuration flag. No cached frozen activations can stand in for a waveform forward while encoder weights change.

AST: attach a one-logit classifier to the pretrained encoder's documented pooled representation. Compare frozen encoder, last two transformer blocks, and full adaptation. Use the paper-inspired staged schedule above as a declared reference arm. Keep native filterbank normalization; any train-estimated alternative must fit training only. Verify gradients and weight changes only in selected modules, including the head. Record final-norm/pooler policy explicitly; the prototype layer selector leaves them frozen for partial arms.

ResNet50: pin a local ImageNet checkpoint; replace its final classifier with one binary output. Convert log-Mel features to resized 224×224 tensors and replicate channels without colormap images or plot axes. Explicitly document log scaling and normalization, including whether ImageNet channel normalization is used. Compare head-only, final residual stage (`layer4`), and full-network adaptation. Freeze BatchNorm running statistics during head-only training; declare whether full adaptation updates them. Do not silently apply image flips, which reverse time or frequency semantics. Repetition of short audio belongs only in a separate replication sensitivity arm; use the existing valid-length policy in the primary comparison.

Use equal participant exposure or inverse-recording-count loss weighting. The effective batch of 32 can use gradient accumulation, but accumulation does not reproduce BatchNorm batch statistics. Predeclare five seeds and save the best validation checkpoint under one selection rule. Measure AUROC, AUPRC, log loss, calibration, sensitivity at the specified operating point, participant-bootstrap uncertainty, wall time and peak GPU memory.

## Partial wav2vec2 / WavLM fine-tuning (proposed)

Start from all four existing wav2vec2/WavLM checkpoints. Warm up a binary pooling/classifier head, then unfreeze only the last two or four transformer blocks; keep the convolutional frontend frozen. Proposed starting search: AdamW, backbone LR 1e-6/5e-6/1e-5, head LR 1e-4. These are our tuning candidates, not literature-validated PD optima. Keep search budgets comparable across architectures.

Respect [checkpoint-specific normalization and masks](https://huggingface.co/docs/transformers/model_doc/wav2vec2). Pool only valid output frames. Use length bucketing or per-recording differentiable forwards with accumulated gradients when padding changes representations. Set frozen lower blocks to evaluation mode and selected blocks/head to training mode; freezing parameters alone does not freeze dropout or BatchNorm state. Disable latent time masking in the primary sustained-vowel arm and label any masking augmentation separately. Use no CTC/ASR decoder. Validate checkpoint reload, gradient flow, frozen-parameter invariance, finite outputs and per-participant split isolation before a bounded Slurm training pilot.

## Whisper–acoustic contrastive experiment (proposed)

The scientific hypothesis is that recording-paired alignment makes an acoustic representation more sensitive to interpretable phonatory variation. It does not imply that jitter/shimmer are independent views, causal biomarkers, or noise-free targets: both inputs are derived from the same signal.

1. Join waveform/Whisper features and acoustic rows by verified recording ID, with one-to-one cardinality, source hashes and identical analysis interval. Begin with full-interval acoustic measurements and the same complete input span. If the model sees only the first 10 seconds, recompute acoustic targets on that span; do not pair a cropped embedding with full-file targets. Stable-interval jitter/shimmer belong in a separate interval-aligned arm. Do not align a whole recording to a different visit's measurement.
2. Use a prespecified compact panel: jitter local/RAP/PPQ5, shimmer local/dB/APQ, HNR, F0 variability, and selected spectral/eGeMAPS descriptors. Fit column selection, all-missing handling, imputation, scaling and any transforms on training only. Preserve missingness with masks; undefined perturbations are never biological zeros. Retain recordings with invalid acoustic pairs in the audio-only baseline and report paired-subset losses separately.
3. Start with frozen Whisper Small intermediate-layer features, with Large-v3 as a prespecified replication rather than automatically expanding every configuration. Project Whisper and acoustic vectors through separate MLPs into 128-dimensional normalized embeddings. Optimize symmetric recording-paired InfoNCE, proposed temperatures 0.05/0.1/0.2. The new module implements these projections and loss but is not a fitted training pipeline.
4. Sample one recording per participant per batch. Mask same-participant and duplicate-component off-diagonal negatives; require genuine negatives for every anchor. Do not declare every PD recording a positive in the primary instance-alignment experiment. Distinct people with similar dysphonia can still be false negatives, so temperature, matched negative sampling and class-aware loss belong to explicit sensitivity analyses. Never include validation/test recordings in contrastive pretraining, even without labels.
5. Freeze the projections after training and assess audio-only aligned embeddings using the same downstream head protocol. Also assess fusion of the two projections. Only fusion requires acoustics at deployment; audio-only aligned inference can omit them. Joint PD supervision or partial Whisper unfreezing is a later, separately labeled arm, not a hidden change to unsupervised pairing.
6. Avoid pitch shift/time stretch in the primary arm because they modify the intended targets. An augmented view requires either re-extracted acoustic targets or a justified invariant-target policy. Broad augmentations that distort jitter/shimmer can teach the model to ignore the very signal being investigated.

Required ablations: acoustic-only, original Whisper-only, concatenation/late fusion, aligned audio-only, aligned fusion, shuffled-training-pair control, missingness-only control, and a supervised-contrastive secondary arm if pursued. Compare all on the same paired subset in addition to reporting each method's coverage. Evaluate cross-view retrieval and target prediction only on unseen participants; retrieval improvement alone is not PD prediction improvement. Inspect representation variance/collapse, age/device predictability and demographic-matched results.

## Execution sequence and decision points

1. Resolve the incomplete HeAR cache and pin an available ResNet checkpoint/runtime.
2. Preserve the frozen development results and sealed test. Finish task/duplicate review; freeze a common eligibility manifest and specify demographic sensitivity analyses before selecting new methods.
3. Implement a differentiable runner, real checkpoint loaders and Slurm smoke tests for AST/ResNet/partial speech encoders. Existing new components are building blocks only; no production fine-tuning command exists yet.
4. Run bounded development pilots with equal participant sampling, then expand only after finite-loss/gradient/reload checks. Use `scripts/submit_mpower.sh` and keep all jobs/logs/checkpoints outside Git; training must not run on the login session.
5. Run the alignment ablations first with frozen cached Whisper inputs and training-only acoustic preprocessing; unfreeze Whisper only if the controlled result justifies the additional experiment.
6. Select models using development data, lock thresholds/aggregation and report all arms. Open the final test only under the existing release protocol. Do not claim a new external generalization result from more validation searches.

## Delivered implementation and verification

- `configs/adaptation_research.yaml`: explicit research-design values; not a runnable training configuration.
- `src/models/adaptation.py`: validated layer-selection policies and BatchNorm-statistics freezing helper.
- `src/training/acoustic_alignment.py`: normalized dual projections, symmetric paired loss, participant/duplicate negative masking and invalid-pair checks.
- `tests/test_adaptation_research.py`: synthetic gradient, freeze-policy, malformed-batch and batch-statistics checks.
- `scripts/plot_all_model_results.py`: reproducible aggregate-only PNG/SVG, complete 364-pipeline CSV and unrun-model status table.

Actual training loops, paired-data loaders/train-fitted transforms, model-specific differentiable frontend adapters, evaluation integration, and Slurm runners are still required before any new experiment can be called implemented end to end. No scores are assigned to HeAR, ResNet fine-tuning, partial adaptation or contrastive learning until those runs actually execute.

Research-agent synthesis: the AST/ResNet reviewer flagged the paper’s inconsistent AST frontend table; the HeAR reviewer independently confirmed missing checkpoint assets and highlighted batch-global preprocessing normalization; the contrastive reviewer identified the NeuroVoz precedent and recommended separating instance pairing from diagnosis supervision. These findings are incorporated above rather than treating any proposed recipe as an executed result.

Execution update: independent supervised training is now implemented and submitted following GPU smoke validation; see the execution status in [design reassessment](CONTRASTIVE_DESIGN_REASSESSMENT.md). The earlier implementation-incomplete statements describe the initial review snapshot. Contrastive training remains held; its prototype has not been run.
