# Frozen-model development results — November 14, 2026

Verified from Slurm accounting, all 18 DONE records, extraction failure files, the shared snapshot, and the saved validation leaderboards. Private source evidence: `mpower_benchmarks/protocol_experiments/20261114T074847Z/` under approved HPC storage.

Array **27564045** completed all 18 tasks with exit code 0; summary job **27564046** completed with exit code 0. Nine models ran in two contexts (full first-10-second context and matched 2-second windows). All conditions consumed the same 53,122-recording / 4,126-participant development manifest: 43,702 training recordings from 3,397 participants and 9,420 validation recordings from 729 participants. Rankings use one initial eligible recording per validation participant (146 PD, 583 controls), rather than counting repeated recordings as independent assessments.

All 18 conditions have zero saved extraction failures and unchanged frozen encoder states. Completion records and summary report no test-label access. There are 364 fitted pipelines: 182 technical-primary and 182 pitch-sensitivity configurations. No missing AUROC or log-loss values were found. Summary stderr was empty; array stderr had no traceback, OOM or convergence warning markers.

## Best technical-primary configuration per model

These are validation-selected maxima across contexts, representations and heads, not unbiased final-test estimates.

| Model | Context | Head | Validation AUROC | Log loss |
|---|---|---|---:|---:|
| whisper_large_v3 | matched_2s | logistic | 0.8951 | 0.3056 |
| whisper_small | matched_2s | logistic | 0.8878 | 0.3207 |
| ast_audioset | full | mlp | 0.8867 | 0.3201 |
| whisper_medium | full | mlp | 0.8867 | 0.3141 |
| wavlm_large | matched_2s | logistic | 0.8822 | 0.3232 |
| hubert_large | matched_2s | logistic | 0.8816 | 0.3269 |
| wav2vec2_large_robust | matched_2s | logistic | 0.8774 | 0.3290 |
| wavlm_base_plus | matched_2s | logistic | 0.8589 | 0.3552 |
| wav2vec2_base_960h | matched_2s | logistic | 0.8243 | 0.3817 |

The predefined summary shortlists Whisper Large-v3 and Whisper Small, both using matched 2-second windows, intermediate-layer mean/SD representations and logistic heads. Their validation sensitivities are 71.9% and 65.8%, respectively, at validation-calibrated specificity 89.9%. These thresholds were selected on validation controls; they are not externally validated operating points.

## Confounding and interpretation

Age-only validation AUROC is **0.9330**; age plus device is **0.9337**; device-only is **0.6368** on the same 729 participants. Age alone outperforms the best audio configuration (**0.8951**). Thus the current rankings do not establish voice-specific diagnostic value beyond demographic confounding. Differences between audio models are descriptive; no significance or generalization claim follows from these maxima.

The final test remains sealed. Task verification, near-duplicate review, demographic-matched or adjusted evaluation, and a locked selection/evaluation protocol remain necessary before final performance claims or fine-tuning. No final-test run or further training was launched while checking these results.

Machine-readable aggregate evidence: [frozen_development_summary.json](results/frozen_development_summary.json).
