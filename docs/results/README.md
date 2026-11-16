# Aggregate evidence for the completed notebooks

Snapshot: November16,2026. Start with [the executed notebooks](../../notebooks/README.md).
All predictions are development validation; the final test remains sealed.

| Folder/file | Evidence and source |
|---|---|
| `dataset_characterization/` | Reviewed full-scope EDA tables/figures from `MPOWER_OUTPUT_ROOT`; new age/visit/duration bins contain counts only. Source inventory omits cache filenames and individual values. |
| `all_models/all_364_development_pipelines.csv` | Full9-encoder ×2-context frozen sweep, including182 primary and182 pitch-sensitivity rows. Primary n729; sensitivity n725. |
| `all_models/selected_hyperparameters.csv` | Actual selected LR C or MLP epoch for every frozen pipeline. |
| `all_models/head_search_history.csv` | Complete training-CV C scores and MLP validation epoch histories. |
| `all_models/model_status.csv` | Updated cross-experiment completion index; do not interpret differing protocols as a matched leaderboard. |
| `historical_adaptation/` | All14 original30-epoch runs; explicitly labeled requested policy vs selected phase, with original configurations/history. |
| `hear_completed/` | Earlier HeAR frozen/tuning, corrected AST and downstream/fusion metrics, audits and paired uncertainty. |
| `downstream_completed/` | Actual selected classifier/PCA/C and complete panel training-CV search; selected earlier HeAR LR C values. Earlier HeAR cached-head CV history/MLP epoch metadata were not persisted; this gap is explicit. |
| `literature_completed/` | Five completed strong-LP/full/LoRA and HuBERT CE/SupCon arms; independently recomputed metrics, histories, configs, parameter audits and paired intervals. |
| `literature_tabular/` | All18 initial-visit/local TabICL and matched LR comparisons, plus actual LR training-CV search. |
| `acoustic_xgboost/` | All7 acoustic/control fits;60 GPU training-CV trials; selected parameters and aggregate SHAP importance. |
| `native_frontends.json` | Native checkpoint preprocessing configs and the pinned HeAR frontend constants. |

Original private evidence snapshots are under `MPOWER_BENCHMARK_ROOT`:

```text
protocol_experiments/20261114T074847Z/
adaptation_20261114/
controlled_ast/20261116T002439Z/
hear_protocol/20261116T011601Z/
literature_methods/20261116T091246Z/
  speech_retry_20261116T092846Z/    corrected HuBERT production only
acoustic_xgboost/20261116T162251Z/
```

No cancelled partial run or smoke score is presented as a completed production
comparison. Earlier submission/cache JSON files remain historical evidence; a
failed initial download is not the present model status. Notebook refresh does
not open final-test labels. No participant rows, source exports, audio, embeddings,
fitted weights, per-person SHAP values or raw logs are published.

Paired bootstrap intervals condition on fitted/selected models and the observed
validation cohort. They do not account for training seeds, repeated validation
selection or multiple comparisons. They are not independent-test clinical evidence.
