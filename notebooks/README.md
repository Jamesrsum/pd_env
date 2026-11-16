# Completed research reports

Open these in order. Both contain saved aggregate tables, figures and explanations;
GitHub readers do not need to run them or obtain participant data.

1. [Dataset characterization](01_mPower_Dataset_Characterization.ipynb): full-data
   census, files/schema, age/diagnosis/repeated-visit/duration distributions, QC,
   feature definitions, exploratory associations, cohorts, splits and failures.
2. [All model methods and results](02_All_Models_Methods_and_Results.ipynb): all
   completed production comparisons, shared QC/native frontends, training and
   downstream procedures, selected hyperparameters, exact configurations, paired
   uncertainty intervals, provenance, failures and paper-by-paper methodology.

These are development results. The final test remains sealed. Smoke/cancelled
partial runs are not scientific result rows. Different procedures are separated
so that a validation-selected maximum is not mistaken for a matched control.

To regenerate from the reviewed, checked-in aggregates:

```bash
python -m pip install -r requirements-reports.txt
python scripts/build_results_notebooks.py
```

Run from the repository root. Notebook code reads only `docs/results`; no Slurm
jobs, downloads, credentials or audio are required. Jupyter must be able to bind
local kernel sockets. Rebuilding updates stored outputs and execution counters.

An authorized HPC researcher can refresh the underlying aggregates separately:

```bash
python scripts/analyze_completed_literature.py \
  --root "$MPOWER_BENCHMARK_ROOT/literature_methods/20261116T091246Z" \
  --snapshot "$MPOWER_BENCHMARK_ROOT/protocol_experiments/20261114T074847Z/snapshot"
python scripts/build_results_notebooks.py --refresh-private-aggregates \
  --eda-root "$MPOWER_OUTPUT_ROOT" --benchmark-root "$MPOWER_BENCHMARK_ROOT"
```

Refreshing requires the benchmark environment and approved data access. Review
exports before publication. Report generation does not retrain models or read
final-test manifests. The original output-free EDA source template is preserved
under `private/` and is still used by `scripts/execute_characterization_notebook.py`;
its executed outputs remain in private HPC storage. It is a separate workflow.

The 100 real training recordings are prepared outside Git; see
[the sample handoff](../samples/mpower_100/README.md).
