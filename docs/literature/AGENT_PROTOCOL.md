# Single literature-review agent protocol

Use one reviewer, with no child agents. Review primary full PDFs, appendices,
supplements and authors' code, including directly relevant adjacent fields.
Use the paper's experimental sample unit and evaluation design to judge its
relevance; do not rank methods by incomparable headline accuracy/WER/F1 scores.

1. Keep the bibliography, PDF SHA256, source URL, page count and extraction
   completeness in `pdf_manifest.json`. Download/extract full PDFs privately
   with `scripts/archive_literature.py`; the existing archive is under
   `$MPOWER_BENCHMARK_ROOT/literature_review`. A download failure is a recorded
   gap, not permission to bypass access restrictions or fabricate settings.
2. For each method, record dataset/cohort/participant count, signal task, input
   span and preprocessing, initialization/revision, layers trained, head,
   augmentation, objective and gradient routing, optimizer/LR/scheduler,
   actual contrastive batch, stopping/checkpoint selection, seeds, split unit,
   metrics and code/license status. Give PDF page/table references.
3. Inspect formulas and code rather than inferring mechanisms from an abstract.
   Record missing/contradictory settings explicitly. Distinguish train/validation
   data, gradient accumulation from negative-batch size, and publication
   pretraining data from the apparent small downstream label count.
4. Map each reviewed mechanism to this project's sustained-vowel endpoint.
   Preserve participant/duplicate-disjoint splits and the sealed test. Flag
   pretrained disease checkpoints with unresolved subject overlap. Classify the
   implementation as original-dataset reproduction, method adaptation, or an
   exploratory hypothesis. List every material deviation.
5. Produce original summaries in `method_cards.json` and `REVIEW.md`, and select
   a bounded executable matrix with matched controls and train-only transforms.
   Keep full paper text, participant information and model weights outside Git.
   Code/model references must use explicit revisions and checksums.

The 2026-11-16 review used exactly one agent, reviewed/extracted 13 full PDFs,
and produced the cards already in this folder. It supports the LP-FT/LoRA,
detached-SupCon and local TabICL comparisons; it does not certify a universal
state-of-the-art model for mPower or guarantee improved validation performance.
