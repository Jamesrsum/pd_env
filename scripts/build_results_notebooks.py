#!/usr/bin/env python3
"""Build portable, executed notebooks from reviewed aggregates, optionally refresh HPC exports.

No notebook reads participant records, credentials, checkpoints, or private paths.
Refreshing exports is an explicit separate step performed only on approved HPC storage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'docs/results'
EDA_TABLES = [
    'cohort_characteristics', 'demographic_coverage', 'demographics_by_diagnosis',
    'medication_by_diagnosis', 'longitudinal_by_diagnosis', 'full_audio_summary',
    'duration_threshold_counts', 'quality_by_diagnosis', 'qc_threshold_sensitivity',
    'full_feature_summary', 'praat_opensmile_comparison', 'exploratory_pd_associations',
    'covariate_adjusted_pd_associations', 'qc_exclusion_flow', 'cohort_demographic_balance',
    'age_gender_matching_flow', 'cohort_A_splits', 'cohort_B_splits', 'cohort_C_splits',
    'cohort_A_split_demographics', 'cohort_B_split_demographics', 'cohort_C_split_demographics',
    'participant_spearman_correlations',
]
EDA_FIGURES = ['demographic_distributions', 'diagnosis_availability', 'longitudinal_recordings',
               'audio_duration', 'qc_distributions', 'acoustic_distributions', 'selected_feature_correlations']


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def refresh(eda, benchmark):
    import numpy as np
    import pandas as pd
    destination = RESULTS / 'dataset_characterization'
    (destination / 'tables').mkdir(parents=True, exist_ok=True)
    (destination / 'figures').mkdir(exist_ok=True)
    for name in EDA_TABLES:
        path = eda / 'eda/tables' / f'{name}.csv'
        d = pd.read_csv(path)
        if any(c in d for c in ['participant_id', 'healthCode', 'recording_id', 'recordId', 'audio_path']):
            raise ValueError('Participant identifiers in aggregate table')
        shutil.copyfile(path, destination / 'tables' / path.name)
    for name in EDA_FIGURES:
        shutil.copyfile(eda / 'eda/figures' / f'{name}.png', destination / 'figures' / f'{name}.png')
    discovery = read_json(eda / 'eda/tables/discovery_summary.json')
    # Earlier EDA metadata retained stale workflow messages; retain explicitly as history.
    discovery['historical_blockers_at_metadata_audit'] = discovery.pop('blockers', [])
    discovery['current_scope_note'] = 'Full extraction completed: 64973/65022; historical pilot/runtime notes are superseded.'
    write_json(destination / 'discovery_summary.json', discovery)
    write_json(destination / 'demographic_imbalance.json', read_json(eda / 'eda/tables/demographic_imbalance.json'))
    duplicate = read_json(eda / 'eda/diagnostics/duplicate_summary.json')
    write_json(destination / 'duplicate_summary.json', duplicate)
    p = pd.read_parquet(eda / 'processed/participant_manifest.parquet',
                        columns=['diagnosis', 'age', 'recordings', 'followup_days'])
    p['age_band'] = pd.cut(p.age, [0, 18, 30, 40, 50, 60, 70, 80, 90, 120, np.inf]).astype(str)
    p.groupby(['diagnosis', 'age_band'], observed=True).size().rename('participants').reset_index().to_csv(
        destination / 'tables/participant_age_bins.csv', index=False)
    p['visit_band'] = pd.cut(p.recordings, [0, 1, 2, 5, 10, 25, 50, 100, 250, np.inf]).astype(str)
    p.groupby(['diagnosis', 'visit_band'], observed=True).size().rename('participants').reset_index().to_csv(
        destination / 'tables/participant_visit_bins.csv', index=False)
    q = pd.read_parquet(eda / 'processed/full_audio_quality_metrics.parquet')
    if 'duration_s' not in q:
        raise ValueError('Expected QC duration_s field')
    q['duration_band'] = pd.cut(q.duration_s, [0, 2, 5, 9, 10, 11, 15, 20, 30, 60, 180, np.inf]).astype(str)
    q.groupby('duration_band', observed=True).size().rename('recordings').reset_index().to_csv(
        destination / 'tables/duration_bins.csv', index=False)
    dictionary = pd.read_csv(eda / 'processed/feature_dictionary.csv')
    dictionary.to_csv(destination / 'tables/feature_dictionary.csv', index=False)
    # Aggregate schemas: field names/types only, no example participant values.
    schemas = read_json(eda / 'eda/diagnostics/table_schemas.json')
    canonical = [s for s in schemas if s.get('table') in
                 ['voice_activity/Voice_Activity.csv', 'demographics/Demographics.csv']]
    safe = {'canonical_tables': [{k: s[k] for k in ['table', 'rows', 'columns', 'duplicate_rows'] if k in s}
                                  for s in canonical],
            'total_audited_tables': len(schemas),
            'cached_query_and_download_shards': len(schemas) - len(canonical),
            'interpretation': 'Cache tables and download shards overlap the canonical export; never add their rows as observations.'}
    write_json(destination / 'table_inventory.json', safe)
    historical = RESULTS / 'historical_adaptation'
    historical.mkdir(exist_ok=True)
    rows, history, configs = [], [], {}
    for run in sorted((benchmark / 'adaptation_20261114/runs').iterdir()):
        if not (run / 'DONE.json').exists():
            raise ValueError('Incomplete historical production run')
        done, metrics = read_json(run / 'DONE.json'), read_json(run / 'validation_metrics.json')
        if done['test_labels_read'] or done['status'] != 'development_complete':
            raise ValueError('Invalid historical production run')
        h = read_json(run / 'history.json')
        best = max(h, key=lambda x: (x['auroc'], -x['log_loss']))
        rows.append({'run': run.name, 'requested_policy': done['requested_policy'],
                     'selected_phase': best['phase'], 'selected_epoch_zero_based': best['epoch'], **metrics})
        history += [{'run': run.name, **r} for r in h]
        source = read_json(run / 'provenance.json')
        configs[run.name] = {k: source[k] for k in ['snapshot_sha256', 'task', 'config', 'schedule', 'framework',
                                                  'code_sha256', 'frontend', 'batchnorm', 'sampling'] if k in source}
    pd.DataFrame(rows).to_csv(historical / 'metrics.csv', index=False)
    pd.DataFrame(history).to_csv(historical / 'learning_history.csv', index=False)
    write_json(historical / 'configurations.json', configs)
    selected_rows, search_rows = [], []
    frozen_root = benchmark / 'protocol_experiments/20261114T074847Z'
    # Bounded traversal of run artifacts only: never walk the large waveform/cache tree.
    for model in sorted((frozen_root / 'runs').iterdir()):
        for context in sorted(model.iterdir()):
            for arm in ['technical_primary', 'pitch_sensitivity']:
                for representation in sorted((context / arm).iterdir()):
                    for head in sorted(representation.iterdir()):
                        if not (head / 'search.json').is_file():
                            continue
                        search = read_json(head / 'search.json')
                        identity = {'model': model.name, 'context': context.name, 'arm': arm,
                                    'representation': representation.name.replace('__', '/'), 'head': head.name}
                        search_rows += [{**identity, **row} for row in search]
                        if head.name == 'logistic':
                            selected = sorted(search, key=lambda r: (-r['cv_auroc'], r['C']))[0]
                        else:
                            selected = max(search, key=lambda r: (r['auroc'], -r['log_loss']))
                        selected_rows.append({**identity, **selected})
    if len(selected_rows) != 364:
        raise ValueError('Expected 364 selected frozen fits')
    pd.DataFrame(selected_rows).to_csv(RESULTS / 'all_models/selected_hyperparameters.csv', index=False)
    pd.DataFrame(search_rows).to_csv(RESULTS / 'all_models/head_search_history.csv', index=False)
    hear_root = benchmark / 'hear_protocol/20261116T011601Z'
    panel = RESULTS / 'downstream_completed'
    panel.mkdir(exist_ok=True)
    shutil.copyfile(hear_root / 'downstream_panel/full/validation_results.csv', panel / 'selected_models.csv')
    write_json(panel / 'search_history.json', {p.stem: read_json(p)
                                             for p in sorted((hear_root / 'downstream_panel/full').glob('*_search.json'))})
    import joblib
    hear_head_parameters = {}
    for representation in ['mean', 'mean_sd']:
        for kind in ['logistic', 'mlp']:
            state = joblib.load(hear_root / 'runs/frozen/heads' / representation / kind / 'head.joblib')
            hear_head_parameters[f'{representation}/{kind}'] = (
                {'selected_C': float(state['head'].C), 'cv_folds': 3, 'C_grid': [.01, .1, 1.],
                 'search_history_saved': False} if kind == 'logistic' else
                {'input_dim': state['input_dim'], 'hidden_dim': state['hidden_dim'], 'epoch_budget': 10,
                 'selected_epoch': None,
                 'audit_gap': 'The original HeAR cached-head runner saved selected weights but did not persist epoch/search metadata.'})
    write_json(panel / 'hear_frozen_selected_hyperparameters.json', hear_head_parameters)
    obsolete = panel / 'hear_frozen_head_search.json'
    obsolete.unlink(missing_ok=True)
    tabular_lr = []
    tabular_root = benchmark / 'literature_methods/20261116T091246Z/tabular_runs'
    for encoder in ['hear', 'whisper_small', 'wavlm_large']:
        for budget in sorted((tabular_root / encoder).iterdir()):
            state = joblib.load(budget / 'lr.joblib')
            for row in state['search']:
                tabular_lr.append({'encoder': encoder, 'labelled_train_people': int(budget.name),
                                   'selected': row['C'] == state['head'].C, **row})
    pd.DataFrame(tabular_lr).to_csv(RESULTS / 'literature_tabular/lr_search_history.csv', index=False)
    # Refresh obsolete status table; preserve regime and source, rather than pretending matched protocols.
    frozen = read_json(RESULTS / 'frozen_development_summary.json')
    frontends = {}
    for name in [r['model'] for r in frozen['best_primary_per_model']] + ['hubert_base_ls960']:
        source = benchmark / 'models/checkpoints' / name / 'preprocessor_config.json'
        frontends[name] = read_json(source)
        frontends[name]['preprocessor_config_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        for key, value in list(frontends[name].items()):
            if isinstance(value, list) and len(value) > 30:
                frontends[name][key] = {'shape': list(np.asarray(value).shape),
                    'canonical_json_sha256': hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest(),
                    'note': 'Native checkpoint matrix retained in original asset; report summarizes rather than prints every constant.'}
    frontends['hear_pytorch'] = {
        'source': 'src/models/hear_vendor/audio_utils.py; pinned source revision in provenance appendix',
        'sample_rate': 16000, 'real_samples_per_window': 32000,
        'waveform_normalization': 'minmax to [-1,1] independently per cloned window; epsilon1e-8',
        'frame_length': 400, 'fft_length': 400, 'frame_step': 160, 'window': 'periodic Hann', 'pad_end': True,
        'spectral_power': 2, 'mel_bins': 128, 'mel_scale': 'HTK', 'mel_min_hz': 0, 'mel_max_hz': 8000,
        'PCEN': {'alpha': .8, 'smooth_coef': .04, 'delta': 2., 'root': 2., 'floor': 1e-8},
        'resize': 'TF-compatible bilinear to [1,192,128]; no antialias',
        'recording_pool': 'equal-window mean512D or mean+populationSD1024D'}
    write_json(RESULTS / 'native_frontends.json', frontends)
    status = [{'model': r['model'], 'regime': 'frozen_development_best_validation_selected',
               'status': 'completed', 'auroc': r['auroc']} for r in frozen['best_primary_per_model']]
    hear = pd.read_csv(RESULTS / 'hear_completed/metrics.csv')
    method_column = 'method' if 'method' in hear else hear.columns[0]
    status += [{'model': r[method_column], 'regime': 'HeAR_AST_downstream_controlled',
                'status': 'completed', 'auroc': r['auroc']} for r in hear.to_dict('records')]
    for r in pd.read_csv(RESULTS / 'literature_completed/metrics.csv').to_dict('records'):
        status.append({'model': r['method'], 'regime': 'literature_controlled', 'status': 'completed', 'auroc': r['auroc']})
    status += [{'model': r['run'], 'regime': 'historical_adaptation_confounded', 'status': 'completed',
                'auroc': r['auroc']} for r in rows]
    for r in pd.read_csv(RESULTS / 'literature_tabular/metrics.csv').to_dict('records'):
        status.append({'model': f"{r['encoder']}_{r['classifier']}_{r['labelled_train_people']}_labels",
                       'regime': 'initial_visit_PCA_tabular', 'status': 'completed', 'auroc': r['auroc']})
    for r in pd.read_csv(RESULTS / 'acoustic_xgboost/metrics.csv').to_dict('records'):
        status.append({'model': r['experiment'], 'regime': 'acoustic_GPU_tree_and_controls',
                       'status': 'completed', 'auroc': r['auroc']})
    status.append({'model': 'Whisper_acoustic_alignment', 'regime': 'paired_contrastive', 'status': 'prototype_not_trained', 'auroc': None})
    pd.DataFrame(status).to_csv(RESULTS / 'all_models/model_status.csv', index=False)


def markdown(text):
    import nbformat
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    import nbformat
    return nbformat.v4.new_code_cell(text.strip())


def document(relative):
    text = (ROOT / relative).read_text()
    # Resolve the source document's local links from the notebook's directory.
    base = (ROOT / relative).parent
    def replace(match):
        label, target = match.groups()
        if re.match(r'https?://|#|mailto:', target):
            return match.group(0)
        return f'[{label}]({os.path.relpath((base / target).resolve(), ROOT / "notebooks")})'
    return markdown(re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace, text))


SETUP = '''
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/mpower_preprocessing.yaml').exists())
RESULTS = REPO/'docs/results'
pd.set_option('display.max_colwidth', 120)
def table(relative):
    data = pd.read_csv(RESULTS/relative)
    with pd.option_context('display.max_rows', len(data)+1, 'display.max_columns', None):
        display(data)
    return data
def report(relative):
    display(Markdown('```json\\n'+json.dumps(json.loads((RESULTS/relative).read_text()),indent=2)+'\\n```'))
def figure(relative):
    display(Image(filename=str(RESULTS/relative)))
'''


def dataset_notebook():
    cells = [markdown('''
    # 1. mPower dataset: completed characterization and distributions

    **Evidence snapshot: 2026-11-16.** This notebook contains saved aggregate results from the completed full-data analysis,
    rather than the original 120-recording extraction pilot. It runs from a Git clone with no mPower credentials or audio.
    Every code cell only reads reviewed aggregates under `docs/results`; none launches preprocessing or training.

    Read in order: source/data dictionary → diagnosis/age → repeated visits → duration/QC → acoustic features → cohort/split audit.
    Notebook 2 covers the later supervised experiments. The original private analysis template is retained separately under
    `notebooks/private/`; rerunning it requires authorized HPC storage.

    These data were contributed by users of the Parkinson mPower mobile application as part of the mPower study developed
    by Sage Bionetworks and described in Synapse [doi:10.7303/syn4993293](https://doi.org/10.7303/syn4993293).
    '''), code(SETUP), markdown('''
    ## What succeeded and what failed

    Source census: **65,022 recordings / 5,826 participants**. The repaired full extraction succeeded on **64,973** recordings;
    **49** malformed/empty source objects remained undecodable. The earlier worker/Numba failures were implementation failures,
    repaired before this report; they do not invalidate the source census or age distributions. The source-file audit found
    the remaining 49 local objects matched Synapse sizes/checksums. They were retained as failures, never filled with fabricated audio.

    Diagnosis remains self-report of professional diagnosis, not an independent clinical assessment. The cleaned participant
    classification has **970 PD, 3,890 control, 864 unknown, 102 conflicting**. Unknown/conflicting people are excluded from binary modeling.
    Medication timing is never used as a diagnosis proxy. Full descriptive summaries precede supervised splitting and must not
    be mistaken for estimates from an independent prospective cohort.
    '''), markdown('''
    ## Storage tree and the files that encode the data

    ```text
    projects/
      pd_env/                              GitHub: portable code and aggregate reports
        notebooks/                         these two executed reports
        notebooks/private/                 private-analysis source template, output-free
        configs/                           preprocessing, native model and training settings
        scripts/; src/; tests/              executable procedures and synthetic verification
        docs/results/                      aggregate CSV/JSON/figures used in these notebooks
      pd_data/                             approved HPC data; not included in a Git clone
        mpower/                            MPOWER_DATA_ROOT
          voice_activity/Voice_Activity.csv 65,022 source observations; one recording per row
          voice_activity/metadata_parts/   download-query shards, not extra observations
          voice_activity/audio/            encoded sustained-phonation files; inspect codec
          demographics/Demographics.csv    separately linked baseline survey export
          synapse_cache/                   cached source objects; not a second dataset
        mpower_phase1/                     MPOWER_OUTPUT_ROOT
          processed/                       linked manifests, QC, acoustic features, A/B/C splits
          eda/{tables,figures,diagnostics}/ completed private characterization evidence
          archive/                         older/pilot snapshots; not current denominators
        mpower_benchmarks/                  MPOWER_BENCHMARK_ROOT
          models/; activations/            model assets and private derived embeddings
          protocol_experiments/            immutable development manifests and frozen runs
          adaptation_20261114/             historical training runs
          controlled_ast/; hear_protocol/  corrected controlled runs
          literature_methods/              LP-FT/LoRA/SupCon/TabICL experiments
          acoustic_xgboost/                GPU search database, fitted trees, private predictions
        assistant_samples/                 private 100-recording training-only handoff
    ```

    | File/field | Meaning and handling |
    |---|---|
    | `Voice_Activity.csv` | Original recording table, Synapse `syn5511444`; use the consolidated export once. |
    | `recordId` / `healthCode` | Recording and participant linkage keys; healthCode is a pseudonym, not anonymization. |
    | `audio_audio.m4a` | Synapse file-handle field pointing to sustained phonation. It is not numeric audio features. |
    | `audio_countdown.m4a` | Separate countdown handle; not the intended phonation signal. |
    | `audio_path` | Local source object path added by the downloader; resolve from approved storage. |
    | `createdOn`, `medTimepoint` | Acquisition time and self-reported medication timing; not clinical ON/OFF adjudication. |
    | `Demographics.csv` | Baseline table `syn5511429`: age, gender, diagnosis and other self-reported covariates. |
    | `participant_manifest.parquet` | One cleaned participant row with conservative diagnosis/conflict and demographic linkage. |
    | `recording_manifest.parquet` | One source recording row with participant link, task, dates, label and path. |
    | `full_audio_quality_metrics.parquet` | Duration/native-format/QC/hash/status per recording, including failed rows. |
    | `full_acoustic_features.parquet` | 184 numeric acoustic columns plus processing status; undefined measurements are missing. |
    | `feature_dictionary.csv` | Feature names, units and definitions; safe schema dictionary displayed below. |
    | `cohort_{A,B,C}/` | Private manifests/splits for labeled technical, pitch sensitivity and matched cohorts. |
    | `development_manifest.parquet` | Later immutable train/validation snapshot: source hashes, waveform links, component and initial-visit flags. |
    | `.m4a`, `.tmp`, extensionless files | The original filename suffix may not identify the actual media codec. Probe/decode content. |
    | `.npy`, `.npz`, `.pt`, `.joblib`, prediction CSVs | Derived waveforms, embeddings, weights, estimators and individual predictions; private. |

    The historical filesystem inventory includes cache/download artifacts and therefore is not a count of unique recordings.
    '''), code("report('dataset_characterization/table_inventory.json')\nreport('dataset_characterization/discovery_summary.json')"),
    markdown('''
    ## Diagnosis availability and demographic coverage

    Recording-level PD prevalence is much higher than participant-level PD prevalence because PD volunteers contribute more visits.
    Use the participant denominator for screening performance and demographic comparisons. Baseline demographics matched 5,044
    participants / 63,255 recordings; unmatched participants and missing fields remain explicit.
    '''), code("table('dataset_characterization/tables/cohort_characteristics.csv')\ntable('dataset_characterization/tables/demographic_coverage.csv')\nfigure('dataset_characterization/figures/diagnosis_availability.png')"),
    markdown('''
    ## Age, sex and confounding

    The completed participant summaries show median age **61 in PD versus 29 in controls**. Age and diagnosis are strongly entangled;
    this is a major explanation for the high age-only predictive baseline in Notebook 2. Reporting apparent voice separation
    without age/device controls would be misleading. Sex/gender is self-reported; missing/conflicting entries are retained.
    The table includes both participant and recording summaries: compare the `level` column before interpreting a row.
    The binned age table includes an explicit `nan` category for missing values.
    '''), code("table('dataset_characterization/tables/demographics_by_diagnosis.csv')\nreport('dataset_characterization/demographic_imbalance.json')\ntable('dataset_characterization/tables/participant_age_bins.csv')\nfigure('dataset_characterization/figures/demographic_distributions.png')"),
    markdown('''
    ## Repeated visits, follow-up and medication

    **4,299** participants contributed repeated recordings. Median visits/person is **3**, maximum **522**;
    the most active **1% contributed 30.4%** of all recordings. Treating rows as independent people would over-weight these volunteers
    and leak participant identity if visits were split independently. Later training either gives each person's visits total weight one
    or samples one visit/person/epoch. Validation uses one canonical initial eligible visit/person.

    Medication groups are descriptive. A report of before/after medication is not proof of a verified pharmacological ON/OFF pair.
    Baseline survey age can precede recordings; no longitudinal clinical severity trajectory is inferred.
    '''), code("table('dataset_characterization/tables/longitudinal_by_diagnosis.csv')\ntable('dataset_characterization/tables/participant_visit_bins.csv')\ntable('dataset_characterization/tables/medication_by_diagnosis.csv')\nfigure('dataset_characterization/figures/longitudinal_recordings.png')"),
    markdown('''
    ## Audio length and technical distributions

    These summaries use the full extraction scope, before the later encoder first-10-second cap. `n` and `missing` differ by
    measurement: use the row-specific denominator. No failed file is treated as a zero-duration recording. Histogram bins describe
    decoded/finite lengths; the `nan` bin retains missing durations. The extraction resource guard is 180 seconds.
    '''), code("table('dataset_characterization/tables/full_audio_summary.csv')\ntable('dataset_characterization/tables/duration_threshold_counts.csv')\ntable('dataset_characterization/tables/duration_bins.csv')\nfigure('dataset_characterization/figures/audio_duration.png')"),
    markdown('''
    ## QC warnings, retention and thresholds

    Near-silence, clipping, amplitude, energy activity, pitch coverage and decode failures are separately recorded. Aperiodicity,
    low amplitude and undefined perturbation values can themselves relate to dysphonia, so they are not blanket primary exclusions.
    The historical `estimated_snr_db` is an RMS-percentile energy-range proxy, **not measured environmental SNR**;
    its 10 dB flag is unsuitable as a validated noise gate or weight. QC threshold sweeps describe selection sensitivity, not an optimized diagnosis rule.
    '''), code("table('dataset_characterization/tables/quality_by_diagnosis.csv')\ntable('dataset_characterization/tables/qc_threshold_sensitivity.csv')\nfigure('dataset_characterization/figures/qc_distributions.png')"),
    document('docs/ACOUSTIC_FEATURES.md'), code("table('dataset_characterization/tables/feature_dictionary.csv')\ntable('dataset_characterization/tables/praat_opensmile_comparison.csv')"),
    markdown('''
    ## Full feature distributions, missingness and correlations

    The 184-column summary includes finite counts, missingness, mean/SD and quantiles. Praat full-clip and pitch-selected-interval
    measurements are different estimands. Whole-recording plots are descriptive; participant-median correlations reduce repeated-visit
    dominance but remain observational and can reflect age, device and phonation duration. Formants in eGeMAPS are algorithmic
    descriptors, not independently validated anatomical measurements. CPP and a separate Praat formant analysis were not implemented.
    '''), code("table('dataset_characterization/tables/full_feature_summary.csv')\nfigure('dataset_characterization/figures/acoustic_distributions.png')\nfigure('dataset_characterization/figures/selected_feature_correlations.png')"),
    markdown('''
    ## Exploratory PD associations

    Participant-median rank comparisons report rank-biserial effects and Benjamini–Hochberg FDR. Adjusted HC3 OLS summaries include
    the stated age/gender/quality covariates. These are descriptive association screens, not proof of disease-specific mechanisms,
    an externally validated biomarker or training-time feature selection. No feature is selected here using the sealed final-test endpoint.
    '''), code("table('dataset_characterization/tables/exploratory_pd_associations.csv')\ntable('dataset_characterization/tables/covariate_adjusted_pd_associations.csv')"),
    markdown('''
    ## Cohort flow and split readiness

    Legacy full extraction produced A: **62,123 recordings / 4,854 people**, B: **61,651 / 4,830**, C: **22,820 / 838**.
    A is the labeled decoded cohort; B adds pitch/phonation criteria; C matches within reported gender and five-year age bins,
    ages 50–80. These differ from the later technical development cohort, which excludes test people and requires at least two seconds.
    Split tables below summarize the original frozen assignments without exposing identities or individual test outcomes.
    Exact-duplicate-connected participants stay in one partition; near-duplicate/manual task-compliance review remains unresolved.
    '''), code("table('dataset_characterization/tables/qc_exclusion_flow.csv')\ntable('dataset_characterization/tables/cohort_demographic_balance.csv')\ntable('dataset_characterization/tables/age_gender_matching_flow.csv')\nfor name in ['A','B','C']:\n    table(f'dataset_characterization/tables/cohort_{name}_splits.csv')\n    table(f'dataset_characterization/tables/cohort_{name}_split_demographics.csv')\nreport('dataset_characterization/duplicate_summary.json')"),
    markdown('''
    ## Extraction failure audit and exact preprocessing methods

    The following source-audited methods and failure report document implementation, diagnosis linkage and what was repaired.
    Historical pilot counts are explicitly labeled and must not replace the full-scope results above.
    '''), document('docs/MPOWER_PREPROCESSING_METHODS.md'), document('docs/FAILED_AUDIO_AUDIT.md'),
    markdown('## Exact extraction configuration'), markdown('```yaml\n'+(ROOT/'configs/mpower_preprocessing.yaml').read_text()+'\n```'),
    markdown('''
    ## Limits and lab handoff

    The primary analysis is observational, self-selected and demographically imbalanced; it does not establish diagnosis in a clinical setting.
    Near-duplicate review, manual phonation compliance, age/device common-support evaluation and independent final testing remain necessary.
    Real recordings stay in approved storage; `samples/mpower_100/README.md` explains the prepared private training sample and access procedure.

    Dataset reference: Bot et al., *The mPower study, Parkinson disease mobile data collected using ResearchKit*, Scientific Data (2016),
    [doi:10.1038/sdata.2016.11](https://doi.org/10.1038/sdata.2016.11).
    Acoustic descriptor reference: Eyben et al., *The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing*,
    [doi:10.1109/TAFFC.2015.2457417](https://doi.org/10.1109/TAFFC.2015.2457417).
    ''')]
    return cells


def models_notebook():
    cells = [markdown('''
    # 2. All model results, exact procedures and literature rationale

    **Completed-results snapshot: 2026-11-16.** All numbers here are **development validation**, not final-test performance.
    The immutable development manifest is `35e5a70030b4bc116a8f11854119f80c0d70f2d059171931269e8c065988c53b`.
    It includes **53,122 recordings / 4,126 people**: train **43,702 visits / 3,397 people**, validation **9,420 visits / 729 people**.
    Primary scoring uses **729 canonical initial eligible validation visits: 146 PD, 583 controls**.

    This notebook includes all 364 frozen pipelines, the 14 historical adaptation runs, corrected AST and HeAR controls,
    downstream/fusion panels, all 18 TabICL comparisons, five completed literature training arms, and all seven XGBoost/control fits.
    Smoke tests and cancelled partial runs are not included as completed scientific comparisons.

    Read: evidence map → common QC and native model inputs → frozen heads → original and corrected tuning → HeAR/LoRA/SupCon
    → downstream and acoustic search → uncertainty/failures → exact configuration appendix → papers.
    All cells run from GitHub aggregates without audio/model downloads or GPU training.
    '''), code(SETUP), markdown('''
    ## Results at a glance, with comparison boundaries

    | Experiment | Validation AUROC | Interpretation |
    |---|---:|---|
    | Whisper Large-v3, best frozen pipeline | 0.8951 | Selected across representations/contexts/heads; optimistic development maximum. |
    | HeAR converged-LP → full fine-tuning | 0.8940 | New matched literature arm, selected epoch 13/20. |
    | HeAR converged-LP → Q/V LoRA | 0.8908 | New matched literature arm, selected epoch 18/20. |
    | HeAR converged-LP → head-only | 0.8721 | Matched initialization/schedule control. |
    | WavLM Large + TabICLv2/PCA128 | 0.8811 | Initial-training-visit classifier experiment. |
    | HeAR frozen mean + standardized LR | 0.8700 | Strong earlier repeated-visit probe; starting point for LP-FT. |
    | Earlier HeAR full tuning from weak warm-up | 0.8701 | Different head initialization from the converged-LP runs. |
    | HuBERT Base CE-only | 0.8604 | 100-epoch, actual balanced-person batch control. |
    | HuBERT Base detached CE + SupCon | 0.8527 | Did not improve over matched CE-only. |
    | Acoustic XGBoost, initial training visits | 0.8169 | Training-only grouped CV/TPE; acoustic LR reference 0.8210. |
    | Acoustic XGBoost, equal-person repeated visits | 0.8380 | More training exposure; fixed primary-selected hyperparameters. |
    | Age-only / age+device | 0.9330 / 0.9337 | Confounding controls; not acoustic biomarkers. |

    These rows are an evidence map, not a fair single leaderboard: representation, training visit exposure and selection budgets differ.
    Use the matched controls and paired intervals below to evaluate a specific modification. AP is average precision, not trapezoidal PR area.
    '''), code("report('frozen_development_summary.json')"),
    markdown('''
    ## A. Data quality, eligibility and leakage controls

    1. Join baseline demographics to source voice activity using audited participant keys; preserve unknown/conflicting labels.
    2. Use the sustained-phonation source column, never countdown audio. Require successful finite decode and at least two real seconds.
    3. Reuse immutable participant/duplicate-component train/validation assignments. Reject test rows before reading labels.
    4. Native-channel QC records RMS/peak/DC, clipping runs, zero/flat runs and possible downmix cancellation. It does not certify clinical quality.
    5. Derive mono float32 16 kHz PCM once from native decode. No denoising, pre-emphasis, amplitude normalization or adaptive trimming in the shared protocol.
       Checkpoint-native frontends may subsequently apply their required normalization, including HeAR's per-window min/max operation.
    6. Keep the first min(N,160,000) real samples. Matched-2s windows use `ceil(N/32,000)` rounded, endpoint-anchored linspace starts.
       The union covers the entire span; windows may overlap. Mean pooled windows receive equal weights.
    7. Do not exclude aperiodicity, low amplitude, jitter/HNR failure or abnormal pitch from the technical primary cohort.
       Pitch-detected ≥2 seconds is a separate sensitivity arm. Legacy energy-range pseudo-SNR is not used as a weight.
    8. Fit all scaling, imputation/PCA, classifiers and sample-dependent transforms on training data only; re-fit inside CV folds.

    Technical quality weights/Huber aggregation, participant-day weighting, automatic task recognition and near-duplicate release review
    were proposed separately and are **not** fitted production components. Equal-person visit weighting is implemented and should not
    be confused with the proposed technical reliability weights.
    '''),
    markdown('''
    ## B. Fitting data to each pretrained model

    | Family | Native input and readout | Important constraints |
    |---|---|---|
    | AST AudioSet | Pinned HF extractor: native 128-bin filterbank/normalization and checkpoint frame padding; native pooler or separately named patch statistics. | Native AST path is not resized to 224×224. Padded attention can still influence real tokens. |
    | Whisper Small/Medium/Large-v3 | Native checkpoint log-Mel and 30s frontend context; encoder only, last/middle block mean or mean+population SD. | No tokenizer/ASR decoding; waveform normalization explicitly disabled. Checkpoint-specific Mel dimensions retained. |
    | wav2vec2, WavLM, HuBERT Large | Native checkpoint waveform extractor/normalization and applicable masks; last/middle hidden states or normalized convolution tap. | Preserve checkpoint-specific group/layer normalization and mask conventions; population SD; no invented token counts. |
    | HeAR PyTorch | Exactly 32,000 real mono16k samples/window; vendored pinned Google frontend applied independently to cloned windows. 512D native pooler; window mean or mean+SD. | Frontend extrema span its input batch and can mutate input: independent preprocessing prevents cross-recording coupling. Pool windows inside the gradient graph. |
    | HuBERT Base literature arms | Separate 768D `facebook/hubert-base-ls960`; first ≤5 real seconds; native waveform extractor and temporal mean pooling. | CNN frozen; loaded native masking tensor before disabling mask-time/mask-feature/layerdrop on actual model config. |
    | ResNet50 historical control | 128-bin power Mel, n_fft/win_length400 (25ms), hop160 (10ms), Hamming; dB clip[-80,0]→[0,1]; bilinear224², three channels, ImageNet mean/SD. | ImageNet1K-V2 weights; BatchNorm running statistics frozen. This is the image model path, not AST preprocessing. |
    | Acoustic XGBoost/LR | 22 whole-clip Praat +22 pitched-interval Praat +88 eGeMAPSv02 +52 spectral descriptors. | Existing full-clip acoustic extraction is not recomputed at the encoder 10s cap; native tree missing branches vs train-fit LR imputation. |

    Frozen native-inference code: `src/models/frozen.py`; HeAR: `src/models/hear.py` and vendored frontend provenance;
    differentiable encoders: `src/training/adaptation_runner.py` and `src/models/literature_adapter.py`.
    Model revisions and source asset checksums are included in the configuration appendix; public weights remain outside Git.
    '''),
    markdown('''
    ## C. Frozen representation and downstream head sweep: all 364 results

    Nine checkpoints × full/matched-2s contexts, multiple named feature taps and mean/statistical pooling, LR/MLP,
    plus a separately labeled pitch-retention sensitivity arm. There are **182 technical-primary and 182 sensitivity** pipelines.
    The sensitivity endpoint includes **725 validation people**, compared with **729 in technical-primary**; do not pair their scores as identical cohorts.
    Encoder state hashes were unchanged; zero extraction failures were permitted before fitting common-cohort heads.

    LR: inverse-visit weights total one/person in scaler and loss; C∈{.01,.1,1}, training-only 3-fold StratifiedGroupKFold by duplicate component;
    max_iter1000. MLP: weighted train scaler; input→64/ReLU→1, AdamW LR.001, weight_decay.0001, batch256 people, ten epochs,
    one randomly selected visit/person/epoch; validation AUROC then log loss selects epoch. Native common seed20261112.
    Each pipeline's validation score is displayed; the best-per-model ranking additionally selects over contexts/representations/heads.
    The originally reported specificity in this older table is .898799 for many rows because of float32 threshold storage;
    later analyses use float64 nextafter and report the actual specificity. Do not treat every row as exactly90%.
    '''), code("all_frozen = table('all_models/all_364_development_pipelines.csv')\nassert len(all_frozen) == 364\nfigure('all_models/all_models_validation.png')"),
    markdown('### Actual selected regularization and neural epochs\n\nEach row below maps to one frozen pipeline above. The complete training-CV and MLP epoch history is also saved in `docs/results/all_models/head_search_history.csv`.'),
    code("table('all_models/selected_hyperparameters.csv')"),
    markdown('''
    ## D. Historical adaptation batch: complete, but confounded

    These 14 runs are retained to explain what actually happened. Each ran ten head warm-up plus20 adaptation epochs, batch32 distinct
    people via singleton accumulation, one random visit/person/epoch, BCE, gradient norm clip1, seed20261112.
    AST/ResNet use SGD momentum.9/Nesterov weight_decay.01; waveform arms AdamW weight_decay.01.
    Warm-up head LR.001. Adapted AST backbone/head start1e-5; ResNet1e-4; waveform backbone5e-6 and head1e-4.
    Adaptation cosine endpoint1e-6. Frozen AST/ResNet head branches retained the warm-up LR behavior, confounding the comparison.

    Adaptation began from the last warm-up state rather than its best checkpoint. The global selected model could still be from
    warm-up. The table therefore explicitly reports **requested policy and selected phase**; requested unfreezing alone does not
    establish that the chosen checkpoint benefited from unfreezing. These are actual original-weight updates, not LoRA.
    Historical late-resume support was incomplete and is superseded by the corrected runner.
    '''), code("table('historical_adaptation/metrics.csv')"),
    markdown('''
    ## E. Corrected paired AST controls

    A single best ten-epoch warm-up initialized all three branches; native LayerNorm+Linear head, SGD momentum.9/Nesterov,
    weight_decay.01, batch32 (microbatch4), clip1. All branches used the same20-epoch head cosine1e-3→1e-5;
    last2/full backbone cosine1e-5→1e-6. Optimizers reset identically at branching. Test rows refused; full RNG/optimizer resume
    and production frozen/trainable parameter audits implemented.

    **Head-only0.8715, last2 0.8671, full0.8647.** Fine-tuning did not improve this matched AST control.
    Tougui et al.'s paper compared from-scratch AST with pretrained+fine-tuned AST, not a final frozen-pretrained control;
    its repeated-visit endpoint/cohort/five-seed evaluation differ from this initial-assessment experiment.
    '''),
    markdown('''
    ## F. Earlier HeAR frozen and weak-warm-up tuning, plus downstream panels

    The four cached frozen heads use native512D mean or1024D mean+SD window features and the same LR/MLP development fitting above.
    The earlier trainable run used LayerNorm+Linear initialized from a ten-epoch AdamW warm-up at1e-4. Its warm-up AUROC was only.6866.
    Three20-epoch branches shared that warm-up: head cosine1e-4→1e-5; adapted backbone5e-6→1e-6, weight_decay.01,
    effectivebatch32/microbatch1, clip1, float32. Head-only.7229, last2 .8012, full.8701. These controls show improvement over
    a weak head, but full tuning merely matched the much stronger standardized frozen LR probe. The next section fixes that initialization.

    Downstream panel: canonical initial training visit/person, train-fold median imputation/StandardScaler; LR C∈{.01,.1,1,10}
    crossed with PCA∈{none,32,128}; LinearSVC C∈{.01,.1,1}, no PCA. Training-only grouped3-fold CV selects heads.
    SVM probabilities use a training-OOF logistic calibrator. Early fusion concatenates encoder features then fits the same pipeline.
    Equal average uses fixed LR C.1; stacking trains LR C.1 on training-OOF predictions, never validation-label fit.
    Age/device controls fit imputation/encoding on training only. Acoustic PCA alignment remains a prototype, not a completed contrastive model.

    The following full table includes earlier HeAR/AST metrics, every completed fusion and demographic control. Its summary
    includes paired intervals and parameter audits. Thresholds target ≥90% specificity **on validation**, so operating points
    are exploratory and not independently locked test performance.
    The original HeAR cached-head runner saved selected weights but did not persist the LR CV history or selected MLP epoch.
    Actual LR C values are recovered from the saved estimators below; the missing MLP epoch is reported as unavailable, never inferred.
    Later frozen-sweep and literature runners retain the full histories shown elsewhere in this notebook.
    '''), code("table('hear_completed/metrics.csv')\nreport('hear_completed/summary.json')\nfigure('hear_completed/learning_and_roc.png')\ntable('downstream_completed/selected_models.csv')\nreport('downstream_completed/search_history.json')\nreport('downstream_completed/hear_frozen_selected_hyperparameters.json')"),
    markdown('''
    ## G. Literature-inspired strong initialization: LP → full tuning and LoRA

    All three arms begin from exactly the same converged train-fitted HeAR standardized logistic probe, AUROC.8700.
    Fixed train mean/scale are preserved as buffers; real-waveform initial probabilities were checked against sklearn.
    This removes the poor random-head warm-up as the starting point. The initial LP is an explicit epoch-zero selection candidate.

    All arms: AdamW betas(.9,.999), fixed head LR1e-4, weight_decay.01,20epochs, batch32/microbatch1, one visit/person/epoch,
    no AMP/TF32, clip1, seed20261112. Full tuning: backboneLR5e-6,303,199,744 trainable backbone parameters.
    LoRA: original weights frozen, Q/V adapters in all24 blocks, rank8/alpha16/dropout0, adapterLR1e-4,
    786,432 trainable adapter parameters (~.26% of original backbone). Update `W+(alpha/r)BA`; zeroB/randomA gives initially identical logits.
    Non-reentrant encoder checkpointing is used for full/LoRA. No additional cosine schedule or early-stop patience is invented;
    all20epochs complete, with epoch-zero/epoch AUROC then log-loss checkpoint selection.

    Selected: head-only epoch18 **.8721**; full epoch13 **.8940**; LoRA epoch18 **.8908**.
    These are the first strong matched fine-tuning gains here. Parameter audits confirm full changed original weights and LoRA
    changed adapters while original pretrained tensors stayed identical. They are one-seed validation findings, not final-test gains.

    Inspiration: Kumar et al. LP-FT (ICLR2022), Hu et al. LoRA (ICLR2022); hyperparameters transferred to HeAR are project hypotheses,
    not a claim that the source papers established optimal settings for mPower.
    '''),
    markdown('''
    ## H. HuBERT Base: precisely what is contrasted

    Two matched100-epoch arms use actual batches of up to32 distinct people, class-balanced sampling and one selected training visit/person.
    Each epoch uses all679 PD people and a seeded without-replacement subset of679 controls; the majority subset varies across epochs.
    Thus each epoch trains on1,358 people, rather than all3,397; the two arms use the same schedule.
    The final partial batch can be smaller (observed minimum14). CE classifier losses are weighted by `2*PD_prior` / `2*(1-PD_prior)`
    to recover the original training-person prevalence (~.19988). Native768D temporal mean→64/ReLU/dropout.2→2 classifier.
    AdamW2e-5 for backbone/head/projector, betas(.9,.999), weight_decay.01, fixedLR, clip1, frozenCNN, mask-time/feature/layerdrop0.

    In the contrastive arm, **positive pairs share diagnosis across different participants**; self, same-person and duplicate-component
    pairs are masked. Different-diagnosis eligible people supply negatives. A declared single-linear32D projector with normalized vectors
    and temperature.07 supplies SupCon. CE sees detached encoder features, so CE updates the classifier while SupCon updates encoder/projector;
    objective coefficient1 for each term. The actual contrastive batch graph is retained via checkpointing. Accumulation across separate
    batches is not substituted for a larger denominator. In the CE control, CE updates the encoder directly.

    **CE .8604 (epoch33); detached SupCon .8527 (epoch37).** The proposed contrastive mechanism did not improve AUROC here.
    This adapts a NeuroVoz repeated-sentence method to sustained vowels. The source's hard-negative formula/projector depth and filtering
    were underspecified; no formula/filter was fabricated. Project choices and deviations are explicit, so this is not exact numerical SOTA reproduction.
    '''), code("table('literature_completed/metrics.csv')\nreport('literature_completed/summary.json')"),
    code("history = pd.read_csv(RESULTS/'literature_completed/learning_history.csv')\nfig, axes = plt.subplots(1, 2, figsize=(13,4))\nfor method, group in history.groupby('method', sort=False):\n    axes[0 if method.startswith('hear') else 1].plot(group.epoch, group.auroc, label=method)\nfor ax in axes:\n    ax.set(xlabel='Epoch', ylabel='Initial-visit validation AUROC')\n    ax.legend(fontsize=8); ax.grid(alpha=.2)\nplt.tight_layout(); plt.show()"),
    markdown('''
    ## I. Adjacent small-data classification: all 18 TabICL comparisons

    Frozen HeAR/WhisperSmall/WavLMLarge embeddings, one initial training visit/person, StandardScaler+randomizedPCA fit on labeled context.
    Full context3,397 people usesPCA128. HeAR label budgets50/100/500 usePCA16/33/128, respectively; fixedseed20261112,
    independently sampled stratified contexts, not nested subsets. LR C∈{.001,.01,.1,1}, max_iter2000, three training-only groupedfolds
    refitting scaler/PCA. TabICL2025 v1 uses32 estimators; TabICLv2 2026 uses8; library2.2.0, local checkpoint path, no auto-download,
    no remote inference, no AMP/FA3, batch_size2/n_jobs4. No participant data are used to pretrain either foundation model.

    Full-label WavLM: LR.8718, v1 .8743, v2 .8811. The paired v2−LR interval includes zero.
    Compare classifiers **within** a budget; changing PCA dimension and one-seed context limits interpretation of the label-efficiency curve.
    TabPFN was reviewed but not run; TabICL's explicit pinned BSD checkpoint was chosen for these controlled comparisons.
    '''), code("table('literature_tabular/metrics.csv')\nreport('literature_tabular/summary.json')\ntable('literature_tabular/lr_search_history.csv')"),
    markdown('''
    ## J. GPU acoustic XGBoost: preprocessing, hyperparameter tree search and all diagnostics

    All184 acoustic features, no participant/demographic/device/medication predictors. Source hashes matched the immutable manifest.
    Primarytraining3,397 initial visits; nativeNaN branches, infinities→NaN; reject all-missing rows; no scaler/imputer/PCA.
    Training-only3-fold StratifiedGroupKFold by participant/duplicate component;60 seeded multivariateTPE trials with12startup trials;
    serial trials on oneGPU with4CPU helperthreads. Each CV fit permits2,500 rounds and earlystop75 on training-CV AUROC only.
    Outer validation is never supplied to the search. Trial40 selected823trees, depth4, eta.0442857; all exact values below.

    Tuned depth2–8, min_child_weight1–64log, eta.005–.2log, subsample.6–1, colsample.5–1,
    lambda.001–100log, alpha1e-8–10log, gamma0–5, max_bin∈{128,256,512}. No class weighting.
    Feature-family ablations and equal-person repeated-visit sensitivity reuse primary-selected settings; no extra validation search.
    The repeatedfit uses43,702 visits with weight1/number_of_person_visits. LRreference: training-fold median imputer+missing indicators,
    StandardScaler, C∈{.001,.01,.1,1}, groupedCV; selectedC1.

    Official GPUXGBoost3.4.1 verified `cuda:0`/GPUhist at runtime, native model reload, SHAPcontribution sum to rawmargin.
    Optuna4.6.0 SQLite/sampler checkpoints support deterministic resume. Smokejob27619782 and searchjob27619783 completed;
    production search/diagnostics elapsed4m26s. Trees and per-person SHAP/predictions remain private.
    Acoustic features use fullclips, unlike encoder10scap; the repeatedfit also uses more visits than the initialfit.
    '''), code("table('acoustic_xgboost/metrics.csv')\nreport('acoustic_xgboost/summary.json')\nfigure('acoustic_xgboost/search_and_importance.png')\ntable('acoustic_xgboost/search_trials.csv')\ntable('acoustic_xgboost/feature_importance.csv')"),
    markdown('''
    ## K. Interpretation, uncertainty and what remains unfinished

    HeAR LP-FT and LoRA improve the matched head-only control on this validation cohort; detached HuBERT SupCon does not.
    Acoustic initialvisitXGBoost does not beat matchedLR; repeatedvisitXGBoost is higher but changes training exposure and its pairedinterval includeszero.
    Age/device strongly predict labels. Audio model scores alone cannot establish added disease-specific value.

    Bootstrap intervals resample validation duplicate-connected components (here each of729 components is one person),2,000replicates,
    seed20261116. They condition on fitted/selected models; they do not correct repeated validation selection, epoch selection,
    hyperparameter search, training-seed variability or multiple comparisons. AUROC is ranking, AP depends on prevalence,
    logloss/Brier assess probability quality, and sensitivity depends on the development-selected operating threshold.
    Missingness-only XGBoost has tied predictions and achieved98.8% specificity, so its sensitivity is not an exact90%-specificity comparison.

    No independent final-test results are released. No five-seed replication has been completed.
    Near-duplicate/task review, demographic-common-support and device robustness, multiseed checks and a locked final protocol remain required.
    Whisper/acoustic paired alignment and proposed quality/medication weighting are documented prototypes, not trained winners.
    No unsupported claim of reproducing another cohort's clinical SOTA is made.
    '''),
    markdown('''
    ## L. Execution failures and their resolution

    | Event | Cause / outcome |
    |---|---|
    | Initial full acoustic run timed out | Worker/Librosa-Numba zero-crossing path failed; repaired deterministic NumPy ZCR and bounded CPU parallelism; completed full retry. |
    | Remaining49 source decode failures | Matched original Synapse object sizes/checksums; normal/error-tolerant decoding produced no samples. Preserved failures. |
    | Initial pretrained downloads | Historical networking/cache failures; later pinned downloads and offlineGPU smoke completed. |
    | HeAR access403 | Model-gate access not granted initially; authorized download and hash verification later completed; all current HeAR runs finished. |
    | Original AST comparison | HeadLR and warm-up checkpoint confounds; corrected paired runner and separate phase labels. |
    | First HuBERT literature load | Disabling masking before nativeconstruction hid a tensor; fixed fullnative load, then disabled augmentation on actual copiedconfig. Cancelled partial run excluded. |
    | CPU-onlyXGBoost build | Replaced with officialGPU3.4.1 wheel in privateenv; rejectedCPU fallback in production. |
    | Optuna/SQLite import ABI | Child-process-only libstdc++ preload wrapper; no global system changes. |

    Relevant synthetic checks verify split refusal, transforms/weights, native batching/padding, LoRA identity/base invariance,
    LP prediction equivalence, SupCon gradient routes, exactresume, GPUfallback rejection and searchresumption. Passing software checks
    establishes execution correctness for their scope, not clinical validity.
    '''),
    markdown('''
    ## M. Exact configuration and provenance appendix

    The checked-in YAML below records declared schedules/search spaces; aggregate run provenance records actual selected epochs,
    parameter-update audits and versions. Historical design-only YAML is explicitly marked and does not prove execution.
    Native checkpoint locks retain revisions and asset hashes. The completed snapshot remains immutable even if working templates change.
    ''')]
    for relative in ['configs/mpower_preprocessing.yaml', 'configs/frozen_model_testing.yaml', 'configs/adaptation_run.yaml',
                     'configs/ast_controlled_finetuning.yaml', 'configs/hear_protocol.yaml',
                     'configs/literature_benchmarks.yaml', 'configs/acoustic_xgboost.yaml']:
        cells += [markdown(f'### `{relative}`\n\n```yaml\n'+(ROOT/relative).read_text()+'\n```')]
    cells += [code("report('native_frontends.json')\nreport('literature_completed/provenance.json')\nreport('historical_adaptation/configurations.json')")]
    for relative in ['configs/pretrained_models.lock.json', 'configs/literature_models.lock.json', 'configs/tabicl_models.lock.json',
                     'src/models/hear_vendor/provenance.json']:
        cells += [markdown(f'### `{relative}`\n\n```json\n'+(ROOT/relative).read_text()+'\n```')]
    cells += [markdown('''
    ## N. Papers: what was implemented, adapted or only considered

    The single requested literature agent reviewed13 fullPDFs; PDFhashes, extraction lengths and page references are retained in
    method cards below. Fulltext/PDFs stay in private research storage. Method summaries are original paraphrases.
    Implemented: LP-FT initialization, LoRA Q/V mechanism, unscaled detached-classifier SupCon adaptation, TabICL v1/v2.
    Reviewed inspiration only: LungAdapter, device-adversarialSG-SCL, dysarthricASR/AdaLoRA, VoiceFM, WavLMemotion and TabPFN.
    Different endpoints/labels/datasets mean their reported scores are not mPower targets or directly comparable AUROCs.
    '''), document('docs/literature/REVIEW.md')]
    cards = read_json(ROOT / 'docs/literature/method_cards.json')
    for card in cards['papers'] if 'papers' in cards else cards['methods']:
        cells += [markdown(f"### {card['title']} ({card['year']})\n\n[Primary PDF]({card['primary_pdf']['url']})\n\n"
                          '```json\n'+json.dumps(card, indent=2)+'\n```')]
    cells += [markdown('''
    ### Foundational dataset, encoders, features and tree boosting

    | Source | Role in this project |
    |---|---|
    | [Bot et al., mPower,2016](https://doi.org/10.1038/sdata.2016.11) | Dataset design and access context; not a model training recipe. |
    | [HeAR,2024](https://arxiv.org/abs/2403.02522) / [official checkpoint](https://huggingface.co/google/hear-pytorch) | Health-audio pretrained encoder and pinned nativefrontend. |
    | [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) | Native AST architecture, AudioSet initialization and pooling. |
    | [Whisper](https://arxiv.org/abs/2212.04356) | Pretrainedencoder representations; no transcript/decoder objective used here. |
    | [wav2vec2](https://arxiv.org/abs/2006.11477), [HuBERT](https://arxiv.org/abs/2106.07447), [WavLM](https://arxiv.org/abs/2110.13900) | Pretrained speech backbone families; local revisions pinned. |
    | [Eyben et al.,GeMAPS](https://doi.org/10.1109/TAFFC.2015.2457417) | Interpretable acoustic descriptor basis; eGeMAPSv02 native implementation used. |
    | [Chen &Guestrin,XGBoost](https://arxiv.org/abs/1603.02754) / [GPUimplementation](https://xgboost.readthedocs.io/en/stable/gpu/index.html) | Boosted-tree classifier; GPUhist is the installed library implementation, not a rewritten tree engine. |

    To refresh these reports: first export/verify private completed runs on HPC, then rebuild/execute using
    `scripts/build_results_notebooks.py`. A clone can regenerate the two notebooks from checked-in aggregates alone.
    Training uses the separate documented Slurm runners; reading a report never submits jobs.
    ''')]
    return cells


def build(execute=True):
    import nbformat
    from nbclient import NotebookClient
    destinations = [('01_mPower_Dataset_Characterization.ipynb', dataset_notebook()),
                    ('02_All_Models_Methods_and_Results.ipynb', models_notebook())]
    for name, cells in destinations:
        nb = nbformat.v4.new_notebook(cells=cells, metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'evidence_snapshot': '2026-11-16', 'aggregate_only': True, 'final_test': 'sealed'})
        if execute:
            # Use the calling environment, not whichever global python3 kernel happens to exist.
            import sys
            from tempfile import TemporaryDirectory
            from jupyter_client.kernelspec import KernelSpecManager
            with TemporaryDirectory(prefix='mpower-report-kernel-') as kernel_root:
                directory = Path(kernel_root) / 'mpower-report'
                directory.mkdir()
                write_json(directory / 'kernel.json', {'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                           'display_name': 'mPower aggregate reports', 'language': 'python'})
                manager = KernelSpecManager(kernel_dirs=[kernel_root])
                from jupyter_client.manager import KernelManager
                km = KernelManager(kernel_name='mpower-report', kernel_spec_manager=manager)
                try:
                    NotebookClient(nb, timeout=180, kernel_name='mpower-report', km=km,
                                   resources={'metadata': {'path': str(ROOT)}}).execute()
                finally:
                    if km.has_kernel:
                        km.shutdown_kernel(now=True)
        # No host paths/kernel timing metadata in the published notebook.
        nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
        for i, cell in enumerate(nb.cells):
            cell['id'] = hashlib.sha256((name + str(i) + cell.source).encode()).hexdigest()[:12]
            cell.metadata.pop('execution', None)
        nbformat.validate(nb)
        nbformat.write(nb, ROOT / 'notebooks' / name)
        print(f'{name}: {len(cells)} cells; executed={execute}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--refresh-private-aggregates', action='store_true')
    parser.add_argument('--eda-root', type=Path, default=os.environ.get('MPOWER_OUTPUT_ROOT'))
    parser.add_argument('--benchmark-root', type=Path, default=os.environ.get('MPOWER_BENCHMARK_ROOT'))
    parser.add_argument('--no-execute', action='store_true')
    a = parser.parse_args()
    if a.refresh_private_aggregates:
        if not a.eda_root or not a.benchmark_root:
            parser.error('Explicit private EDA/benchmark roots are required for refreshing aggregates')
        refresh(a.eda_root, a.benchmark_root)
    build(execute=not a.no_execute)
