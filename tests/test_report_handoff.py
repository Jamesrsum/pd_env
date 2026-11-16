"""Publication and training-sample boundaries for the completed lab handoff."""
import importlib.util
import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_reports_are_executed_and_use_only_portable_aggregates():
    for name in ['01_mPower_Dataset_Characterization.ipynb', '02_All_Models_Methods_and_Results.ipynb']:
        nb = nbformat.read(ROOT / 'notebooks' / name, as_version=4)
        nbformat.validate(nb)
        cells = [c for c in nb.cells if c.cell_type == 'code']
        assert cells and all(c.execution_count for c in cells)
        assert all(o.output_type != 'error' for c in cells for o in c.outputs)
        assert sum(bool(c.outputs) for c in cells) >= len(cells) - 1
        code = '\n'.join(c.source for c in cells)
        assert not any(value in code for value in ['read_parquet', 'np.load(', 'torch.load(', 'os.environ', 'sbatch'])
        assert nb.metadata.aggregate_only and nb.metadata.final_test == 'sealed'


def test_result_counts_and_frozen_parameter_mapping():
    result = ROOT / 'docs/results'
    frozen = pd.read_csv(result / 'all_models/all_364_development_pipelines.csv')
    parameters = pd.read_csv(result / 'all_models/selected_hyperparameters.csv')
    keys = ['model', 'context', 'arm', 'representation', 'head']
    assert len(frozen) == len(parameters) == 364
    assert len(frozen.merge(parameters, on=keys, validate='one_to_one')) == 364
    latest = pd.read_csv(result / 'literature_completed/metrics.csv')
    assert len(latest) == 5 and latest.participants.eq(729).all()
    summary = json.loads((result / 'literature_completed/summary.json').read_text())
    assert not summary['test_labels_read']
    assert summary['training_audits']['hear_lp_lora']['pretrained_weights_changed'] is False
    assert summary['training_audits']['hear_lp_full']['pretrained_weights_changed'] is True


def sample_module():
    spec = importlib.util.spec_from_file_location('assistant_sample', ROOT / 'scripts/prepare_assistant_sample.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_refuses_git_and_existing_source_destinations(tmp_path):
    module = sample_module()
    with pytest.raises(ValueError, match='outside Git'):
        module.prepare(tmp_path / 'snapshot', ROOT / 'samples/unsafe', tmp_path / 'data')
    with pytest.raises(ValueError, match='outside Git'):
        module.prepare(tmp_path / 'snapshot', tmp_path, tmp_path / 'data')


def test_sample_uses_training_people_and_detects_tampered_source(tmp_path, monkeypatch):
    import hashlib
    module = sample_module()
    data = tmp_path / 'private/mpower'
    data.mkdir(parents=True)
    rows, sources = [], []
    for label in [0, 1]:
        for i in range(51):
            rid = f'{label}_{i}'
            path = data / f'{rid}.m4a'
            path.write_bytes(f'synthetic fixture {rid}'.encode())
            rows.append({'recording_id': rid, 'participant_id': rid, 'duplicate_component': rid,
                         'label': label, 'split': 'train', 'initial_recording': True,
                         'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            sources.append({'recording_id': rid, 'audio_path': str(path)})
    # Include held-out people: even canonical initial validation visits cannot enter the sample.
    rows += [{**rows[0], 'recording_id': 'validation', 'participant_id': 'validation', 'split': 'val'}]
    monkeypatch.setattr(module, 'training_manifest', lambda path: (pd.DataFrame(rows), 'fixture_digest'))
    monkeypatch.setattr(module.pd, 'read_parquet', lambda path: pd.DataFrame(sources))
    first, second = tmp_path / 'sample_a', tmp_path / 'sample_b'
    module.prepare(tmp_path / 'snapshot', first, data)
    module.prepare(tmp_path / 'snapshot', second, data)
    a, b = pd.read_csv(first / 'manifest_PRIVATE.csv'), pd.read_csv(second / 'manifest_PRIVATE.csv')
    assert len(a) == a.participant_id.nunique() == 100
    assert a.label.value_counts().to_dict() == {0: 50, 1: 50}
    assert a.split.eq('train').all() and 'validation' not in set(a.recording_id)
    assert a.recording_id.tolist() == b.recording_id.tolist()
    (data / f'{a.recording_id.iloc[0]}.m4a').write_bytes(b'tampered')
    output = tmp_path / 'tampered_sample'
    with pytest.raises(ValueError, match='hash mismatch'):
        module.prepare(tmp_path / 'snapshot', output, data)
    assert not output.exists()
