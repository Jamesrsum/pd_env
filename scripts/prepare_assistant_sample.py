#!/usr/bin/env python3
"""Create a private, reproducible 100-recording training-only methodology sample.

Run only within approved storage using an individually authorized mPower account.
This is a development convenience sample, never a benchmark or data release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import pandas as pd
from src.training.adaptation_runner import training_manifest


def prepare(snapshot, output, data_root, seed=20261116, eda_root=None):
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    if output.is_relative_to(ROOT) or output == data_root or data_root.is_relative_to(output):
        raise ValueError('Output must be outside Git and must not contain the source data root')
    if output.exists():
        raise FileExistsError('Use a new private output directory; existing samples are immutable')
    frame, digest = training_manifest(Path(snapshot))
    candidates = frame.loc[frame.split.eq('train') & frame.initial_recording].copy()
    if candidates.participant_id.duplicated().any():
        raise ValueError('Repeated canonical training participant')
    eda_root = Path(eda_root or os.environ.get('MPOWER_OUTPUT_ROOT', data_root.parent / 'mpower_phase1'))
    recording = pd.read_parquet(eda_root / 'processed/recording_manifest.parquet')
    candidates = candidates.drop(columns=['audio_path'], errors='ignore').merge(
        recording[['recording_id', 'audio_path']], on='recording_id', how='left', validate='one_to_one')
    selected = pd.concat([candidates.loc[candidates.label.eq(label)].sort_values('recording_id')
                          .sample(n=50, random_state=seed + label) for label in [0, 1]], ignore_index=True)
    if selected.participant_id.nunique() != 100 or selected.duplicate_component.nunique() != 100:
        raise ValueError('Sample must contain 100 independent people/components')
    # Validate everything before creating/copying any output.
    for row in selected.itertuples():
        path = Path(row.audio_path).resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise ValueError('Missing recording or source path outside approved data root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.file_sha256:
            raise ValueError('Source recording hash mismatch')
    (output / 'audio').mkdir(parents=True)
    records = []
    for i, row in enumerate(selected.itertuples(), start=1):
        destination = output / 'audio' / f'sample_{i:03d}.m4a'
        shutil.copyfile(row.audio_path, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != row.file_sha256:
            raise ValueError('Copied recording failed checksum verification')
        records.append({'sample_file': str(destination.relative_to(output)), 'recording_id': row.recording_id,
                        'participant_id': row.participant_id, 'label': int(row.label),
                        'split': 'train', 'sha256': row.file_sha256, 'bytes': destination.stat().st_size})
    pd.DataFrame(records).to_csv(output / 'manifest_PRIVATE.csv', index=False)
    summary = {'recordings': 100, 'participants': 100, 'pd': 50, 'controls': 50,
               'split': 'train_only', 'canonical_initial_visits': True, 'seed': seed,
               'source_manifest_sha256': digest, 'all_copies_hash_verified': True,
               'total_bytes': sum(r['bytes'] for r in records),
               'selection': 'seeded stratified convenience sample; not population representative',
               'redistribution': 'Restricted mPower data: no GitHub upload; authorized users only.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# Private mPower methodology sample\n\n100 original encoded recordings, 100 training people, '
        '50 PD/50 controls; no validation or test recordings. '
        'Renamed filenames are pseudonyms, not anonymization. The private manifest retains linkage '
        'and checksums. Do not upload any part of this folder to GitHub. '
        'Each assistant must hold their own Synapse mPower access and work within approved permissions.\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=os.environ.get('MPOWER_DATA_ROOT'))
    p.add_argument('--eda-root', type=Path, default=os.environ.get('MPOWER_OUTPUT_ROOT'))
    p.add_argument('--seed', type=int, default=20261116)
    a = p.parse_args()
    if not a.data_root:
        p.error('Set MPOWER_DATA_ROOT or --data-root')
    os.umask(0o077)
    prepare(a.snapshot, a.output, a.data_root, a.seed, a.eda_root)
