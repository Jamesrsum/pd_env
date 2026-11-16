# 100-recording mPower methodology sample

A real sample is prepared on approved HPC storage at
`$MPOWER_DATA_ROOT/../assistant_samples/mpower_100_20261116/`:

```text
mpower_100_20261116/
  audio/sample_001.m4a ... sample_100.m4a
  manifest_PRIVATE.csv
  summary.json
  README.md
```

It contains **100 original encoded recordings from 100 distinct training
participants**, 50 PD and 50 controls, using each person's canonical initial
eligible visit. Every original/copy SHA256 was verified. Total size is
133,961,942 bytes. No validation/test recordings are included. Filenames are
pseudonyms; they do not anonymize voices. The private manifest preserves source
linkage, diagnosis, checksums and sizes for an authorized analyst.

Selection is a seeded stratified convenience sample (seed20261116), intended
for feature/preprocessing development, not representative performance estimation.
It contains successfully decoded eligible recordings; it does not cover every
malformed-source failure mode. Use synthetic failure fixtures for those tests.

## Why the recordings are outside GitHub

mPower voice table `syn5511444` has managed access requirements5549295 and
terms5549296. They require validated/certified users, an intended-data-use statement
and confidential/secure handling. Synapse prohibits redistributing controlled
data, including to collaborators who have not individually obtained access.
A private GitHub repository does not establish each recipient's authorization.
See the [mPower access instructions](https://www.synapse.org/Synapse:syn4993293/wiki/247860),
[public voice-table access requirements](https://repo-prod.prod.sagebase.org/repo/v1/entity/syn5511444/accessRequirement?limit=10&offset=0),
and [Synapse controlled-data sharing rules](https://docs.synapse.org/synapse-docs/sharing-settings-permissions-and-conditions-for-use).

The lab assistant should obtain individual Synapse mPower access and use this
folder within the lab's approved data environment. Do not copy it into a Git
checkout or send it through GitHub/email. No participant manifest, audio, voice
spectrogram, individual prediction, embedding or trained research checkpoint is
included here.

An authorized researcher can recreate the sample in a **new** private directory:

```bash
python scripts/prepare_assistant_sample.py \
  --snapshot "$MPOWER_BENCHMARK_ROOT/protocol_experiments/20261114T074847Z/snapshot" \
  --data-root "$MPOWER_DATA_ROOT" \
  --eda-root "$MPOWER_OUTPUT_ROOT" \
  --output "$MPOWER_DATA_ROOT/../assistant_samples/mpower_100_new"
```

Requires the benchmark environment; source manifests remain read-only. The
script refuses existing output directories, in-repository destinations, source
hash mismatches and reused participant/duplicate components.
