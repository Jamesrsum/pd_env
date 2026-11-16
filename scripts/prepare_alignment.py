#!/usr/bin/env python3
"""Align cached full-context Whisper vectors with recomputed same-span Praat metrics."""
import argparse,hashlib,json,os,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd,yaml
from src.data.benchmark import digest
from src.training.adaptation_runner import training_manifest


def one(inputs):
    row,source,cache,out=inputs
    target=Path(out)/'pairs'/f'{row["recording_id"]}.npz'
    if target.exists():return {'recording_id':row['recording_id'],'status':'ok','path':str(target)}
    try:
        key=digest({'recording_id':row['recording_id'],'source':row['file_sha256'],'derived':row['mono16k_sha256'],'signature':source['signature']})
        with np.load(Path(cache)/f'{key}.npz',allow_pickle=False) as v:whisper=v['block_6/mean_sd/window_mean'].copy()
        from extract_acoustic_features import praat
        cfg=yaml.safe_load((ROOT/'configs/mpower_preprocessing.yaml').read_text())
        wave=np.load(row['waveform_path'])[:160000].astype('float32')
        metrics,_=praat(wave,16000,cfg['pitch'],prefix='')
        names=['jitter_local','jitter_rap','jitter_ppq5','shimmer_local','shimmer_db','shimmer_apq3','shimmer_apq5','hnr_db','f0_cv','f0_mean_hz']
        acoustic=np.array([metrics[n] for n in names],dtype='float32')
        if not np.isfinite(acoustic).any():raise ValueError('No defined acoustic targets')
        np.savez(target,whisper=whisper,acoustic=acoustic)
        return {'recording_id':row['recording_id'],'status':'ok','path':str(target)}
    except Exception as exc:return {'recording_id':row['recording_id'],'status':'error','reason':f'{type(exc).__name__}: {exc}'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();os.umask(0o077)
    original=Path(os.environ['MPOWER_FROZEN_ROOT']);out=Path(os.environ['MPOWER_ADAPT_ROOT'])/'alignment';(out/'pairs').mkdir(parents=True,exist_ok=True)
    frame,sha=training_manifest(os.environ['MPOWER_ADAPT_SNAPSHOT'],args.smoke,4)
    source=json.loads((original/'runs/whisper_small/full/provenance.json').read_text())
    if source['manifest_sha256']!=sha:raise ValueError('Whisper/source snapshot mismatch')
    cache=original/'pooled_features/whisper_small/full'/source['signature'];rows=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:
        for r in pool.map(one,[(r,source,str(cache),str(out)) for r in frame.to_dict('records')],chunksize=8):
            rows.append(r)
            if len(rows)%1000==0:print('Aligned',len(rows),'/',len(frame),flush=True)
    status=pd.DataFrame(rows);status.to_csv(out/('smoke_status.csv' if args.smoke else 'pair_status.csv'),index=False)
    paired=frame.merge(status[status.status.eq('ok')],on='recording_id',validate='one_to_one')
    # Scientific target failures remain audited; missing Whisper cache failures abort rather than change membership silently.
    errors=status[status.status.ne('ok')]
    if len(errors) and errors.reason.str.contains('FileNotFound|KeyError').any():raise ValueError('Incomplete or incompatible Whisper cache')
    paired.to_parquet(out/('smoke_pairs.parquet' if args.smoke else 'pairs.parquet'),index=False)
    (out/('SMOKE_READY.json' if args.smoke else 'READY.json')).write_text(json.dumps({'snapshot_sha256':sha,'attempted':len(frame),'paired':len(paired),'excluded':len(errors),'features':['jitter_local','jitter_rap','jitter_ppq5','shimmer_local','shimmer_db','shimmer_apq3','shimmer_apq5','hnr_db','f0_cv','f0_mean_hz'],'source_signature':source['signature'],'interval':'same cached first <=10s waveform, full Praat interval','test_labels_read':False},indent=2))
    print('Pair preparation complete',len(paired),flush=True)
if __name__=='__main__':main()
