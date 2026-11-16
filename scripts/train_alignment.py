#!/usr/bin/env python3
"""Train recording-paired projections; evaluate controlled ablations on same pairs."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd,torch,yaml,joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.training.acoustic_alignment import AcousticAlignment,paired_alignment_loss
from src.training.adaptation_runner import epoch_indices
from src.training.development import fit_development


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();os.umask(0o077);torch.set_num_threads(4)
    root=Path(os.environ['MPOWER_ADAPT_ROOT']);pairs=root/'alignment';out=root/('alignment_smoke' if a.smoke else 'alignment_runs');out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_parquet(pairs/('smoke_pairs.parquet' if a.smoke else 'pairs.parquet')).reset_index(drop=True)
    if not frame.split.isin(['train','val']).all():raise ValueError('Test data forbidden')
    frame['initial_recording']=~frame.duplicated('participant_id')
    if frame.groupby('participant_id').split.nunique().gt(1).any() or frame.groupby('duplicate_component').split.nunique().gt(1).any():raise ValueError('Leakage')
    vectors=[np.load(path,allow_pickle=False) for path in frame.path]
    w=np.stack([v['whisper'] for v in vectors]);x=np.stack([v['acoustic'] for v in vectors]);[v.close() for v in vectors]
    train=frame.split.eq('train').to_numpy();tr=frame[train]
    initial=train & frame.initial_recording.to_numpy()
    columns=np.isfinite(x[initial]).any(0);x=x[:,columns];mask=np.isfinite(x);x=np.where(mask,x,np.nan)
    if not len(x[0]):raise ValueError('No train-observed acoustic columns')
    imputer=SimpleImputer(strategy='median').fit(x[initial]);x=imputer.transform(x)
    scaler=StandardScaler().fit(x[initial]);x=scaler.transform(x).astype('float32')
    ws=StandardScaler().fit(w[initial]);w=ws.transform(w).astype('float32')
    joblib.dump({'imputer':imputer,'acoustic_scaler':scaler,'whisper_scaler':ws,'columns':columns},out/'transforms.joblib')
    torch.manual_seed(20261112);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AcousticAlignment(w.shape[1],x.shape[1]).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    pid=torch.tensor(pd.factorize(frame.participant_id)[0],device=device);dup=torch.tensor(pd.factorize(frame.duplicate_component)[0],device=device)
    wx=torch.tensor(w,device=device);ax=torch.tensor(x,device=device);mx=torch.tensor(mask,device=device)
    log=[];epochs=2 if a.smoke else 30;last=out/'last.pt';begin=0
    if last.exists():
        state=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);begin=state['epoch']+1;log=state['history']
    for epoch in range(begin,epochs):
        indices=epoch_indices(tr,20261112+epoch);model.train();losses=[]
        for start in range(0,len(indices),32):
            ix=indices[start:start+32]
            if len(ix)<2:continue
            optimizer.zero_grad();aw,bw=model(wx[ix],ax[ix],mx[ix]);loss=paired_alignment_loss(aw,bw,pid[ix],dup[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite alignment loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        log.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'train_participants':len(indices)})
        torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':log},out/'last.tmp.pt');(out/'last.tmp.pt').replace(last)
        print(log[-1],flush=True)
    model.eval();aligned_audio=[];aligned_acoustic=[]
    with torch.no_grad():
        for start in range(0,len(frame),512):
            aa,bb=model(wx[start:start+512],ax[start:start+512],mx[start:start+512]);aligned_audio.append(aa.cpu().numpy());aligned_acoustic.append(bb.cpu().numpy())
    aa=np.concatenate(aligned_audio);bb=np.concatenate(aligned_acoustic)
    # Negative-control projections train on permuted TRAIN targets only; evaluation pairing remains correct.
    shuffled=AcousticAlignment(w.shape[1],x.shape[1]).to(device);torch.manual_seed(20261112)
    optimizer2=torch.optim.AdamW(shuffled.parameters(),lr=1e-4);rng=np.random.default_rng(20261112)
    permutation=np.arange(len(frame));permutation[train]=rng.permutation(np.flatnonzero(train))
    for epoch in range(epochs):
        indices=epoch_indices(tr,20261112+epoch)
        for start in range(0,len(indices),32):
            ix=indices[start:start+32]
            if len(ix)<2:continue
            optimizer2.zero_grad();a2,b2=shuffled(wx[ix],ax[permutation[ix]],mx[permutation[ix]])
            loss=paired_alignment_loss(a2,b2,pid[ix],dup[ix]);loss.backward();optimizer2.step()
    shuffled.eval()
    with torch.no_grad():control=np.concatenate([shuffled(wx[i:i+512],ax[i:i+512],mx[i:i+512])[0].cpu().numpy() for i in range(0,len(frame),512)])
    base=yaml.safe_load((ROOT/'configs/frozen_model_testing.yaml').read_text())
    if a.smoke:base['head'].update(epochs=2,cv_folds=2,C_grid=[.1])
    matrices={'whisper_only':w,'acoustic_only':np.c_[x,mask.astype('float32')],'concatenation':np.c_[w,x,mask.astype('float32')],
        'aligned_audio_only':aa,'aligned_fusion':np.c_[aa,bb],'shuffled_training_pairs':control,'missingness_only':mask.astype('float32')}
    rows=[]
    for name,matrix in matrices.items():
        directory=out/name;directory.mkdir(exist_ok=True)
        fitted=fit_development(matrix,frame,base['head'],20261112)
        for head,result in fitted.items():
            joblib.dump(result['state'],directory/f'{head}.joblib');result['initial_validation_predictions'].to_csv(directory/f'{head}_predictions.csv',index=False)
            rows.append({'experiment':name,'head':head,**result['metrics']})
    pd.DataFrame(rows).to_csv(out/'validation_leaderboard.csv',index=False)
    (out/'DONE.json').write_text(json.dumps({'status':'smoke_complete' if a.smoke else 'development_complete','epochs':epochs,'test_labels_read':False,
        'paired_recordings':len(frame),'train_participants':tr.participant_id.nunique(),'val_participants':frame[~train].participant_id.nunique(),
        'selected_acoustic_columns':int(columns.sum()),'transforms_fit_on':'initial training recording per participant','interval':'identical first <=10s complete span','ablations':list(matrices)},indent=2))
if __name__=='__main__':main()
