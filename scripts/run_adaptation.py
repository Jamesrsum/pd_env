#!/usr/bin/env python3
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch,yaml
from src.training.adaptation_runner import BinaryAdapter,training_manifest,run_supervised


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    os.umask(0o077);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Training requires allocated GPU')
    cfg=yaml.safe_load((ROOT/'configs/adaptation_run.yaml').read_text());base=yaml.safe_load((ROOT/'configs/frozen_benchmark_local.yaml').read_text())
    spec=cfg['runs'][args.task];torch.manual_seed(cfg['seed'])
    frame,sha=training_manifest(os.environ['MPOWER_ADAPT_SNAPSHOT'],args.smoke,cfg['smoke_participants_per_cell'])
    model=BinaryAdapter(spec['model'],os.environ['MPOWER_MODEL_ROOT'],base)
    out=Path(os.environ['MPOWER_ADAPT_ROOT'])/('smoke' if args.smoke else 'runs')/f'{args.task:02d}_{spec["model"]}_{spec["policy"]}'
    if out.resolve()==ROOT or ROOT in out.resolve().parents:raise ValueError('Outputs must be private')
    out.mkdir(parents=True,exist_ok=True)
    schedule=model.configure(spec['policy'])
    (out/'provenance.json').write_text(json.dumps({'snapshot_sha256':sha,'task':spec,'config':cfg,'asset':model.asset,'schedule':schedule,'test_labels_read':False,
        'framework':torch.__version__,'code_sha256':hashlib.sha256((ROOT/'src/training/adaptation_runner.py').read_bytes()).hexdigest(),
        'frontend':'native checkpoint frontend; ResNet128mel25ms10ms resized224RGB fixed [-80,0]dB ImageNet normalized',
        'batchnorm':'all ResNet running statistics frozen','sampling':'one random recording per training participant per epoch'},indent=2))
    trainable={n:p.detach().cpu().clone() for n,p in model.backbone.named_parameters() if p.requires_grad} if args.smoke else {}
    run_supervised(frame,model,spec['policy'],cfg,out,'cuda',args.smoke)
    if args.smoke and trainable:
        updated=[n for n,p in model.backbone.named_parameters() if n in trainable and not torch.equal(trainable[n],p.detach().cpu())]
        if not updated:raise AssertionError('No requested backbone parameter updated')
        (out/'gradient_check.json').write_text(json.dumps({'backbone_parameters_changed':len(updated),'requested_trainable_parameters':len(trainable)}))
    print((out/'DONE.json').read_text(),flush=True)
if __name__=='__main__':main()
