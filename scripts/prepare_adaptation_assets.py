"""Download public ResNet weights; check local HeAR without assuming availability."""
import hashlib,json,os
from pathlib import Path
import torch
from torchvision.models import resnet50, ResNet50_Weights
root=Path(os.environ['MPOWER_ADAPT_ROOT']); models=Path(os.environ['MPOWER_MODEL_ROOT'])
models.mkdir(parents=True,exist_ok=True)
weights=ResNet50_Weights.IMAGENET1K_V2
path=models/'resnet50_imagenet1k_v2.pth'
if not path.exists():torch.hub.download_url_to_file(weights.url,str(path),hash_prefix='11ad3fa6',progress=False)
state=torch.load(path,map_location='cpu',weights_only=True);resnet50(weights=None).load_state_dict(state,strict=True)
info={'resnet50':{'status':'ready','path':str(path),'url':weights.url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'hear':{'status':'blocked_missing_weights'}}
# Use only already-downloaded HeAR assets; no extra gated download is implied.
from huggingface_hub import snapshot_download
try:
    p=Path(os.environ['MPOWER_HEAR_CHECKPOINT']) if os.environ.get('MPOWER_HEAR_CHECKPOINT') else Path(snapshot_download('google/hear',local_files_only=True))
    if (p/'saved_model.pb').exists() and list((p/'variables').glob('*.data-*')) and (p/'variables/variables.index').exists():
        info['hear']={'status':'assets_present_runtime_unverified','path':str(p)}
except Exception as exc: info['hear']['reason']=type(exc).__name__
(root/'assets.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
