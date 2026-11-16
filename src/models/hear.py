"""HeAR's official frontend and differentiable record-level adapter.

Weights are separately gated; this module never downloads or executes remote code.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.data.frozen_protocol import common_windows
from src.models.hear_vendor.audio_utils import preprocess_audio


def verify_assets(model_root, lock_path):
    entry = json.loads(Path(lock_path).read_text())['models']['hear_pytorch']
    root = Path(model_root) / 'checkpoints' / 'hear_pytorch'
    for item in entry['files']:
        path = root / item['name']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise FileNotFoundError(f'HeAR asset unavailable: {item["name"]}; obtain authorized Hugging Face access and run the pinned downloader')
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest() if item.get('sha256') else hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if actual != (item.get('sha256') or item['git_blob']):
            raise ValueError(f'HeAR integrity failure: {item["name"]}')
    return root, entry


def frontend(windows):
    """Per-window extrema preserve batching invariance and input immutability."""
    result = []
    for window in windows:
        x = torch.as_tensor(window, dtype=torch.float32)
        if x.ndim != 1 or x.numel() != 32000 or not torch.isfinite(x).all():
            raise ValueError('HeAR requires exactly 32000 finite real mono16k samples')
        # Upstream subtracts min in-place, across its entire input batch.
        result.append(preprocess_audio(x.clone().unsqueeze(0)))
    if not result:
        raise ValueError('Empty window batch')
    return torch.cat(result)


class HearAdapter(nn.Module):
    def __init__(self, backbone, block_path=None):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1))
        self.kind = 'hear'
        self.policy = 'frozen'
        self.block_path = block_path

    @classmethod
    def from_local(cls, model_root, lock_path, block_path=None):
        from transformers import AutoModel
        root, entry = verify_assets(model_root, lock_path)
        backbone, loading = AutoModel.from_pretrained(root, local_files_only=True,
            trust_remote_code=False, output_loading_info=True)
        failures = {k: v for k,v in loading.items() if v}
        if failures:
            raise ValueError(f'HeAR checkpoint did not load completely: {failures}')
        model = cls(backbone, block_path)
        model.asset = entry
        model.loading_info = loading
        return model

    def blocks(self):
        if not self.block_path:
            raise ValueError('Inspect the authorized model and explicitly configure its transformer block path before last2 training')
        layers = self.backbone.get_submodule(self.block_path)
        if not isinstance(layers, (nn.ModuleList, nn.Sequential)) or len(layers) < 2:
            raise ValueError('Configured HeAR block path is not a sequence of >=2 blocks')
        return layers

    def configure(self, policy):
        if policy not in ('frozen', 'last2', 'full'):
            raise ValueError('Unknown HeAR policy')
        layers = self.blocks() if policy == 'last2' else None
        self.backbone.requires_grad_(policy == 'full')
        if layers is not None:
            for layer in list(layers)[-2:]:
                layer.requires_grad_(True)
        self.policy = policy
        return {'trainable_names': [n for n,p in self.backbone.named_parameters() if p.requires_grad],
                'trainable_parameters': sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
                'total_parameters': sum(p.numel() for p in self.backbone.parameters())}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode and self.policy == 'full':
            self.backbone.train()
        elif mode and self.policy == 'last2':
            for layer in list(self.blocks())[-2:]:
                layer.train()
        return self

    def encode_windows(self, windows, device):
        output = self.backbone(frontend(windows).to(device)).pooler_output
        if output.shape != (len(windows), 512) or not torch.isfinite(output).all():
            raise ValueError('HeAR must produce finite [windows,512] pooler_output')
        return output

    def record_features(self, wave, device, mean_sd=False):
        windows, _ = common_windows(np.asarray(wave), 'matched_2s')
        z = self.encode_windows(windows, device)
        mean = z.mean(0)
        return torch.cat([mean, z.std(0, correction=0)]) if mean_sd else mean

    def forward_waves(self, waves, device):
        # Pool inside the autograd graph BEFORE the record-level binary loss.
        return self.head(torch.stack([self.record_features(w, device) for w in waves])).reshape(-1)
