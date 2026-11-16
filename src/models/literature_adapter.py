"""Differentiable pretrained encoders for explicitly labelled literature adaptations."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.models.hear import HearAdapter
from src.models.parameter_efficient import StandardizedLinear,inject_lora,LoRALinear


class LiteratureAdapter(nn.Module):
    def __init__(self,model_root,repo,cfg,lp_state=None):
        super().__init__();self.family=cfg['family'];self.policy=cfg['policy'];self.lora_targets=[]
        if self.family=='hear_lpft':
            native=HearAdapter.from_local(model_root,Path(repo)/'configs/pretrained_models.lock.json','encoder.layer')
            self.encoder=native;self.backbone=native.backbone;self.asset=native.asset
            if lp_state is None:raise ValueError('LP-FT requires the actual converged train-fitted probe')
            scaler=lp_state['scaler'];head=lp_state['head']
            if list(head.classes_)!=[0,1]:raise ValueError('Binary probe class ordering mismatch')
            self.head=StandardizedLinear(scaler.mean_,scaler.scale_,head.coef_,head.intercept_)
            self.width=512
            # Native random head is unused; keep it frozen and out of parameter audits.
            native.head.requires_grad_(False)
        elif self.family=='hubert_supcon':
            from transformers import AutoModel,AutoFeatureExtractor,AutoConfig
            lock=json.loads((Path(repo)/'configs/literature_models.lock.json').read_text())['models']['hubert_base_ls960']
            path=Path(model_root)/'checkpoints/hubert_base_ls960'
            provenance=json.loads((path/'download_provenance.json').read_text())
            if provenance['revision']!=lock['revision']:raise ValueError('HuBERT checkpoint revision mismatch')
            config=AutoConfig.from_pretrained(path,local_files_only=True)
            self.backbone,loading=AutoModel.from_pretrained(path,config=config,local_files_only=True,
                trust_remote_code=False,output_loading_info=True)
            if any(loading.values()):raise ValueError(f'Incomplete HuBERT encoder load: {loading}')
            config=self.backbone.config
            # Construct/load the native masked-spec parameter before disabling
            # latent augmentation. Setting probability=0 before construction
            # suppresses that module and creates a misleading unexpected key.
            for name in ['mask_time_prob','mask_feature_prob','layerdrop']:
                if hasattr(config,name):setattr(config,name,0.)
            self.runtime_config={name:getattr(config,name) for name in ['mask_time_prob','mask_feature_prob','layerdrop']}
            self.feature=AutoFeatureExtractor.from_pretrained(path,local_files_only=True)
            self.width=config.hidden_size
            if self.width!=768:raise ValueError('Paper-inspired HuBERT baseline requires 768D base encoder')
            self.head=nn.Sequential(nn.Linear(768,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,2))
            # Paper leaves projector depth unspecified. A single linear32 is explicit.
            self.projector=nn.Linear(768,32)
            self.asset={**lock,'loading_info':loading}
        else:raise ValueError('Unknown literature family')
        self.backbone.requires_grad_(self.policy=='full')
        if self.policy=='lora':
            self.lora_targets=inject_lora(self.backbone,['attention.attention.query','attention.attention.value'],
                cfg['lora_rank'],cfg['lora_alpha'],cfg['lora_dropout'])
        elif self.policy not in ['full','frozen']:raise ValueError('Unknown adaptation policy')
        if self.family=='hubert_supcon':
            # Deliberate project deviation: frozen CNN, matched in both speech arms.
            self.backbone.feature_extractor.requires_grad_(False)
            if hasattr(self.backbone,'masked_spec_embed'):self.backbone.masked_spec_embed.requires_grad_(False)
        if cfg.get('gradient_checkpointing') and self.policy!='frozen':
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def train(self,mode=True):
        super().train(mode)
        if self.family=='hear_lpft':
            # Zero-dropout HeAR; keep the frozen base deterministic for LoRA too.
            self.backbone.eval()
            if mode and self.policy=='full':self.backbone.train()
            if mode and self.policy=='lora':
                for module in self.backbone.modules():
                    if isinstance(module,LoRALinear):module.dropout.train()
        elif mode:
            self.backbone.feature_extractor.eval()
        return self

    def representations(self,waves,device):
        if self.family=='hear_lpft':
            return torch.stack([self.encoder.record_features(w,device) for w in waves])
        vectors=[]
        for wave in waves:
            # Same real <=5s interval in CE and SupCon; never invent a padded tail.
            x=self.feature([np.asarray(wave,dtype=np.float32)[:80000]],sampling_rate=16000,
                           return_tensors='pt',padding=False)
            output=self.backbone(input_values=x.input_values.to(device),return_dict=True)
            vectors.append(output.last_hidden_state.mean(1))
        return torch.cat(vectors)

    def classify(self,z,detached=False):
        return self.head(z.detach() if detached else z)
