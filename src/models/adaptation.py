"""Explicit trainable-layer policies for future supervised fine-tuning runners.

Input is a bare encoder/backbone, with a SEPARATE classifier head. FrozenEncoder
cannot be reused for training: its inference_mode/eval/cache path is deliberate.
"""
import torch


def set_trainable_backbone(backbone, architecture, last_blocks=0, full=False):
    if last_blocks<0:raise ValueError('last_blocks must be nonnegative')
    paths={'audio-spectrogram-transformer':'encoder.layer','wav2vec2':'encoder.layers',
           'wavlm':'encoder.layers','hubert':'encoder.layers','whisper':'layers',
           'resnet50':'layer4'}
    if architecture not in paths:raise ValueError(f'Unsupported architecture: {architecture}')
    # Resolve and validate first, so a bad request does not partially mutate the model.
    layers=backbone
    for field in paths[architecture].split('.'):
        if not hasattr(layers,field):raise ValueError(f'Missing backbone path {paths[architecture]}')
        layers=getattr(layers,field)
    if last_blocks>len(layers):raise ValueError('Requested more blocks than exist')
    backbone.requires_grad_(bool(full))
    if not full and last_blocks:
        for layer in list(layers)[-last_blocks:]:layer.requires_grad_(True)
    return {'total_parameters':sum(p.numel() for p in backbone.parameters()),
            'trainable_parameters':sum(p.numel() for p in backbone.parameters() if p.requires_grad),
            'trainable_names':[n for n,p in backbone.named_parameters() if p.requires_grad]}


def freeze_batchnorm_statistics(backbone):
    """Call after model.train() when a ResNet arm freezes running statistics."""
    for module in backbone.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
