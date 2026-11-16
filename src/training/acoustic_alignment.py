"""Prototype recording-paired Whisper/acoustic alignment loss, not a training runner.

Call only on training examples with acoustic transforms fitted on training data.
Identifiers are integer-coded within the batch, never model inputs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AcousticAlignment(nn.Module):
    def __init__(self, whisper_dim, acoustic_dim, projection_dim=128):
        super().__init__()
        if min(whisper_dim, acoustic_dim, projection_dim) < 1:
            raise ValueError('Positive feature dimensions required')
        # Acoustic input includes transformed values and their observed-value masks.
        self.audio = nn.Sequential(nn.Linear(whisper_dim, 256), nn.GELU(), nn.Linear(256, projection_dim))
        self.acoustic = nn.Sequential(nn.Linear(acoustic_dim*2, 128), nn.GELU(), nn.Linear(128, projection_dim))

    def forward(self, whisper, acoustics, observed):
        if acoustics.shape != observed.shape or observed.dtype != torch.bool:
            raise ValueError('Provide an aligned boolean acoustic observation mask')
        if not torch.isfinite(whisper).all() or not torch.isfinite(acoustics).all():
            raise ValueError('Impute/scale using training-fitted transforms before projection')
        if not observed.any(dim=1).all():
            raise ValueError('Rows with no observed acoustic metrics cannot form valid pairs')
        return (F.normalize(self.audio(whisper), dim=-1),
                F.normalize(self.acoustic(torch.cat([acoustics, observed.to(acoustics.dtype)], dim=-1)), dim=-1))


def paired_alignment_loss(audio, acoustic, participant_ids, duplicate_groups=None, temperature=.1):
    """Symmetric recording-paired InfoNCE, excluding same-person/duplicate negatives.

    A diagonal pair must describe the SAME recording and analysis interval.
    Other visits by the same person are masked, not assumed physiologically equal.
    Batch should sample one recording per participant and retain real negatives.
    """
    if audio.ndim!=2 or audio.shape!=acoustic.shape or len(audio)<2:
        raise ValueError('Require >=2 paired vectors of equal shape')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be finite and positive')
    n=len(audio)
    if participant_ids.shape!=(n,) or participant_ids.device!=audio.device:
        raise ValueError('Batch identifiers must align with embeddings on the same device')
    if not torch.isfinite(audio).all() or not torch.isfinite(acoustic).all():
        raise ValueError('Nonfinite embeddings')
    if (audio.norm(dim=-1)==0).any() or (acoustic.norm(dim=-1)==0).any():
        raise ValueError('Zero embeddings cannot define cosine similarity')
    allowed=participant_ids[:,None]!=participant_ids[None,:]
    if duplicate_groups is not None:
        if duplicate_groups.shape!=(n,) or duplicate_groups.device!=audio.device:
            raise ValueError('Duplicate groups must align with embeddings')
        allowed &= duplicate_groups[:,None]!=duplicate_groups[None,:]
    if not allowed.any(dim=1).all():
        raise ValueError('Every anchor needs a genuine negative participant/component')
    allowed.fill_diagonal_(True)
    logits=F.normalize(audio,dim=-1) @ F.normalize(acoustic,dim=-1).T / temperature
    logits=logits.masked_fill(~allowed, -torch.inf)
    target=torch.arange(n,device=audio.device)
    return .5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))
