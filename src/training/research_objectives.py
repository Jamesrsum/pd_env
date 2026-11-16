"""Explicit participant-aware objectives for controlled adaptation experiments."""
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn


class DiagnosisProjector(nn.Sequential):
    def __init__(self, representation_dim):
        super().__init__(nn.Linear(representation_dim,128),nn.GELU(),nn.Linear(128,32))


def controlled_objective(logits, labels, representations, participants, components,
                         mode='bce', projector=None, auxiliary_prediction=None,
                         targets=None, observed=None, groups=None,
                         balanced_batch=False, training_pd_prior=None):
    """Matched BCE/SupCon/acoustic arms; auxiliary heads never replace diagnosis."""
    if mode not in ('bce','supcon','acoustic'):raise ValueError('Unknown research arm')
    binary=F.binary_cross_entropy_with_logits(logits,labels.float(),reduction='none')
    if balanced_batch:
        if training_pd_prior is None or not 0<training_pd_prior<1:raise ValueError('Participant training prior required')
        weights=torch.where(labels.bool(),2*training_pd_prior,2*(1-training_pd_prior))
        binary=binary*weights
    binary=binary.mean()
    if mode=='supcon':
        if projector is None:raise ValueError('Projection head required')
        auxiliary=diagnosis_supcon(projector(representations),labels,participants,components)
    elif mode=='acoustic':
        if any(v is None for v in [auxiliary_prediction,targets,observed,groups]):raise ValueError('Acoustic targets/mask/groups required')
        auxiliary=grouped_masked_huber(auxiliary_prediction,targets,observed,groups)
    else:auxiliary=binary*0
    return binary+.1*auxiliary,{'bce':float(binary.detach()),'auxiliary':float(auxiliary.detach())}


def acoustic_target_scaling(values, frame):
    """Fit only observed canonical initial TRAIN values, preserving missingness."""
    if not frame.split.isin(['train','val']).all():raise ValueError('Sealed test forbidden')
    ix=(frame.split.eq('train') & frame.initial_recording).to_numpy()
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or len(x)!=len(frame) or not ix.any():raise ValueError('Invalid target table')
    x=np.where(np.isfinite(x),x,np.nan)
    keep=np.isfinite(x[ix]).any(0)
    mean=np.nanmean(x[ix][:,keep],axis=0);sd=np.nanstd(x[ix][:,keep],axis=0);sd=np.where(sd>1e-8,sd,1.)
    transformed=(x[:,keep]-mean)/sd
    return transformed,np.isfinite(transformed),{'columns':np.flatnonzero(keep),'mean':mean,'scale':sd}


def diagnosis_supcon(z, labels, participants, components, temperature=.07):
    if temperature <= 0 or len(z) != len(labels):
        raise ValueError('Invalid contrastive batch')
    same_person = torch.as_tensor(np.equal.outer(participants, participants), device=z.device)
    same_component = torch.as_tensor(np.equal.outer(components, components), device=z.device)
    eligible = ~(same_person | same_component)
    same_label = labels[:, None].eq(labels[None, :])
    positive = eligible & same_label
    negative = eligible & ~same_label
    if not positive.any(1).all() or not negative.any(1).all():
        raise ValueError('Every anchor needs distinct-person/component positive and negative pairs')
    normalized = F.normalize(z, dim=-1)
    logits = normalized @ normalized.T / temperature
    denominator = torch.logsumexp(logits.masked_fill(~eligible, -torch.inf), dim=1)
    logp = logits - denominator[:, None]
    return -(logp.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()


def grouped_masked_huber(prediction, target, observed, groups):
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError('Auxiliary target/mask shape mismatch')
    flat = [i for group in groups for i in group]
    if len(set(flat)) != len(flat) or sorted(flat) != list(range(target.shape[1])):
        raise ValueError('Target groups must partition the target schema')
    observed = observed.bool()
    if not torch.isfinite(target[observed]).all():
        raise ValueError('Observed targets must be finite')
    # Remove undefined targets before calculating the loss, including NaNs.
    safe_target = torch.where(observed, target, prediction.detach())
    losses = F.huber_loss(prediction, safe_target, reduction='none')
    per_group = []
    for group in groups:
        mask = observed[:, group]
        count = mask.sum(1)
        per_record = (losses[:, group] * mask).sum(1) / count.clamp_min(1)
        if count.gt(0).any():
            per_group.append(per_record[count.gt(0)].mean())
    return torch.stack(per_group).mean() if per_group else prediction.sum() * 0


def balanced_people_batches(frame, seed, batch_size=32):
    """One visit/person per epoch; never fill a batch by repeating a person."""
    if batch_size % 2 or batch_size < 4:
        raise ValueError('Contrastive batch size must be even and >=4')
    rng = np.random.default_rng(seed)
    people = frame.groupby('participant_id', sort=True)
    if people.label.nunique().gt(1).any():
        raise ValueError('Conflicting participant labels')
    rows = [rng.choice(g.index.to_numpy()) for _,g in people]
    selected = frame.loc[rows]
    classes = [rng.permutation(selected.index[selected.label.eq(c)]) for c in (0,1)]
    # Equal batches sample the majority without replacement; vary it each epoch.
    size = min(map(len, classes))
    for start in range(0, size, batch_size // 2):
        pieces = [a[start:min(size,start+batch_size//2)] for a in classes]
        if min(map(len,pieces)) < 2:
            continue
        ix = np.concatenate(pieces); rng.shuffle(ix)
        yield ix
