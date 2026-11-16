import pytest
import torch
from torch import nn
from src.models.adaptation import set_trainable_backbone, freeze_batchnorm_statistics
from src.training.acoustic_alignment import paired_alignment_loss, AcousticAlignment


def test_freeze_policies():
    model=nn.Module();model.encoder=nn.Module();model.encoder.layers=nn.ModuleList([nn.Linear(3,3) for _ in range(4)])
    model.feature_extractor=nn.Linear(3,3)
    result=set_trainable_backbone(model,'wavlm',last_blocks=2)
    assert result['trainable_parameters']==24
    assert not model.feature_extractor.weight.requires_grad
    assert not model.encoder.layers[0].weight.requires_grad
    assert model.encoder.layers[-1].weight.requires_grad
    with pytest.raises(ValueError):set_trainable_backbone(model,'wavlm',last_blocks=9)
    assert model.encoder.layers[-1].weight.requires_grad
    assert set_trainable_backbone(model,'wavlm',full=True)['trainable_parameters']==60
    assert set_trainable_backbone(model,'wavlm')['trainable_parameters']==0


def test_alignment_gradients_and_pairing():
    torch.manual_seed(42)
    a=torch.eye(4,requires_grad=True);b=torch.eye(4,requires_grad=True);ids=torch.arange(4)
    correct=paired_alignment_loss(a,b,ids)
    wrong=paired_alignment_loss(a,b.roll(1,0),ids)
    assert correct<wrong
    correct.backward();assert a.grad is not None and torch.isfinite(a.grad).all()
    # Same person has no off-diagonal negative even when there are multiple visits.
    a=torch.eye(3);ids=torch.tensor([1,1,2]);loss=paired_alignment_loss(a,a,ids)
    modified=a.clone();modified[1]=a[0]
    assert torch.isfinite(paired_alignment_loss(a,modified,ids))
    with pytest.raises(ValueError):paired_alignment_loss(a,a,torch.ones(3))
    with pytest.raises(ValueError):paired_alignment_loss(a,a,ids,duplicate_groups=torch.ones(3))
    with pytest.raises(ValueError):paired_alignment_loss(a,a,ids,temperature=0)


def test_projection_validity():
    model=AcousticAlignment(8,3);w=torch.randn(4,8);x=torch.randn(4,3);mask=torch.ones(4,3,dtype=torch.bool)
    a,b=model(w,x,mask);assert a.shape==(4,128)
    loss=paired_alignment_loss(a,b,torch.arange(4));loss.backward()
    assert model.acoustic[0].weight.grad is not None
    mask[0]=False
    with pytest.raises(ValueError):model(w,x,mask)


def test_batchnorm_freeze():
    model=nn.Sequential(nn.BatchNorm2d(3),nn.Conv2d(3,3,1));model.train()
    freeze_batchnorm_statistics(model)
    assert not model[0].training and model[1].training
