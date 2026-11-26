"""Keep checkpointing enabled while matching deterministic old-policy scoring."""
import torch


def disable_training_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.p=0.0
        # Qwen3-VL attention passes this scalar to functional attention kernels.
        if isinstance(getattr(module,'attention_dropout',None),(float,int)):
            module.attention_dropout=0.0
