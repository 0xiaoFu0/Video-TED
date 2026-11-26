"""Reference preparation independent of Ray, CUDA and model implementation."""
from __future__ import annotations
import torch
from .reward import parse_answer, target_label
from .response_format import validate_reference_format


def encode_references(tokenizer, texts, targets, max_length, eligible=None):
    """Encode eligible references; inactive rows contain only masked PAD tokens.

    Eligibility comes from the annotation status, never from a failed validation.
    An eligible malformed or overlong reference remains a hard error.
    """
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must define EOS and PAD')
    if len(texts) != len(targets) or not len(texts):
        raise ValueError('Reference batch shape mismatch')
    if max_length < 1:
        raise ValueError('Reference max length must be positive')
    if eligible is None:
        eligible = torch.ones(len(texts), dtype=torch.bool)
    else:
        eligible = torch.as_tensor(eligible, dtype=torch.bool)
    if eligible.shape != (len(texts),):
        raise ValueError('Reference eligibility shape mismatch')
    encoded=[]
    for text,target,active in zip(texts,targets,eligible):
        if not active:
            encoded.append([])
            continue
        if parse_answer(text) != target_label(target):
            raise ValueError('Reference answer does not match target')
        validate_reference_format(text)
        ids=tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        if len(ids)>max_length:
            raise ValueError('Reference exceeds max length; refuse to truncate its answer')
        encoded.append(ids)
    # Keep a stable response axis even when no row has a reference. No answer,
    # reasoning text, or EOS is fabricated for those rows.
    width=max(1,max(map(len,encoded)))
    responses=torch.full((len(encoded),width),tokenizer.pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(responses)
    for i,ids in enumerate(encoded):
        responses[i,:len(ids)]=torch.tensor(ids)
        mask[i,:len(ids)]=1
    return responses,mask


def reference_eligibility(statuses, allow_unreviewed):
    permitted={'verified','reviewed'}
    if allow_unreviewed:
        permitted.add('unreviewed')
    known=permitted|{'unreviewed','rejected'}
    if any(status not in known for status in statuses):
        raise ValueError('Unknown reference quality status')
    return torch.tensor([status in permitted for status in statuses],dtype=torch.bool)


def concat_prompt_response(prompt_ids, prompt_mask, prompt_positions, responses, response_mask):
    """Same mRoPE continuation convention as upstream rollout/OPSD concat."""
    delta=torch.arange(1,responses.shape[-1]+1,device=prompt_positions.device)
    delta=delta.view(1,-1).expand(prompt_ids.shape[0],-1)
    if prompt_positions.ndim==3:
        delta=delta[:,None,:].expand(-1,prompt_positions.shape[1],-1)
    positions=torch.cat([prompt_positions,prompt_positions[...,-1:]+delta],dim=-1)
    return torch.cat([prompt_ids,responses],dim=1),torch.cat([prompt_mask,response_mask],dim=1),positions
