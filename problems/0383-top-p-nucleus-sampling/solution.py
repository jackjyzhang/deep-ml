import torch

def top_p_sampling(logits: list[float], p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a torch.Tensor
    """

    logits = torch.tensor(logits)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, descending=True)
    cump = torch.cumsum(sorted_probs, dim=-1)
    to_mask = torch.zeros_like(indices).bool()
    to_mask[1:] = cump[:-1] >= p
    logits[indices[to_mask]] = -torch.inf
    return torch.softmax(logits, dim=-1)





    '''
    logits = torch.tensor(logits)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, descending=True)
    cursum = 0
    for i in range(len(logits)):
        cursum += sorted_probs[i]
        if cursum >= p:
            break
    print(logits)
    logits[indices[i+1:]] = -torch.inf
    print(logits)
    renorm_probs = torch.softmax(logits, dim=-1)

    return renorm_probs
    '''