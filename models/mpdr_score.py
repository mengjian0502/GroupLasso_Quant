import numpy as np
import torch

def get_mpdr_score(weight):
    normalizer = weight.norm() ** 2

    # sort weight
    sorted_weight, sorted_idx = weight.abs().view(-1).sort(descending=False)

    # compute (weight_squared_cumsum[i] = \sum_{j < i} sorted_weight[j]^2)
    weight_square_cumsum_temp = (sorted_weight ** 2).cumsum(dim=0)
    weight_square_cumsum = torch.zeros(weight_square_cumsum_temp.shape)
    weight_square_cumsum[1:] = weight_square_cumsum_temp[: len(weight_square_cumsum_temp) - 1]

    # normalized weights with appropriate normalizers
    # (i-th normalizer = sqrt(|W|_F^2 - \sum_{j < i} sorted_weight[j]^2))
    sorted_weight /= (normalizer - weight_square_cumsum).sqrt()

    # rearrange entries to their original potisions
    score = torch.zeros(weight_square_cumsum.shape)
    score[sorted_idx] = sorted_weight

    # reshape weight
    score = score.view(weight.shape)
    return score