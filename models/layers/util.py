import torch


def print_tensor_stats(t: torch.Tensor, desc=''):
    t = t.to(torch.float)
    mean = torch.mean(t)
    min = torch.min(t)
    max = torch.max(t)
    std = torch.std(t)
    print('STAT {}: shape {} device {}; mean {:.3f}; min {:.3f}; max {:.3f}; std {:.3f}'
          .format(desc, t.size(), t.device, mean, min, max, std))
