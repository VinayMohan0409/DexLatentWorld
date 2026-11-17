"""A small stand-in for the sparse backbone: four feature levels with pooling links, so the model code can be tested without a GPU."""
import torch
import torch.nn as nn

WIDTHS = (48, 96, 144, 192)


class Point(dict):
    """dict with attribute access, like the backbone's point container."""
    __getattr__ = dict.__getitem__


class FakeBackbone(nn.Module):
    """Level 0 keeps every input point; each coarser level halves the points of every cloud (cluster i // 2); features are linear in the coordinates."""

    def __init__(self):
        """One linear layer per feature level maps 3-D coordinates to that level's width"""
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(3, w) for w in WIDTHS])

    def forward(self, data):
        """Build the four levels by halving each cloud's points and return the coarsest, with pooling_inverse and pooling_parent links back to level 0"""
        coord, offset = data["coord"], data["offset"]
        sizes = torch.diff(offset, prepend=offset.new_zeros(1)).tolist()
        levels = [dict(coord=coord, offset=offset, feat=self.lin[0](coord), sizes=sizes)]
        for lvl in range(1, 4):
            prev = levels[-1]
            inverse, coarse_sizes, start = [], [], 0
            for n in prev["sizes"]:
                inverse.append(torch.arange(n) // 2 + start)
                coarse_sizes.append((n + 1) // 2)
                start += (n + 1) // 2
            inverse = torch.cat(inverse).to(coord.device)
            counts = torch.bincount(inverse).clamp_min(1).unsqueeze(1).float()
            coarse = torch.zeros(start, 3).index_add_(0, inverse, prev["coord"]) / counts
            offset_c = torch.cumsum(torch.tensor(coarse_sizes), 0).to(coord.device)
            levels.append(dict(coord=coarse, offset=offset_c, feat=self.lin[lvl](coarse), sizes=coarse_sizes, inverse=inverse))
        points = [Point(coord=l["coord"], offset=l["offset"], feat=l["feat"]) for l in levels]
        for i in range(1, 4):
            points[i]["pooling_inverse"] = levels[i]["inverse"]
            points[i]["pooling_parent"] = points[i - 1]
        return points[3]
