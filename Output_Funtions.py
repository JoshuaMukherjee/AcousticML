from typing import Any
import torch
from Utilities import convert_to_complex

class PointNetOut():
    #comvert to complex activations (Bx1024xN) -> (Bx512)
    def __call__(self, out):
        out = convert_to_complex(out)
        out = torch.sum(out,dim=2)
        return out
