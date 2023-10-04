from typing import Any
import torch
from Utilities import convert_to_complex

class PointNetOut():
    #comvert to complex activations (Bx1024xN) -> (Bx512)
    def __call__(self, out):
        out = convert_to_complex(out)
        out = torch.sum(out,dim=2)
        return out

class PointNetPhaseOnly():
    #comvert to complex activations (Bx512xN) -> (Bx512)
    def __call__(self, out):
        
        out = torch.sum(out,dim=2)
        out = torch.e ** (1j*out)

        return out

class PointNetOutConAmp():
    # Bx1024xN -> Bx512, constrain amp
     def __call__(self, out):
        out = convert_to_complex(out)
        out = torch.sum(out,dim=2)
        out = out / torch.abs(out)
        return out

class PointNetOutClipAmp():
     def __call__(self, out):
        out = convert_to_complex(out)
        out = torch.sum(out,dim=2)
        amp = torch.abs(out)
        mask = (amp >= 1)
        out[mask] = out[mask] / amp[mask]
        out = out / torch.abs(out)
        return out
