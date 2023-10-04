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
    # Bx1024xN -> Bx512, constrain amp
     def __call__(self, out):
        out = convert_to_complex(out)
        out = torch.sum(out,dim=2)
       
        mask = (torch.abs(out) > 1)
        
        result = torch.ones_like(out)
        result[mask] = out[mask] / torch.abs(out[mask])
        result[~mask] = out[~mask]

        return result


class PointNetOutSin():
    def __call__(self, out):
        #B x 1024 x N -> B x N x 512 x 2  -> Bx512xN & Bx512xN -> Ae^ix -> Bx512xN (complex)
        out = torch.permute(out,(0,2,1))
        out = out.view((out.shape[0],out.shape[1],-1,2))

        amp = torch.sin(out[:,:,:,0])
        phase = torch.sin(out[:,:,:,1])

        return amp * torch.e ** (1j*phase)

