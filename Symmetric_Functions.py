from matplotlib.pyplot import axes
import torch



class SymMax():
    def __init__(self,axis=2):
        self.axis = axis
    
    def __call__(self,x):
        return torch.max(x,axis=self.axis).values


class SymSum():
    def __init__(self,axis=2):
        self.axis = axis
    
    def __call__(self,x):
        return torch.sum(x,axis = self.axis)
    
