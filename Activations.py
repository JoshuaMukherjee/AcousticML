import torch
from torch.nn import Module


class SineActivation(Module):

    def __init__(self):
        super(SineActivation,self).__init__()
    
    def forward(self,x):
        return torch.sin(x)



def plotSELU():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")

    x = torch.linspace(-10,5,100)
    selu = torch.nn.SELU()
    y = selu(x)
    plt.plot(x,y)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show()

if __name__ == "__main__":
    plotSELU()

