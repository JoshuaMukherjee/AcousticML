import matplotlib.pyplot as plt
import os,sys
import torch
import pickle
from torch.utils.data import DataLoader


p = os.path.abspath('.')
sys.path.insert(1, p)

from acoustools.Solvers import wgs, gspat, wgs_wrapper
from Dataset import DistanceDataset
from acoustools.Utilities import propagate, forward_model, device, DTYPE
from Extra_Point_Functions import add_sine_points

import acoustools.Constants as c

OVERFIT = '-overfit' in sys.argv

if "-latest" in sys.argv:
    latest = "_latest"
    print(latest)
else:
    latest = ""

if "-p" in sys.argv:
    CONST = '-con' in sys.argv

    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+latest+".pth", map_location=torch.device(device))


    N = 4
    P = 4
    if not OVERFIT:
        dataset = DistanceDataset(P,N)
    else:
        print('Using Overfit dataset...')
        dataset = torch.load("./Datasets/DistanceDatasetTrain-4-4.pth")
    
    data = iter(DataLoader(dataset,1,shuffle=True))

    
    press = []
    mins =[]
    wgs_200_ps = []
    gs_pat_ps = []


    for p,d in data:
       
        green = torch.exp(d*1j*c.k) / d
        B = green.shape[0]
        N = green.shape[2]
        green_real = torch.view_as_real(green).reshape(B,-1,N)
        out = model(green_real).unsqueeze(1).mT

        if CONST:
            out = out / torch.abs(out)
        
        pressure = torch.abs(propagate(out,p))

        wgs_x = wgs_wrapper(p)
        wgs_p = torch.abs(propagate(wgs_x, p))
       
        A = forward_model(p[0,:])
        backward = torch.conj(A).T
        R = A@backward
        _,pres = gspat(R,A,backward,torch.ones(N,1).to(device)+0j, 200)

        gs_pat_p = torch.abs(pres)
      
        press.append(pressure.cpu().detach().numpy())
        wgs_200_ps.append(wgs_p.squeeze_().cpu().detach().numpy())
        gs_pat_ps.append(gs_pat_p.squeeze_().cpu().detach().numpy())

    fig, axs = plt.subplots(1,P)
    fig.tight_layout()
    for i in range(P):
        to_plot = {}
        to_plot["Model"] = press[i].squeeze().real

        to_plot["WGS"] = wgs_200_ps[i].real
        to_plot["GS PAT"] = gs_pat_ps[i].real
        axs[i].boxplot(to_plot.values())
        axs[i].set_xticklabels(to_plot.keys(), rotation=90)
        axs[i].set_ylim(bottom=0,top=6000)
        # axs[i].set_yticklabels(range(0,6000,2000), rotation=90)
        axs[i].set_ylabel("Pressure (Pa)")

        
    plt.show()


if "-l" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth")

    loss = pickle.load(open("Losses/loss_"+model_name+".pth","rb"))
    train,test = loss
    print(len(train))
    plt.plot(train,label="train")
    plt.plot(test,label="test")
    plt.yscale("symlog")
    
    plt.xlabel("epoch")
    plt.ylabel("loss")

    try:
        max_epoch = model.epoch_saved
        plt.plot(max_epoch,test[max_epoch],"x",label="Best Test Loss")
    except AttributeError:
        pass

    plt.legend()
    plt.show()

if '-draw' in sys.argv:
    from torchviz import make_dot  

    dataset = DistanceDataset(1,4)
    data = iter(DataLoader(dataset,1,shuffle=True))


    model_name = sys.argv[1]
    model = torch.load("Models/model_"+model_name+latest+".pth", map_location=torch.device(device))

    p,d = next(data)
    y = model(d)

    
    make_dot(y, params=dict(model.named_parameters())).render(model_name+'_diagram', format="png")
