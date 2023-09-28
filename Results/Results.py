import matplotlib.pyplot as plt
import os,sys
import torch
import pickle
from torch.utils.data import DataLoader 


p = os.path.abspath('.')
sys.path.insert(1, p)

from Solvers import wgs, gspat
from Dataset import PointDataset
from Utilities import propagate, forward_model, device


model_name = sys.argv[1]

model = torch.load("Models/model_"+model_name+".pth")



if "-p" in sys.argv:

    N = 4
    P = 5
    dataset = PointDataset(P,N)
    data = iter(DataLoader(dataset,1,shuffle=True))

    
    press = []
    wgs_200_ps = []
    gs_pat_ps = []


    for p,a,pr in data:
    
        out = model(p)
        pressure = torch.abs(propagate(out,p))
        wgs_p = torch.abs(pr)


        A = forward_model(p[0,:])
        backward = torch.conj(A).T
        R = A@backward
        _,pres = gspat(R,A,backward,torch.ones(N,1).to(device)+0j, 200)

        gs_pat_p = torch.abs(pres)

        press.append(pressure.cpu().detach().numpy())
        wgs_200_ps.append(wgs_p.squeeze_().cpu().detach().numpy())
        gs_pat_ps.append(gs_pat_p.squeeze_().cpu().detach().numpy())

    fig, axs = plt.subplots(1,P)
    for i in range(P):
        to_plot = {}
        to_plot["Model"] = press[i]
        to_plot["WGS"] = wgs_200_ps[i]
        to_plot["GS PAT"] = gs_pat_ps[i]
        axs[i].boxplot(to_plot.values())
        axs[i].set_xticklabels(to_plot.keys())
        axs[i].set_ylim(bottom=0,top=13000)
    plt.show()


if "-l" in sys.argv:
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



