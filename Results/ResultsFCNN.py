import matplotlib.pyplot as plt
import os,sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader 


p = os.path.abspath('.')
sys.path.insert(1, p)

from Solvers import wgs, gspat
from Dataset import PointDataset, FDataset, FDatasetNorm
from Utilities import propagate, forward_model, device

if "-latest" in sys.argv:
    latest = "_latest"
    print(latest)
else:
    latest = ""

if "-normD" in sys.argv: #Use a normalised dataset
    norm = "Norm"
    print(norm)
else:
    latest = ""

if "-l" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+latest+".pth")

    loss = pickle.load(open("Losses/loss_"+model_name+".pth","rb"))
    train,test = loss
    if "-abs" not in sys.argv:
        train = [t/20000 for t in train]
        test = [t/1000 for t in test]
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

if "-p" in sys.argv:
    
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+latest+".pth")

    N = 4
    P = 5
    if "-overfit" not in sys.argv:
        dataset = FDataset(P,N)
        print("Generated data...")
        
    else:
        dataset = torch.load("Datasets\FDataset"+norm+"Train-4-4.pth")
        P = len(dataset)

    data = iter(DataLoader(dataset,1,shuffle=True))
    press = []
    mins =[]
    wgs_200_ps = []
    gs_pat_ps = []


    for F,p,a,pr in data:


        out = model(F)
        
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
    fig.tight_layout()
    for i in range(P):
        to_plot = {}

        to_plot["Model"] = press[i]
        to_plot["WGS"] = wgs_200_ps[i]
        to_plot["GS PAT"] = gs_pat_ps[i]

        axs[i].boxplot(to_plot.values())
        axs[i].set_xticklabels(to_plot.keys(), rotation=90)
        axs[i].set_ylim(bottom=0,top=13000)
        axs[i].set_yticklabels(range(0,13000,2000), rotation=90)
        axs[i].set_ylabel("Pressure (Pa)")

        print(press[i])
        
    plt.show()


if "-t" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth")
    N = 4
    P = 1
    dataset = FDataset(P,N)
    data = iter(DataLoader(dataset,1,shuffle=True))

    F,p,a,pr = next(data)
    out = model(F)

    out.squeeze_()

    print(torch.abs(out))
    print(torch.angle(out))

if "-h" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth")

    P = 100
    N = 4
    dataset = FDataset(P,N)
    data = iter(DataLoader(dataset,1,shuffle=True))
    pressure_means = []
    pressure_means_wgs = []
    for F,p,a,pr in data:
        out = model(F)
        out = propagate(out,p)
        presssure = torch.abs(out)
        pressure_means.append(torch.mean(presssure).cpu().detach().numpy())
        pressure_means_wgs.append(torch.mean(torch.abs(pr)).cpu().detach().numpy())
    
    
    plt.hist(pressure_means, label="Model", histtype=u'step')
    plt.hist(pressure_means_wgs,label="WGS", histtype=u'step')
    plt.legend()
    plt.show()