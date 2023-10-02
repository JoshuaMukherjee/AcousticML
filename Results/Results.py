import matplotlib.pyplot as plt
import os,sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader 


p = os.path.abspath('.')
sys.path.insert(1, p)

from Solvers import wgs, gspat
from Dataset import PointDataset
from Utilities import propagate, forward_model, device


if "-p" in sys.argv:

    
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth")


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


if "-h" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth")

    P = 100
    N = 4
    dataset = PointDataset(P,N)
    data = iter(DataLoader(dataset,1,shuffle=True))
    pressure_means = []
    pressure_means_wgs = []
    for p,a,pr in data:
        out = model(p)
        out = propagate(out,p)
        presssure = torch.abs(out)
        pressure_means.append(torch.mean(presssure).cpu().detach().numpy())
        pressure_means_wgs.append(torch.mean(torch.abs(pr)).cpu().detach().numpy())
    
    
    plt.hist(pressure_means, label="Model", histtype=u'step')
    plt.hist(pressure_means_wgs,label="WGS", histtype=u'step')
    plt.legend()
    plt.show()

if "-c" in sys.argv:
    start = int(sys.argv[-2])
    end = int(sys.argv[-1])
    P = 5
    N = 4
    dataset = PointDataset(P,N)
    # data = iter(DataLoader(dataset,1,shuffle=True))

    means = {}
    ranges = {}
    for model_name in range(start,end):
        model = torch.load("Models/model_PN"+str(model_name)+".pth")
        
        data = iter(DataLoader(dataset,1,shuffle=True))
        
        for p,a,pr in data:
            out = model(p)
            out = propagate(out,p)
            presssure = torch.abs(out)
            # print(model_name,presssure)

            if model_name not in means:
                means[model_name] = []
                ranges[model_name] = []


            means[model_name].append(torch.mean(presssure).cpu().detach().numpy())
            ranges[model_name].append(torch.std(presssure).cpu().detach().numpy())
    to_plot = []
    sizes = []
    for model in means:

        to_plot.append(np.mean(means[model]))
    
    for model in ranges:
        sizes.append(np.mean(ranges[model]))
    # print(means)
    # plt.boxplot(to_plot.values())
    # plt.xticklabels(to_plot.keys())
    # plt.ylim(bottom=0,top=13000)

    x = np.linspace(start,end-1,end-start)
    print(start, to_plot)

    plt.scatter(x,to_plot,s=sizes)

    plt.show()