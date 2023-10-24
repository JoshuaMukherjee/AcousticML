import matplotlib.pyplot as plt
import os,sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader 


p = os.path.abspath('.')
sys.path.insert(1, p)

from Solvers import wgs, gspat,naive_solver, naive_solver_batch
from Dataset import PointDataset, FDataset, FDatasetNorm, NaiveDataset
from Utilities import propagate, forward_model, device, do_NCNN

if "-latest" in sys.argv:
    latest = "_latest"
    print(latest)
else:
    latest = ""

if "-normD" in sys.argv: #Use a normalised dataset
    norm = "Norm"
    print(norm)
else:
    norm = ""


if "-l" in sys.argv:

    TRAINSIZE = 100000
    TESTSIZE=1000

    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+latest+".pth")

    loss = pickle.load(open("Losses/loss_"+model_name+".pth","rb"))
    train,test = loss
    if "-abs" not in sys.argv:
        train = [t/100000 for t in train]
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
        dataset = NaiveDataset(P,N)
        print("Generated data...")
        
    else:
        dataset = torch.load("./Datasets/NaiveDataset"+norm+"Train-4-4.pth")
        P = len(dataset)

    data = iter(DataLoader(dataset,1,shuffle=True))
    press = []
    mins =[]
    wgs_200_ps = []
    gs_pat_ps = []
    naive_ps = []


    for p,a,pr,naive in data:


        out = do_NCNN(model, p)
        
        pressure = torch.abs(propagate(out,p))

        wgs_p = torch.abs(pr)
      
        A = forward_model(p[0,:])
        backward = torch.conj(A).T
        R = A@backward
        _,pres = gspat(R,A,backward,torch.ones(N,1).to(device)+0j, 200)

        gs_pat_p = torch.abs(pres)
    
        naive_p, naive_out = naive_solver_batch(p)
        print(torch.abs(naive_p))
       


        press.append(pressure.cpu().detach().numpy())
        wgs_200_ps.append(wgs_p.squeeze_().cpu().detach().numpy())
        gs_pat_ps.append(gs_pat_p.squeeze_().cpu().detach().numpy())
        naive_ps.append(torch.abs(naive_p).squeeze_().cpu().detach().numpy())

    fig, axs = plt.subplots(1,P)
    fig.tight_layout()
    for i in range(P):
        to_plot = {}

        to_plot["Model"] = press[i]
        to_plot["WGS"] = wgs_200_ps[i]
        to_plot["GS PAT"] = gs_pat_ps[i]
        to_plot["Naive"] = naive_ps[i]

        axs[i].boxplot(to_plot.values())
        axs[i].set_xticklabels(to_plot.keys(), rotation=90)
        axs[i].set_ylim(bottom=0,top=13000)
        # axs[i].set_yticklabels(range(0,13000,2000), rotation=90)
        axs[i].set_ylabel("Pressure (Pa)")

        print(press[i])
        
    plt.show()


if "-t" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth")
    N = 4
    P = 1
    dataset = NaiveDataset(P,N)
    data = iter(DataLoader(dataset,1,shuffle=True))

    p,a,pr,naive = next(data)
    out = do_NCNN(model,p)

    out.squeeze_()

    print(torch.abs(out))
    print(torch.angle(out))

if "-h" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth")

    P = 100
    N = 4
    dataset = NaiveDataset(P,N)
    data = iter(DataLoader(dataset,1,shuffle=True))
    pressure_means = []
    pressure_means_wgs = []
    pressure_means_naive = []
    
    for p,a,pr,naive in data:
        out = do_NCNN(model,p)
        out = propagate(out,p)
        # presssure = torch.abs(out)
        
        for pres in torch.abs(out):
            pressure_means.append(pres.cpu().detach().numpy())
        

        for presWGS in torch.abs(pr).squeeze_():
            pressure_means_wgs.append(presWGS.cpu().detach().numpy())

        naive_p,_ = naive_solver_batch(p)
        for presN in torch.abs(naive_p).squeeze_():
            pressure_means_naive.append(presN.cpu().detach().numpy())
    
    
    plt.hist(pressure_means, label="Model", histtype=u'step')
    plt.hist(pressure_means_wgs,label="WGS", histtype=u'step')
    plt.hist(pressure_means_naive,label="Naive", histtype=u'step')
    plt.legend()
    plt.show()
