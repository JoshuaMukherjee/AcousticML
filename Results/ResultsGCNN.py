import matplotlib.pyplot as plt
import os,sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader 


p = os.path.abspath('.')
sys.path.insert(1, p)

from acoustools.Solvers import wgs, gspat,naive_solver, naive_solver_batch, wgs_wrapper
from acoustools.Utilities import propagate, forward_model, device, create_points, add_lev_sig, propagate_abs, create_board, DTYPE, TRANSDUCERS
from acoustools.Visualiser import Visualise, Visualise_single, get_point_pos
from acoustools.Gorkov import gorkov_autograd

from MLUtilities import do_GCNN
from Dataset import GreenDataset

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

if "-8" in sys.argv:
    N = 8
else:
    N=4


if "-l" in sys.argv:

    if "-overfit" not in sys.argv:
        TRAINSIZE = 600000
        TESTSIZE=1000
    else:
        TRAINSIZE = 4
        TESTSIZE=2
    print(TRAINSIZE,TESTSIZE)


    model_name = sys.argv[1]

    loss = pickle.load(open("Losses/loss_"+model_name+".pth","rb"))
    model = torch.load("Models/model_"+model_name+".pth", map_location=torch.device(device))
    train,test = loss
    if "-abs" not in sys.argv:
        train = [t/TRAINSIZE for t in train]
        test = [t/TESTSIZE for t in test]
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

    model = torch.load("Models/model_"+model_name+latest+".pth", map_location=torch.device(device))

    P = 5
    if "-overfit" not in sys.argv:
        dataset = GreenDataset(P,N)
        print("Generated data...")
        
    else:
        dataset = torch.load("./Datasets/GreenDataset"+norm+"Train-4-4.pth")
        P = len(dataset)
        print("Using Overfitted Dataset...")

    data = iter(DataLoader(dataset,1,shuffle=True))
    press = []
    mins =[]
    wgs_200_ps = []
    gs_pat_ps = []
    naive_ps = []


    for p,d,g in data:

        out = do_GCNN(model, p, TRANSDUCERS)
        
        pressure = torch.abs(propagate(out,p))

        wgs_x = wgs_wrapper(p)
        wgs_p = torch.abs(propagate(wgs_x, p))
      
        A = forward_model(p[0,:])
        backward = torch.conj(A).T
        R = A@backward
        _,pres = gspat(R,A,backward,torch.ones(N,1).to(device)+0j, 200)

        gs_pat_p = torch.abs(pres)
    
        naive_p, naive_out = naive_solver_batch(p)

        press.append(pressure.cpu().detach().numpy())
        wgs_200_ps.append(wgs_p.squeeze_().cpu().detach().numpy())
        gs_pat_ps.append(gs_pat_p.squeeze_().cpu().detach().numpy())
        naive_ps.append(torch.abs(naive_p).squeeze_().cpu().detach().numpy())

    fig, axs = plt.subplots(1,P)
    fig.tight_layout()
    for i in range(P):
        to_plot = {}

        to_plot["Model"] = press[i].squeeze().real
        to_plot["WGS"] = wgs_200_ps[i].real
        to_plot["GS PAT"] = gs_pat_ps[i].real
        to_plot["Naive"] = naive_ps[i].real

        axs[i].boxplot(to_plot.values())
        axs[i].set_xticklabels(to_plot.keys(), rotation=90)
        axs[i].set_ylim(bottom=0,top=13000)
        # axs[i].set_yticklabels(range(0,13000,2000), rotation=90)
        axs[i].set_ylabel("Pressure (Pa)")

        print(press[i])
        
    plt.show()
