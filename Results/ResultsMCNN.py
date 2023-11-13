import matplotlib.pyplot as plt
import os,sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader 


p = os.path.abspath('.')
sys.path.insert(1, p)

from Solvers import wgs, gspat,naive_solver, naive_solver_batch
from Dataset import PointDataset, FDataset, FDatasetNorm, NaiveDataset,PressureTargetDataset
from Utilities import propagate, forward_model, device, do_NCNN, create_points, add_lev_sig, propagate_abs, create_board, generate_pressure_targets
from Visualiser import Visualise, Visualise_single, get_point_pos
from Gorkov import gorkov_autograd

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

    TRAINSIZE = 4
    TESTSIZE=2

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

if "-sp" in sys.argv:

    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+latest+".pth", map_location=torch.device(device))

    P = 100
    N = 4
    if "-overfit" not in sys.argv:
        dataset = PressureTargetDataset(P,N)
        print("Generated data...")
        
    else:
        print("Overfit")
        dataset = torch.load("./Datasets/PressureTargetDataset"+norm+"Train-4-4.pth")
        P = len(dataset)

    data = iter(DataLoader(dataset,1,shuffle=True))
    xs = []
    ys = []
    for p,a,pr,naive,targets in data:
        targets = targets.squeeze_()
        

        act_in = torch.reshape(a,(1,2,16,16))
        act_phases = torch.angle(act_in)
        
        activation_out_img = model(act_phases, targets) 
        
        activation_out = torch.reshape(activation_out_img,(1,512,1))
        pressure_out = torch.abs(propagate(activation_out, p))

        ys = ys+[y.detach().cpu().item() for y in pressure_out]
        xs = xs+[x.detach().cpu().item() for x in targets]
    
    plt.scatter(xs,ys)
    plt.xlim(4000,11000)
    plt.ylim(0,11000)

    plt.xlabel("Target (Pa)")
    plt.ylabel("Output (Pa)")
    z = np.polyfit(xs,ys,1)
    print(z)
    p = np.poly1d(z)

    x_lin = np.linspace(4000,11000)
    plt.plot(xs, p(xs), "r-",label="Trend Line, m="+str(round(z[0],2))+", c="+str(round(z[1],0)))
    # plt.plot(xs, xs, "b-", label="y=x")
    plt.legend()
    plt.show()