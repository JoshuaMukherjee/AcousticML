import matplotlib.pyplot as plt
import os,sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader 


p = os.path.abspath('.')
sys.path.insert(1, p)

from acoustools.Solvers import wgs, gspat,naive
from acoustools.Utilities import propagate, forward_model, device, create_points, add_lev_sig, propagate_abs, create_board
from acoustools.Visualiser import Visualise, Visualise_single, get_point_pos
from acoustools.Gorkov import gorkov_autograd

from MLUtilities import do_NCNN
from Dataset import PointDataset, FDataset, FDatasetNorm, NaiveDataset


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


    for p,a,pr,naive_holo in data:


        out = do_NCNN(model, p)
        
        pressure = torch.abs(propagate(out,p))

        wgs_p = torch.abs(pr)
      
        A = forward_model(p[0,:])
        backward = torch.conj(A).T
        R = A@backward
        _,pres = gspat(R=R,A=A,B=backward,b=torch.ones(N,1).to(device)+0j, iterations=200,return_components=True)

        gs_pat_p = torch.abs(pres)
    
        naive_out, naive_p = naive(p, return_components=True)
        print(torch.abs(naive_p))
       


        press.append(pressure.cpu().detach().numpy())
        wgs_200_ps.append(wgs_p.squeeze_().cpu().detach().numpy())
        gs_pat_ps.append(gs_pat_p.squeeze_().cpu().detach().numpy())
        naive_ps.append(torch.abs(naive_p).squeeze_().cpu().detach().numpy())

    fig, axs = plt.subplots(1,P)
    fig.tight_layout()
    for i in range(P):
        to_plot = {}

        to_plot["Model"] = press[i].squeeze()
        to_plot["WGS"] = wgs_200_ps[i].squeeze()
        to_plot["GS PAT"] = gs_pat_ps[i].squeeze()
        to_plot["Naive"] = naive_ps[i].squeeze()

        axs[i].boxplot(to_plot.values())
        axs[i].set_xticklabels(to_plot.keys(), rotation=90)
        axs[i].set_ylim(bottom=0,top=5500)
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

    model = torch.load("Models/model_"+model_name+".pth",map_location=torch.device(device))

    P = 100
    dataset = NaiveDataset(P,N)
    data = iter(DataLoader(dataset,1,shuffle=True))
    pressure_means = []
    pressure_means_wgs = []
    pressure_means_naive = []
    
    for p,a,pr,naive in data:
        out = do_NCNN(model,p)
        out = propagate(out,p)
        # presssure = torch.abs(out)
        
        for pres in torch.abs(out).squeeze_():
            pressure_means.append(pres.cpu().detach().numpy())
        

        for presWGS in torch.abs(pr).squeeze_():
            pressure_means_wgs.append(presWGS.cpu().detach().numpy())

        naive_p,_ = naive(p)
        for presN in torch.abs(naive_p).squeeze_():
            pressure_means_naive.append(presN.cpu().detach().numpy())

    print(len(pressure_means))
    print(len(pressure_means_wgs))
    print(len(pressure_means_naive))
    
    plt.hist([pressure_means,pressure_means_wgs,pressure_means_naive] , label=["Model","WGS","Naive"], histtype=u'step', bins=30)
    # plt.hist(pressure_means_wgs,label="WGS", histtype=u'step')
    # plt.hist(pressure_means_naive,label="Naive", histtype=u'step')
    plt.legend()
    plt.show()

if "-r" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth")

    P = 100
    dataset = NaiveDataset(P,N)
    data = iter(DataLoader(dataset,1,shuffle=True))
    pressure_std = []
    pressure_std_wgs = []
    pressure_std_naive = []
    
    for p,a,pr,naive in data:
        out = do_NCNN(model,p)
        out = propagate(out,p)
        presssure = torch.abs(out)
        
        
        pressure_std.append(torch.std(presssure).cpu().detach().numpy())
        

        
        pressure_std_wgs.append(torch.std(torch.abs(pr)).cpu().detach().numpy())

        naive_p,_ = naive(p)
        
        pressure_std_naive.append(torch.std(torch.abs(naive_p)).cpu().detach().numpy())

    print(len(pressure_std))
    print(len(pressure_std_wgs))
    print(len(pressure_std_naive))
    
    plt.hist([pressure_std,pressure_std_wgs,pressure_std_naive] , label=["Model","WGS","Naive"], histtype=u'step', bins=30)
    # plt.hist(pressure_means_wgs,label="WGS", histtype=u'step')
    # plt.hist(pressure_means_naive,label="Naive", histtype=u'step')
    plt.legend()
    plt.show()

if "-vg" in sys.argv:
   
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+".pth", map_location=torch.device(device))
    N = 4
    P = 1
    points= create_points(N,y=0)
   
    out = do_NCNN(model,points)
    activation = add_lev_sig(out)

    A = torch.tensor((-0.08, 0, 0.08))
    B = torch.tensor((0.08, 0, 0.08))
    C = torch.tensor((-0.08, 0, -0.08))

    print("Visualising...")
    Visualise(A,B,C,activation,points,[propagate_abs,gorkov_autograd])

if "-v" in sys.argv:
    model_name = sys.argv[1]

    TRANS = '-trans' in sys.argv

    model = torch.load("Models/model_"+model_name+".pth", map_location=torch.device(device))
    N = 4
    P = 1

    if "-overfit" not in sys.argv:
        dataset = NaiveDataset(P,N)
        print("Generated data...")
        
    else:
        dataset = torch.load("./Datasets/NaiveDataset"+norm+"Train-4-4.pth")
    
    data = iter(DataLoader(dataset,1,shuffle=True))

    results = []

    x_left = -0.08
    x_right = 0.08

    A = torch.tensor((x_left, 0, 0.13))
    B = torch.tensor((x_right, 0, 0.13))
    C = torch.tensor((x_left, 0, -0.13))
    res = (300,300)
    # res=(10,10)
    labels = []
    point_poses = []

    pitch=0.0105
    grid_vec=pitch*(torch.arange(-17/2+1, 17/2, 1))
    Z = .234/2
  
    trans = []

    # print(trans_x)
    d = next(data)
    for points,a,pr,naive in [d]:
        print(points)
        out = do_NCNN(model,points)
        activation = add_lev_sig(out)
        
        for i in range(N):
            point = points[:,:,i].unsqueeze_(2)
            y = point[0,1,0]
            A[1] = y
            B[1] = y
            C[1] = y
            if TRANS:
                top_pos = torch.stack([grid_vec,torch.ones_like(grid_vec) * y, torch.ones_like(grid_vec) * Z])
                top_pos.unsqueeze_(0)
                top = get_point_pos(A,B,C,top_pos,flip=False,res=res)

                bottom_pos = torch.stack([grid_vec,torch.ones_like(grid_vec) * y, torch.ones_like(grid_vec) * -1*Z])
                bottom_pos.unsqueeze_(0)
                bottom = get_point_pos(A,B,C,bottom_pos,flip=False,res=res)

                trans.append(top + bottom)

            result = Visualise_single(A,B,C,activation,res=res)
            point_pos = get_point_pos(A,B,C,point,res=res)
            point_poses.append(point_pos)
            results.append(result)
            labels.append(str(round(point[0,0,0].item(),3)) + "," + str(round(point[0,1,0].item(),3)) + "," + str(round(point[0,2,0].item(),3)))

   
    axs = []
    ids = [1,2,4,5]
    for i,result in enumerate(results):
       

        ax = plt.subplot(2,3,ids[i])
        axs.append(ax)
        im = plt.imshow(result.cpu().detach().numpy(),cmap="hot")
        plt.xticks([])
        plt.yticks([])
        # plt.colorbar()
        plt.title(labels[i])
        pts_pos_t = point_poses[i][0]
        plt.scatter(pts_pos_t[1],pts_pos_t[0],marker="x") #Point positions
        if TRANS:
            trans_x = [t[0] for t in trans[i]]
            trans_y = [t[1] for t in trans[i]]
            plt.scatter(trans_x,trans_y,marker="s",color='black') #Transducers
        plt.xlim(0, res[1])
    
    cax = plt.subplot(1,30,21)
    plt.colorbar(im,cax=cax,fraction=0.01)
    
    # plt.tight_layout()
    plt.show()

if "-g" in sys.argv:
    model_name = sys.argv[1]

    model = torch.load("Models/model_"+model_name+latest+".pth", map_location=torch.device(device))

    P = 5
    if "-overfit" not in sys.argv:
        dataset = NaiveDataset(P,N)
        print("Generated data...")
        
    else:
        dataset = torch.load("./Datasets/NaiveDataset"+norm+"Train-4-4.pth")
        P = len(dataset)

    data = iter(DataLoader(dataset,1,shuffle=True))
    us = []
    wgs_200_us = []
    gs_pat_us = []
    naive_us = []


    for p,a,pr,naive in data:


        out = do_NCNN(model, p)
        out = add_lev_sig(out)
        U = gorkov_autograd(out,p)
        

        U_wgs = gorkov_autograd(add_lev_sig(a.unsqueeze_(2)),p)
      
        A = forward_model(p[0,:])
        backward = torch.conj(A).T
        R = A@backward
        act,_ = gspat(R,A,backward,torch.ones(N,1).to(device)+0j, 200)
        act = act.T
        U_GSPAT = gorkov_autograd(add_lev_sig(act.unsqueeze_(2)),p)
    
        naive_p, naive_out = naive(p)
        naive_out.unsqueeze_(2)
        U_naive = gorkov_autograd(add_lev_sig(naive_out),p)
       


        us.append(U.squeeze_().cpu().detach().numpy())
        wgs_200_us.append(U_wgs.squeeze_().cpu().detach().numpy())
        gs_pat_us.append(U_GSPAT.squeeze_().cpu().detach().numpy())
        naive_us.append(U_naive.squeeze_().cpu().detach().numpy())



    fig, axs = plt.subplots(1,P)
    fig.tight_layout()
    for i in range(P):
        to_plot = {}

        print("Model", us[i])
        print("WGS",wgs_200_us[i])
        print("GSPAT",gs_pat_us[i])
        print("Naive",naive_us[i])

        print()

        to_plot["Model"] = us[i]
        to_plot["WGS"] = wgs_200_us[i]
        to_plot["GS PAT"] = gs_pat_us[i]
        to_plot["Naive"] = naive_us[i]

        axs[i].boxplot(to_plot.values())
        axs[i].set_xticklabels(to_plot.keys(), rotation=90)
        axs[i].set_ylim(top=1e-5,bottom=-9e-5)
        # axs[i].set_yticklabels(range(0,13000,2000), rotation=90)
        axs[i].set_ylabel("U")


    plt.show()
