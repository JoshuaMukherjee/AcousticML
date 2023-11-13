
import json
import traceback
import torch
from torch.utils.data import DataLoader

from Train_Network import train
import Networks, Loss_Functions
from Utilities import device
from Dataset import PointDataset, FDataset, FDatasetNorm, NaiveDataset, PressureTargetDataset
import Extra_Point_Functions
import Network_Train_Functions


files = [
   "mCNN20"
]

def parse(params,name):

    start_epochs = params["start-epochs"]
    epochs = params["epochs"]

    if start_epochs == 0:
        net = getattr(Networks, params["net"])(**params["net-args"]).to(device)
    else:
        start_name = params["start_model"] #Which model to start from
        net = torch.load("./Models/model_"+start_name+".pth",map_location=torch.device(device))

    try:
        train_s = [torch.load("./Datasets/"+pth,map_location=torch.device(device)) for pth in params["train"] ]
        test_s =  [torch.load("./Datasets/"+pth,map_location=torch.device(device)) for pth in params["test"]  ]
    except Exception as e:
        print("Datasets not found, please generate using ```python3 Dataset.py```")
        print(e)

    batch = params["batch"]
    train_sets = [DataLoader(d,batch,shuffle=True) for d in train_s]
    test_sets = [DataLoader(d,batch,shuffle=True) for d in test_s]

    optimiser = getattr(torch.optim, params["optimiser"])(net.parameters(),**params["optimiser-args"])
    loss_function = getattr(Loss_Functions, params["loss-function"])
    
    if "loss-params" in params:
        loss_params = params["loss-params"]
    else:
        loss_params = {}
    
    supervised = params["supervised"]
   
    if "scheduler" in params:
        scheduler = getattr(torch.optim.lr_scheduler, params["scheduler"])(optimiser,**params["scheduler-args"])
    else:
        scheduler = None
    
    if "random-stop" in params:
        rand_stop = params["random-stop"]
    else:
        rand_stop = False
    
    if "clip" in params:
        clip = params["clip"]
        if "clip-args" in params:
            clip_params = params["clip-args"]
        else:
            clip_params = {}
    else:
        clip = False
        clip_params = {}
    
    if "log-grad" in params:
        log_grad = params["log-grad"]
    else:
        log_grad = False
    
    if "norm-loss" in params:
        norm_loss = True
    else:
        norm_loss = False

    if "extra-points-fun" in params:
        extra_points_fun = getattr(Extra_Point_Functions,params["extra-points-fun"])
        if "extra-points-args" in params:
            extra_points_args = params["extra-points-args"]
        else:
            extra_points_args = {}

        if "maximise-first-N" in params:
            maximise_first_N = params["maximise-first-N"]
        else:
            maximise_first_N = -1
    else:
        extra_points_fun = None
        extra_points_args = {}
        maximise_first_N = -1

    
    if "train-function" in params:
        train_function = getattr(Network_Train_Functions, params["train-function"])
    else:
        train_function = Network_Train_Functions.default_functions[params["net"]]
        

    if "solver" in params:
        solver = params["solver"]
    else:
        solver = "wgs"

    
    train(net,start_epochs,epochs,train_sets,test_sets,optimiser,
        loss_function,loss_params, supervised, scheduler, 
        name, batch, rand_stop, clip, clip_params, log_grad,
        norm_loss, extra_points_fun, extra_points_args, maximise_first_N,
        train_function, solver)

if __name__ == '__main__':
    for file in files:
        try:
            print(file, "Parsing....")
            params = json.load(open("Params/"+file+".json","r"))
            parse(params,file)
        except Exception as e:
            print(traceback.format_exc())

