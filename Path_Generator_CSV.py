import sys
import pandas as pd
import torch

from acoustools.Utilities import device, propagate, create_points, get_convert_indexes, add_lev_sig
from acoustools.Solvers import wgs_wrapper, gspat_wrapper, naive_solver_wrapper
from MLUtilities import do_NCNN

if __name__ == "__main__":

    FLIP_INDEXES = get_convert_indexes()

    path_file_csv = sys.argv[1]
    model_name = sys.argv[2]
    N = int(sys.argv[3])
    
    print("Loading File...")
    points_file = pd.read_csv("Path_Gen/PointCSVs/"+path_file_csv+'.csv',header=None)
    points = torch.tensor(points_file.values).to(device)
    F = points.shape[0] // N
    frames = torch.reshape(points,(F,1,3,N)).float()
    print("Generated Points")

    if model_name == "wgs":
        method = "wgs"
    elif model_name == "gspat":
        method = "gspat"
    elif model_name == "naive":
        method = "naive"
    else:
        model = torch.load("Models/model_"+model_name+".pth", map_location=torch.device(device))
        method = ''.join([i for i in model_name if not i.isdigit()]) #Assumes format of models is <TYPE><NUMBER> eg NCNN30

    solver = {
        "wgs":wgs_wrapper,
        "gspat":gspat_wrapper,
        "naive":naive_solver_wrapper,
        "NCNN":lambda p: do_NCNN(model, p)
    }

    print("Computing activations...")
    output_f = open("Path_gen/Outputs/"+path_file_csv+model_name+".csv","w")
    output_f.write(str(F)+","+str(512)+"\n")


    for j,points in enumerate(frames):
        act = solver[method](points)
        # print(torch.abs(propagate(act,points)))
        act = add_lev_sig(act)
        row = torch.angle(act).squeeze_()
        row_flip = row[FLIP_INDEXES]
        for i,phase in enumerate(row_flip):
                output_f.write(str(phase.item()))
                if i < 511:
                    output_f.write(",")
                else:
                    output_f.write("\n")
        
        if j % 1000 == 0:
            print("frame "+str(j)+"/"+str(F))

    output_f.close()
    







