import torch
import json
import sys
import itertools

from Utilities import device, propagate, create_points, do_NCNN, get_convert_indexes, add_lev_sig
from Solvers import wgs_wrapper, gspat_wrapper, naive_solver_wrapper



def interpolate(start,end, max_step_size=0.0001):
    diff = end-start
    steps = torch.abs(diff / max_step_size)
    frames = torch.max(steps)
    frames_n = int(round(frames.item(),0))

    to_add = (diff / frames)

    pos = start
    positions = []
    for _ in range(frames_n):
        pos = pos+to_add
        positions.append(pos)
    
    return positions

if __name__ == "__main__":

    FLIP_INDEXES = get_convert_indexes()

    path_file = sys.argv[1]
    model_name = sys.argv[2]
    if len(sys.argv) > 3:
        step_size = float(sys.argv[3])
    else:
        step_size = 0.0001

    if model_name == "wgs":
        method = "wgs"
    elif model_name == "gspat":
        method = "gspat"
    elif model_name == "naive":
        method = "naive"
    else:
        model = torch.load("Models/model_"+model_name+".pth", map_location=torch.device(device))
        method = ''.join([i for i in model_name if not i.isdigit()]) #Assumes format of models is <TYPE><NUMBER> eg NCNN30

    path_params  = json.load(open("Path_Gen/Paths/"+path_file+".json","r"))

    frames = path_params["points"]
    scale = {"cm":100,"mm":1000,"m":1}[ path_params["format"]]

    p = create_points(4)


    solver = {
        "wgs":wgs_wrapper,
        "gspat":gspat_wrapper,
        "naive":naive_solver_wrapper,
        "NCNN":lambda p: do_NCNN(model, p)
    }

    output_f = open("Path_gen/Outputs/"+path_file+model_name+".csv","w")

    a, b = itertools.tee(frames)
    next(b, None) #Get pairs

    rows = []
    for start,end in zip(a,b):
        start = torch.tensor(start) / scale
        end = torch.tensor(end) / scale
        start = start.T.unsqueeze_(0).to(device)
        end = end.T.unsqueeze_(0).to(device)

        positions = interpolate(start,end,step_size)
        for points in positions:
            act = solver[method](points)
            print(torch.abs(propagate(act,points)))
            act = add_lev_sig(act)
            rows.append(torch.angle(act).squeeze_())
            
    num_frames = len(rows)
    num_transducers = 512

    output_f.write(str(num_frames)+","+str(num_transducers)+"\n")
    for row in rows:
        row_flip = row[FLIP_INDEXES]
        for i,phase in enumerate(row_flip):
                output_f.write(str(phase.item()))
                if i < 511:
                    output_f.write(",")
                else:
                    output_f.write("\n")

    output_f.close()


    
    

