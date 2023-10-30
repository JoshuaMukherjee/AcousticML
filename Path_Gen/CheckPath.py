import os, sys
import json
import torch
import itertools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


p = os.path.abspath('.')
sys.path.insert(1, p)

from Utilities import get_convert_indexes, propagate
from Path_Generator import interpolate
from Gorkov import gorkov_autograd

path_file = sys.argv[1]
model_name = sys.argv[2]

step_size = 0.0001

IDX = get_convert_indexes()

path_params  = json.load(open("Path_Gen/Paths/"+path_file+".json","r"))
output_f = open("Path_gen/Outputs/"+path_file+model_name+".csv","r")

frames = path_params["points"]
scale = {"cm":100,"mm":1000,"m":1}[ path_params["format"]]

a, b = itertools.tee(frames)
next(b, None) #Get pairs

positions = []
for start,end in zip(a,b):
    start = torch.tensor(start) / scale
    end = torch.tensor(end) / scale
    start = start.T.unsqueeze_(0)
    end = end.T.unsqueeze_(0)
    N = start.shape[2]

    pos = interpolate(start,end,step_size)
    positions = positions + pos


phase_rows = []
for line in output_f.readlines()[1:]:
    phases = line.split(",")
    phases = torch.tensor([float(i) for i in phases])[IDX]
    activation_out = torch.e** (1j*(torch.reshape(phases,(1,512,1))))
    phase_rows.append(activation_out)



if "-p" in sys.argv:
    pressure = []
    for _ in range(N):
        pressure.append([])
    i = 0
    for i,phase in enumerate(phase_rows):
        pos = positions[i]
        press = torch.abs(propagate(phase,pos))
        for j,p in enumerate(press):
            pressure[j].append(p)
    
    for i,p in enumerate(pressure):
        plt.plot(p,label="Point "+str(i))
    
    plt.ylim(bottom=0,top=12000)
    plt.legend()
    plt.ylabel("Pressure (Pa)")
    plt.title("Point Pressures, "+path_file+", "+ model_name)
    plt.xlabel("Frame")
    plt.show()

if "-ph" in sys.argv:
    phases = []
    for _ in range(N):
        phases.append([])
    i = 0
    for i,phase in enumerate(phase_rows):
        pos = positions[i]
        phs = torch.angle(propagate(phase,pos))
        for j,p in enumerate(phs):
            phases[j].append(p)
    
    for i,ph in enumerate(phases):
        plt.plot(ph,label="Point "+str(i))
    
    plt.ylim(bottom=-1*torch.pi, top=torch.pi)
    plt.legend()
    plt.ylabel("Phase (Rad)")
    plt.xlabel("Frame")
    plt.title("Point Phases, "+path_file+", "+ model_name)
    plt.show()

if '-g' in sys.argv:
    Us = []
    for _ in range(N):
        Us.append([])
    print(Us)
    i = 0
    for i,phase in enumerate(phase_rows):
        pos = positions[i]
        U = gorkov_autograd(phase,pos).squeeze_()

        for j,u in enumerate(U):
            Us[j].append(u)
    
    for i,u in enumerate(Us):
        u = [j.cpu().detach().numpy() for j in u]
        plt.plot(u,label="Point "+str(i))
    
    # plt.ylim(bottom=-1*torch.pi, top=torch.pi)
    plt.legend()
    plt.ylabel("U")
    plt.xlabel("Frame")
    plt.title("Gor'Kov, "+path_file+", "+ model_name)
    plt.show()
