import os, sys
import json
import torch
import itertools
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation


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

if "-t" in sys.argv:

    phase_rows = [i.squeeze_() for i in phase_rows]

    data = torch.angle(torch.reshape(phase_rows[0],(2,16,16)))
    
    fig = plt.figure( figsize=(8,3) )
    fig.suptitle("Phases " +path_file +" " + model_name)
    ax1 = plt.subplot(1,3,1)
    im1 = plt.imshow(data[0,:],vmin=-1*torch.pi, vmax=torch.pi)
    ax2 = plt.subplot(1,3,2)
    im2 = plt.imshow(data[1,:],vmin=-1*torch.pi, vmax=torch.pi)
    cax = plt.subplot(1,30,21)
    plt.colorbar(im1,cax=cax)
    
    ax1.set_title("Board 1 Frame "+str(1))
    ax2.set_title("Board 2 Frame "+str(1))

    def animate(i):
        d = torch.angle(torch.reshape(phase_rows[i],(2,16,16)))
        # d = torch.rand((2,16,16))
        im1.set_array(d[0,:])
        im2.set_array(d[1,:])

        ax1.set_title("Top Board Frame "+str(i))
        ax2.set_title("Top Board Frame "+str(i))

        return [im1,im2]

    ani = animation.FuncAnimation(fig, animate, repeat=True,
                                    frames=len(phase_rows) - 1, interval=1)

    # plt.show()
    ani.save("Figs/TransducerPhases/transducers-"+path_file+"-"+model_name+".gif", dpi=300, writer=animation.PillowWriter(fps=1))
    print("Saved to", "Figs/TransducerPhases/transducers-"+path_file+"-"+model_name+".gif")
     


    
        