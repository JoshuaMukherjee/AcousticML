from Utilities import *
import torch

def wgs(A, b, K):
    #Written by Giorgos Christopoulos 2022
    AT = torch.conj(A).T.to(device)
    b0 = b.to(device)
    x = torch.ones(A.shape[1],1).to(device) + 0j
    for kk in range(K):
        y = torch.matmul(A,x)                                   # forward propagate
        y = y/torch.max(torch.abs(y))                           # normalize forward propagated field (useful for next step's division)
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))     # update target - current target over normalized field
        b = b/torch.max(torch.abs(b))                           # normalize target
        p = torch.multiply(b,torch.divide(y,torch.abs(y)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram  
                    
    return y, p, x



def gspat(R,forward, backward, target, iterations):
    #Written by Giorgos Christopoulos 2022
    field = target 

    for _ in range(iterations):
        
#     amplitude constraint, keeps phase imposes desired amplitude ratio among points     
        target_field = torch.multiply(target,torch.divide(field,torch.abs(field)))  
#     backward and forward propagation at once
        field = torch.matmul(R,target_field)
#     AFTER THE LOOP
#     impose amplitude constraint and keep phase, after the iterative part this step is different following Dieg
    target_field = torch.multiply(target**2,torch.divide(field,torch.abs(field)**2))
#     back propagate 
    complex_hologram = torch.matmul(backward,target_field)
#     keep phase 
    phase_hologram = torch.divide(complex_hologram,torch.abs(complex_hologram))
    points = torch.matmul(forward,phase_hologram)

    return phase_hologram, points


def naive(points):
    activation = torch.ones(points.shape[1]) +0j
    forward = forward_model(points.T,transducers())
    back = torch.conj(forward).T
    trans = back@activation
    trans_phase=  trans / torch.abs(trans)
    out = forward@trans_phase
    pressure = torch.abs(out)
    return out, pressure

def ph_thresh(z_last,z,threshold):

    pi = torch.pi
    ph1 = torch.angle(z_last)
    ph2 = torch.angle(z)
    dph = ph2 - ph1
    
    dph = torch.atan2(torch.sin(dph),torch.cos(dph))    
    # print()
    # dph[dph>pi] = dph[dph>pi] - 2*pi
    # print((dph<-1*pi).any())
    # dph[dph<-1*pi] = dph[dph<-1*pi] + 2*pi
    # print((dph<-1*pi).any())
    
    dph[dph>threshold] = threshold
    dph[dph<-1*threshold] = -1*threshold
    
    # dph = torch.clamp(dph, -1*threshold, threshold)
    
    
    ph2 = ph1 + dph
    z = abs(z)*torch.exp(1j*ph2)
    
    return z

def soft(x,threshold):
    y = torch.max(torch.abs(x) - threshold,0).values
    y = y * torch.sign(x)
    return y

def ph_soft(x_last,x,threshold):
    pi = torch.pi
    ph1 = torch.angle(x_last)
    ph2 = torch.angle(x)
    dph = ph2 - ph1

    dph[dph>pi] = dph[dph>pi] - 2*pi
    dph[dph<-1*pi] = dph[dph<-1*pi] + 2*pi

    dph = soft(dph,threshold)
    ph2 = ph1 + dph
    x = abs(x)*torch.exp(1j*ph2)
    return x


def temporal_wgs(A, y, K,ref_in, ref_out,T_in,T_out):
    '''
    Based off 
    Giorgos Christopoulos, Lei Gao, Diego Martinez Plasencia, Marta Betcke, 
    Ryuji Hirayama, and Sriram Subramanian. 2023. 
    Temporal acoustic point holography.(under submission) (2023)
    '''
    #ref_out -> points
    #ref_in-> transducers
    AT = torch.conj(A).T.to(device)
    y0 = y.to(device)
    x = torch.ones(A.shape[1],1).to(device) + 0j
    for kk in range(K):
        z = torch.matmul(A,x)                                   # forward propagate
        z = z/torch.max(torch.abs(z))                           # normalize forward propagated field (useful for next step's division)
        z = ph_thresh(ref_out,z,T_out); 
        
        y = torch.multiply(y0,torch.divide(y,torch.abs(z)))     # update target - current target over normalized field
        y = y/torch.max(torch.abs(y))                           # normalize target
        p = torch.multiply(y,torch.divide(z,torch.abs(z)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram    
        x = ph_thresh(ref_in,x,T_in);    
    return y, p, x


def compare_phase():
    from Dataset import TimeDatasetAtomic
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('tkagg')

    N = 200
    movement = 0.0001
    dataset = TimeDatasetAtomic(1,N,movement=movement)
    pi = torch.pi
    for points, changes, activations, pressures in dataset:
       
        point = points[0]
        A=forward_model(point).to(device)
        _, _, x_wgs = wgs(A,torch.ones(4,1).to(device)+0j,200)
        z_wgs = A@x_wgs

        x_temp = x_wgs
        z_temp = z_wgs

        temp_phases = [torch.mean(torch.angle(x_temp))]
        wgs_phases = [torch.mean(torch.angle(x_wgs))]

        wgs_pressure = []
        temp_pressure = []
                
        T_in = pi/64
        T_out = 0

        for i in range(1,len(points)):
            point = points[i]
            A=forward_model(point).to(device)


            _, _, x_temp = temporal_wgs(A,torch.ones(4,1).to(device)+0j,200,x_temp,z_temp,T_in,T_out)
            z_temp = A@x_temp
            # print(torch.abs(z_temp))
            temp_phases.append(torch.mean(torch.angle(x_temp)))
            temp_pressure.append(torch.mean(torch.abs(z_temp)))


            _, _, x_wgs = wgs(A,torch.ones(4,1).to(device)+0j,200)
            wgs_phases.append(torch.mean(torch.angle(x_wgs)))
            wgs_pressure.append(torch.mean(torch.abs(A@x_wgs)))
            
        temp_phases = [torch.abs(torch.atan2(torch.sin(i),torch.cos(i))) for i in temp_phases]
        wgs_phases = [torch.abs(torch.atan2(torch.sin(i),torch.cos(i))) for i in wgs_phases]

        temp_changes = []
        for i in range(1,len(temp_phases)):
            temp_changes.append(temp_phases[i] - temp_phases[i-1])

        wgs_changes = []
        for i in range(1,len(temp_phases)):
            wgs_changes.append(wgs_phases[i] - wgs_phases[i-1])

        ax = plt.subplot(2,1,1)
        ax.plot(temp_changes,label="Temporal")
        ax.plot(wgs_changes,label="WGS")
        ax.set_ylabel("Phase Change (rads)")
        ax.set_xlabel("Frame")

        ax.plot(torch.linspace(0,N,N),torch.ones(N)*T_in,label="T_in limit = pi/" + str(int(pi/T_in)),color="green")
        ax.plot(torch.linspace(0,N,N),torch.ones(N)*-1*T_in,color="green")
        # plt.yticks(torch.linspace(0,2*pi,10))
        ax.legend()

        ax = plt.subplot(2,1,2)
        ax.plot(temp_pressure,label="Temporal")
        ax.plot(wgs_pressure,label="WGS")
        ax.set_ylabel("Mean Pressure (Pa)")
        ax.set_xlabel("Frame")
        ax.legend()
        plt.show()


if __name__ == "__main__":
   compare_phase()
        