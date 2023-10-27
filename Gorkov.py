import torch
from Utilities import device, propagate
import Constants as c


def gorkov_autograd(activation, points, K1=None, K2=None):
    var_points = torch.autograd.Variable(points.data, requires_grad=True)

    N = points.shape[2]
    
    if len(activation.shape) < 3:
        activation.unsqueeze_(0)
    
    pressure = propagate(activation,var_points)
    if N > 1:
        grad_pos = torch.autograd.grad(pressure, var_points,grad_outputs=torch.ones((N))+1j)[0]
    else:
        grad_pos = torch.autograd.grad(pressure, var_points)[0]
    
    if K1 is None:
        # K1 = 1/4 * c.V * (1/(c.c_0**2 * c.p_0) - 1/(c.c_p**2 * c.p_p))
        K1 = c.V / (4*c.p_0*c.c_0**2) #Assuming f1=f2=1
    if K2 is None:
        # K2 = 3/4 * c.V * ((c.p_0-c.p_p) / (c.f**2 * c.p_0 * (c.p_0+2*c.p_p)) )
        K2 = 3*c.V / (4*(2*c.f**2 * c.p_0))


    gorkov = K1 * torch.abs(pressure) **2 - K2 * torch.sum((torch.abs(grad_pos)**2),1)
    return gorkov


if __name__ == "__main__":
    from Utilities import create_points, forward_model
    from Solvers import wgs

    N =1
    points=  create_points(N)
    print(points)
    A = forward_model(points[0,:])
    _, _, x = wgs(A,torch.ones(N,1).to(device)+0j,200)

    gorkov = gorkov_autograd(x,points)
    print(gorkov)
