import torch
from Utilities import device, propagate
import Constants as c


def gorkov_autograd(activation, points):
    var_points = torch.autograd.Variable(points.data, requires_grad=True)
    
    if len(activation.shape) < 3:
        activation.unsqueeze_(0)
    
    pressure = propagate(activation,var_points)
    print(pressure)
    grad_pos = torch.autograd.grad(pressure, var_points,grad_outputs=torch.ones((4))+1j)[0]
    
    
    K1 = 1/4 * c.V * (1/(c.c_0**2 * c.p_0) - 1/(c.c_p**2 * c.p_p))
    K2 = 3/4 * c.V * ((c.p_0-c.p_p) / (c.f**2 * c.p_0 * (c.p_0+2*c.p_p)) )


    gorkov = K1 * torch.abs(pressure) **2 - K2 * torch.sum((torch.abs(grad_pos)**2),1)
    return gorkov


if __name__ == "__main__":
    from Utilities import create_points, forward_model
    from Solvers import wgs

    points=  create_points(4)
    A = forward_model(points[0,:])
    _, _, x = wgs(A,torch.ones(4,1).to(device)+0j,200)

    gorkov = gorkov_autograd(x,points)
    print(gorkov)
