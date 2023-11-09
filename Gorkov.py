import torch
from Utilities import device, propagate, add_lev_sig
import Constants as c


def gorkov_autograd(activation, points, K1=None, K2=None, retain_graph=False):

    var_points = torch.autograd.Variable(points.data, requires_grad=True).to(device)

    B = points.shape[0]
    N = points.shape[2]
    
    if len(activation.shape) < 3:
        activation.unsqueeze_(0)    
    

    pressure = propagate(activation,var_points)

    if B > 1:
        if N > 1:
            grad_pos = torch.autograd.grad(pressure, var_points,grad_outputs=torch.ones((B,N),device=device)+1j, retain_graph=retain_graph)[0]
        else:
            grad_pos = torch.autograd.grad(pressure, var_points)[0]
    else:
        if N > 1:
            grad_pos = torch.autograd.grad(pressure, var_points,grad_outputs=torch.ones((N),device=device)+1j, retain_graph=retain_graph)[0]
        else:
            grad_pos = torch.autograd.grad(pressure, var_points)[0]
    
    if K1 is None:
        # K1 = 1/4 * c.V * (1/(c.c_0**2 * c.p_0) - 1/(c.c_p**2 * c.p_p))
        K1 = c.V / (4*c.p_0*c.c_0**2) #Assuming f1=f2=1
    if K2 is None:
        # K2 = 3/4 * c.V * ((c.p_0-c.p_p) / (c.f**2 * c.p_0 * (c.p_0+2*c.p_p)) )
        K2 = 3*c.V / (4*(2*c.f**2 * c.p_0)) #Assuming f1=f2=1


    gorkov = K1 * torch.abs(pressure) **2 - K2 * torch.sum((torch.abs(grad_pos)**2),1)
    return gorkov

def get_finite_diff_points(points, axis, stepsize = 0.000135156253):
    #points = Bx3x4
    points_h = points.clone()
    points_neg_h = points.clone()
    points_h[:,axis,:] = points_h[:,axis,:] + stepsize
    points_neg_h[:,axis,:] = points_neg_h[:,axis,:] - stepsize

    return points_h, points_neg_h


@profile
def gorkov_fin_diff(activations, points, axis="XYZ", stepsize = 0.000135156253,K1=None, K2=None):
    torch.autograd.set_detect_anomaly(True)
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]
    fin_diff_points=  torch.zeros((B,3,((2*D)+1)*N)).to(device)
    fin_diff_points[:,:,:N] = points.clone()
    
    
    if len(activations.shape) < 3:
        activations = torch.unsqueeze(activations,0).clone()

    i = 2
    if "X" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 0, stepsize)
        fin_diff_points[:,:,N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h

        i += 1

    
    if "Y" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 1, stepsize)
        fin_diff_points[:,:,(i-1)*N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h
        i += 1
    
    if "Z" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 2, stepsize)
        fin_diff_points[:,:,(i-1)*N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h
        i += 1


    pressure_points = propagate(activations, fin_diff_points)

    pressure = pressure_points[:,:N]
    pressure_fin_diff = pressure_points[:,N:]

    split = torch.reshape(pressure_fin_diff,(B,2, ((2*D))*N // 2))
    
    grad = (split[:,0,:] - split[:,1,:]) / (2*stepsize)
    
    grad = torch.reshape(grad,(B,D,N))
    grad_abs_square = torch.pow(torch.abs(grad),2)
    grad_term = torch.sum(grad_abs_square,dim=1)
    

    if K1 is None:
        # K1 = 1/4 * c.V * (1/(c.c_0**2 * c.p_0) - 1/(c.c_p**2 * c.p_p))
        K1 = c.V / (4*c.p_0*c.c_0**2) #Assuming f1=f2=1
    if K2 is None:
        # K2 = 3/4 * c.V * ((c.p_0-c.p_p) / (c.f**2 * c.p_0 * (c.p_0+2*c.p_p)) )
        K2 = 3*c.V / (4*(2*c.f**2 * c.p_0)) #Assuming f1=f2=1
    
    U = K1 * torch.abs(pressure)**2 - K2 *grad_term
    
    return U

    

if __name__ == "__main__":
    from Utilities import create_points, forward_model
    from Solvers import wgs
    
    def run():
        N =4
        B=2
        points=  create_points(N,B=B)
        print(points)
        xs = torch.zeros((B,512,1)) +0j
        for i in range(B):
            A = forward_model(points[i,:])
            _, _, x = wgs(A,torch.ones(N,1).to(device)+0j,200)
            xs[i,:] = x


        xs = add_lev_sig(xs)

        gorkov_AG = gorkov_autograd(xs,points)
        print(gorkov_AG)

        gorkov_FD = gorkov_fin_diff(xs,points,axis="XYZ")
        print(gorkov_FD)
    run()