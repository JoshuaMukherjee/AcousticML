import torch
from acoustools.Utilities import device


def add_sine_points(points, extra_points_per_wave=2, direction = 2,distance=0.0084/4):
    '''
    extra_points_per_wave: Must be even
    '''
    # Probably not the most efficient
    # Bx3xN -> Bx3x(N*points_per_wave+N)
    B,P,N = points.shape
    # print(B,N,P)
    out = torch.ones((B,P,N*extra_points_per_wave+N)).to(device)
    for i in range(0,N):
        out[:,:,i] = points[:,:,i]
        mul = 1
        for j in range(0,extra_points_per_wave,2):
            ind = i*extra_points_per_wave+N + j
            
            out[:,:,ind] = points[:,:,i]
            out[:,direction,ind] += distance*mul

            ind += 1
            out[:,:,ind] = points[:,:,i]
            out[:,direction,ind] -= distance*mul

            mul += 1
                
    return out

if __name__ == "__main__":
    from Dataset import PointDataset
    from torch.utils.data import DataLoader 

    dataset = PointDataset(2,4)
    data = iter(DataLoader(dataset,1,shuffle=True))

    p,a,pr = next(data)
    print(p)
    out = add_sine_points(p)
    print(out)

