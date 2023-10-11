import torch
from torch.utils.data import Dataset
from Utilities import *
from Solvers import wgs


class PointDataset(Dataset):
    '''
    outputs points, activations, pressures
    '''
    def __init__(self,length,N=4,seed=None):
        self.length = length #Number of point sets in the Dataset (length)
        self.N = N #Number of points per set
        self.seed = seed #custom seed
    
        self.points = []
        self.activations = []
        self.pressures = []
        print(self.length)

        for batch in range(self.length):
            points = torch.FloatTensor(3,self.N).uniform_(-.06,.06).to(device)
            A=forward_model(points, transducers()).to(device)
            _, _, x = wgs(A,torch.ones(self.N,1).to(device)+0j,200)
            pressures = A@x[:,0]
            activations = x[:,0]


            self.points.append(points)
            self.pressures.append(pressures)
            self.activations.append(activations)

            if batch % 200 == 0:
                print(batch,end=" ",flush=True)
    
    def __len__(self):
        return self.length

    def __getitem__(self,i):
         '''
         returns points, activations, pressures
         '''
         return self.points[i],self.activations[i],self.pressures[i]



if __name__ == "__main__":
    
    length = 50000
    test_length = 0
    N = 4
    dataset_type = PointDataset
    

    
    if length > 0:
            
        train = dataset_type(length)
        torch.save(train,"Datasets/Train-"+str(length)+"-"+str(N)+".pth")

    if test_length > 0:
        test = dataset_type(test_length)
        torch.save(test,"Datasets/Test-"+str(test_length)+"-"+str(N)+".pth")
    
    i = 0
    for p, a, pr in train:
        print(i)
        i += 1

