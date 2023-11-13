import torch
from torch.utils.data import Dataset
from Utilities import forward_model, transducers, device
from Solvers import wgs, naive_solver


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

class FDataset(Dataset):
    '''
    outputs F, points, activations, pressures
    '''
    def __init__(self,length,N=4,seed=None):
        self.length = length #Number of point sets in the Dataset (length)
        self.N = N #Number of points per set
        self.seed = seed #custom seed
    
        self.points = []
        self.activations = []
        self.pressures = []
        self.forwards = []
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
            self.forwards.append(A)

            if batch % 200 == 0:
                print(batch,end=" ",flush=True)
    
    def __len__(self):
        return self.length

    def __getitem__(self,i):
         '''
         returns F, points, activations, pressures
         '''
         return self.forwards[i], self.points[i],self.activations[i],self.pressures[i]
    
class FDatasetNorm(Dataset):
    '''
    outputs F, points, activations, pressures
    '''
    def __init__(self,length,N=4,seed=None):
        self.length = length #Number of point sets in the Dataset (length)
        self.N = N #Number of points per set
        self.seed = seed #custom seed
    
        self.points = []
        self.activations = []
        self.pressures = []
        self.forwards = []
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
            self.forwards.append(A)

            if batch % 200 == 0:
                print(batch,end=" ",flush=True)
    
    def __len__(self):
        return self.length

    def __getitem__(self,i):
         '''
         returns F, points, activations, pressures
         '''
         return torch.nn.functional.normalize(self.forwards[i]), self.points[i],self.activations[i],self.pressures[i]

class NaiveDataset(Dataset):
    '''
    outputs points, activations, pressures, naive_act
    '''
    def __init__(self,length,N=4,seed=None):
        self.length = length #Number of point sets in the Dataset (length)
        self.N = N #Number of points per set
        self.seed = seed #custom seed
    
        self.points = []
        self.activations = []
        self.pressures = []
        self.naive_acts = []
        print(self.length)

        for batch in range(self.length):
            points = torch.FloatTensor(3,self.N).uniform_(-.06,.06).to(device)
            naive_p, naive_act = naive_solver(points)
            A=forward_model(points, transducers()).to(device)
            _, _, x = wgs(A,torch.ones(self.N,1).to(device)+0j,200)
            pressures = A@x[:,0]
            activations = x[:,0]


            self.points.append(points)
            self.pressures.append(pressures)
            self.activations.append(activations)
            self.naive_acts.append(naive_act)

            if batch % 200 == 0:
                print(batch,end=" ",flush=True)
    
    def __len__(self):
        return self.length

    def __getitem__(self,i):
         '''
         returns points, activations, pressures
         '''
         return self.points[i],self.activations[i],self.pressures[i], self.naive_acts[i]
        
class PressureTargetDataset(Dataset):
    '''
    outputs points, activations, pressures, naive_act
    '''
    def __init__(self,length,N=4,seed=None):
        self.length = length #Number of point sets in the Dataset (length)
        self.N = N #Number of points per set
        self.seed = seed #custom seed
    
        self.points = []
        self.activations = []
        self.pressures = []
        self.naive_acts = []
        self.targets = []
        print(self.length)

        for batch in range(self.length):
            points = torch.FloatTensor(3,self.N).uniform_(-.06,.06).to(device)
            naive_p, naive_act = naive_solver(points)
            A=forward_model(points, transducers()).to(device)
            _, _, x = wgs(A,torch.ones(self.N,1).to(device)+0j,200)
            pressures = A@x[:,0]
            activations = x[:,0]

            targets = torch.FloatTensor(self.N).uniform_(6000,10000).to(device)


            self.points.append(points)
            self.pressures.append(pressures)
            self.activations.append(activations)
            self.naive_acts.append(naive_act)
            self.targets.append(targets)

            if batch % 200 == 0:
                print(batch,end=" ",flush=True)
    
    def __len__(self):
        return self.length

    def __getitem__(self,i):
         '''
         returns points, activations, pressures
         '''
         return self.points[i],self.activations[i],self.pressures[i], self.naive_acts[i], self.targets[i]



if __name__ == "__main__":
    
    length = 4
    test_length = 2 
    N = 4
    dataset_type = PressureTargetDataset
    

    
    if length > 0:
            
        train = dataset_type(length,N=N)
        torch.save(train,"Datasets/" +train.__class__.__name__ +"Train-"+str(length)+"-"+str(N)+".pth")

    if test_length > 0:
        test = dataset_type(test_length,N=N)
        torch.save(test,"Datasets/" +test.__class__.__name__+"Test-"+str(test_length)+"-"+str(N)+".pth")
    
    i = 0
    for x in train:
        print(i)
        i += 1

