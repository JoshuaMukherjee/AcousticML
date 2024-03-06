import torch
from torch.utils.data import Dataset
from acoustools.Utilities import forward_model, transducers, device, generate_gorkov_targets, TRANSDUCERS, create_points
from acoustools.Solvers import wgs, naive_solver, gradient_descent_solver
from acoustools.Optimise.Objectives import target_gorkov_mse_objective
from acoustools.Optimise.Constraints import constrain_phase_only


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

class GorkovTargetDataset(Dataset):
    '''
    outputs points, activations, pressures, naive_act
    '''
    def __init__(self,length,N=4,seed=None):
        self.length = length #Number of point sets in the Dataset (length)
        self.N = N #Number of points per set
        self.seed = seed #custom seed
    
        self.points = []
        self.activations = []
        self.targets = []
        

        for item in range(self.length):
            points = create_points(N)
            
            

            targets = generate_gorkov_targets(self.N)
            activations = gradient_descent_solver(points,target_gorkov_mse_objective, 
                                     constrains=constrain_phase_only, lr=1e4, iters=50, targets=targets,
                                     objective_params={"no_sig":True})


            self.points.append(points.squeeze_(0))
            
            self.activations.append(activations.detach().squeeze_(0))
            
            self.targets.append(targets.squeeze_(0))

            if item % 200 == 0:
                print(item,end=" ",flush=True)
    
    def __len__(self):
        return self.length

    def __getitem__(self,i):
         '''
         returns points, activations, pressures
         '''
         return self.points[i],self.activations[i], self.targets[i]
    

class DistanceDataset(Dataset):
    '''
    Dataset of `points`, `distances`
    '''
    def __init__(self,length, N=4, board=TRANSDUCERS):
        self.length = length
        self.N = N
        self.board = board

        self.points = []
        self.distances = []

        M=board.size()[0]
        transducers = torch.unsqueeze(board,2)
        transducers = transducers.expand((-1,-1,N))

        for i in range(length):
            points = create_points(N,1).squeeze(0)

            p = torch.unsqueeze(points,0)
            p = p.expand((M,-1,-1))

            distance_axis = (transducers - p) **2
            distance = torch.sqrt(torch.sum(distance_axis,dim=1))

            self.points.append(points)
            self.distances.append(distance)

            if i % 1000 == 0:
                print(i, end = ' ')

    def __len__(self):
        return self.length

    def __getitem__(self,i):
         '''
         returns points, distances
         '''
         return self.points[i], self.distances[i]

class GreenDataset(Dataset):
    '''
    Dataset of `points`, `distances`, `green`
    '''
    def __init__(self,length, N=4, board=TRANSDUCERS):
        self.length = length
        self.N = N
        self.board = board

        self.points = []
        self.distances = []
        self.greens = []

        M=board.size()[0]
        transducers = torch.unsqueeze(board,2)
        transducers = transducers.expand((-1,-1,N))

        for i in range(length):
            points = create_points(N,1).squeeze(0)

            p = torch.unsqueeze(points,0)
            p = p.expand((M,-1,-1))

            distance_axis = (transducers - p) **2
            distance = torch.sqrt(torch.sum(distance_axis,dim=1))
            distance = torch.reshape(distance,(2*self.N,16,16))
           

            green = torch.exp(1j*726.3798*distance) / distance    
            green_ri = torch.cat((green.real,green.imag),0).to(device)   


            self.points.append(points)
            self.distances.append(distance)
            self.greens.append(green_ri)

            if i % 1000 == 0:
                print(i, end = ' ',flush=True)

    def __len__(self):
        return self.length

    def __getitem__(self,i):
         '''
         returns points, distances, green
         '''
         return self.points[i], self.distances[i], self.greens[i]



def convert_naive_to_PTD_dataset(dataset_in_path,name,N=4):
    '''
    Converts a ```NaiveDataset``` to a ```PressureTargetDataset```
    '''

    to_convert = torch.load(dataset_in_path)
    print(len(to_convert[0]))

    dataset = PressureTargetDataset(0,N)
    for points, activations, pressures, naive_act in to_convert:
       dataset.points.append(points)
       dataset.activations.append(activations)
       dataset.pressures.append(pressures)
       dataset.naive_acts.append(naive_act)
       targets = torch.FloatTensor(N).uniform_(6000,10000).to(device)
       dataset.targets.append(targets)
    
    dataset.length = to_convert.length

    torch.save(dataset,name)



if __name__ == "__main__":

    CREATE_DATASET = True

    length = 100000
    test_length = 0
    N = 4
    
    if CREATE_DATASET:
        dataset_type = GreenDataset
        

        
        if length > 0:
                
            train = dataset_type(length,N=N)
            torch.save(train,"Datasets/" +train.__class__.__name__ +"Train-"+str(length)+"-"+str(N)+".pth")

        if test_length > 0:
            test = dataset_type(test_length,N=N)
            torch.save(test,"Datasets/" +test.__class__.__name__+"Test-"+str(test_length)+"-"+str(N)+".pth")
        
        # i = 0
        # for x in train:
        #     print(i)
        #     i += 1
    
    else:
        if length > 0:
            convert_naive_to_PTD_dataset("Datasets/" +"NaiveDataset" +"Train-"+str(length)+"-"+str(N)+".pth",
                                         "Datasets/" +"PressureTargetDataset" +"Train-"+str(length)+"-"+str(N)+".pth",N)
        
        if test_length > 0:
            convert_naive_to_PTD_dataset("Datasets/" +"NaiveDataset" +"Test-"+str(test_length)+"-"+str(N)+".pth",
                                         "Datasets/" +"PressureTargetDataset" +"Test-"+str(test_length)+"-"+str(N)+".pth",N)
        
