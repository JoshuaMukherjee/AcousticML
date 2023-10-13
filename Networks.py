import torch
from torch.nn import Module
import torch.nn as nn
import torchvision.transforms.functional as TF

'''
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019).
PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). 
Curran Associates, Inc. 
Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
'''

from Symmetric_Functions import SymMax
from Utilities import device

import math

import Activations
import Output_Funtions

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")



class PointNet(Module):
    '''
    Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. 2016. PointNet: Deep
    Learning on Point Sets for 3D Classification and Segmentation. (12 2016). 
    http://arxiv.org/abs/1612.00593
    '''

    def __init__(self, layer_sets,input_size=3, 
                    activation=torch.nn.ReLU, 
                    kernel=1, kernel_pad="same",padding_mode="zeros",
                    batch_norm=None,batch_args={},
                    sym_function=SymMax, sym_args={},
                    output_funct=None, output_funct_args = {}
                ):
        super(PointNet,self).__init__()
        self.layers = torch.nn.ModuleList()
        self.sym_function = sym_function(**sym_args)
        if output_funct is not None:
            self.output_funct = getattr(Output_Funtions,output_funct)(**output_funct_args)
        else:
            self.output_funct = None
        
        Group_norm= False
        if type(batch_norm) == str:
            if batch_norm == "GroupNorm":
                Group_norm = True
                groups = batch_args["groups"]
                batch_args.pop("groups")
            batch_norm = getattr(torch.nn,batch_norm)
        
        if type(activation) == str:
            activation = getattr(torch.nn,activation) 
        


        local_features = layer_sets[0][-1]
        for i,layer_set in enumerate(layer_sets):
            self.layers.append(torch.nn.ModuleList())
            for j,layer in enumerate(layer_set):
                if i == 0 and j == 0:
                    in_channels = input_size
                    out_channels = layer
                elif j != 0:
                    in_channels = layer_set[j-1]
                    out_channels = layer
                elif i == 2 and j == 0:
                    in_channels = layer_sets[i-1][-1]+local_features
                    out_channels = layer
                else:
                    in_channels = layer_sets[i-1][-1]
                    out_channels = layer
                
                mod = torch.nn.Conv1d(in_channels,out_channels,kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode).to(device)
                
                self.layers[i].append(mod)
                
                if type(activation) is not list:
                    self.layers[i].append(activation().to(device))
                else:
                    act = activation[i][j]
                    if act is not None:
                        self.layers[i].append(act().to(device))
                
                if batch_norm is not None:
                    if type(batch_norm) is not list:
                        if Group_norm:
                            norm=  batch_norm(groups,out_channels,**batch_args).to(device)
                        else:
                            norm = batch_norm(out_channels,**batch_args).to(device)
                        self.layers[i].append(norm)
                    else:
                        norm = batch_norm[i][j]
                        if norm is not None:
                            if Group_norm:
                                norm_layer=  batch_norm(groups,out_channels,**batch_args).to(device)
                            else:
                                norm_layer = batch_norm(out_channels,**batch_args).to(device)
                            self.layers[i].append(norm_layer)
            
            
        # print(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers[0]:
            out = layer(out)
            # print(out)
        local_features = out
        for layer in self.layers[1]:
            out = layer(out)
        
        out = self.sym_function(out)
        N = x.shape[2]
        global_features = torch.Tensor.expand(out.unsqueeze_(2),-1,-1,N)
        out = torch.cat((local_features,global_features),dim=1)
        for layer in self.layers[2]:
            out = layer(out)
        
        
        if self.output_funct is not None:
            out = self.output_funct(out)
        

        return out

class Identity(Module):
    def __init__(self):
        super(Identity,self).__init__()
    
    def forward(self,x):
        return x

class MLP(Module):
    def __init__(self, layers, input_size=512,layer_args={},
                    activation=torch.nn.SELU, batch_norm=None, batch_args={},batch_channels=2,batch_old=False):
        super(MLP,self).__init__()
        self.layers = torch.nn.ModuleList()

        
        in_channels= input_size
        out_channels = layers[0]

        Group_norm = False
        if type(batch_norm) == str:
            if batch_norm == "GroupNorm":
                Group_norm = True
                groups = batch_args["groups"]
                batch_args.pop("groups")
            batch_norm = getattr(torch.nn,batch_norm)
        

        self.layers.append(torch.nn.Linear(in_channels,out_channels,**layer_args).to(device))
        
        if type(activation) is not list and activation is not None:
            if type(activation) == str:
                try:
                    activation = getattr(torch.nn,activation) 
                except AttributeError:
                    activation = getattr(Activations,activation) 
            self.layers.append(activation().to(device))
        elif type(activation) is list :
            if type(activation[0]) == str:
                try:
                    act = getattr(torch.nn,activation[0]) 
                except AttributeError:
                    act = getattr(Activations,activation[0]) 
            self.layers.append(act().to(device))

        norm_layer = None
        if type(batch_norm) is not list and batch_norm is not None:
            if batch_old:
                channel = batch_channels
            else:
                channel = out_channels
            if Group_norm:
                norm_layer=  batch_norm(groups,channel,**batch_args).to(device)
            else:
                norm_layer=  batch_norm(channel,**batch_args).to(device)
           
        elif type(batch_norm) is list :
            norm_layer = batch_norm[0](channel,**batch_args).to(device)
        
        if norm_layer is not None:
            self.layers.append(norm_layer)


        for i,layer in enumerate(layers[1:]):
            in_channels = layers[i] #As starting from [1:] in layers i will be actually one off from position
            out_channels = layer
            self.layers.append(torch.nn.Linear(in_channels,out_channels,**layer_args).to(device))

            if type(activation) is not list and activation is not None:
                if type(activation) == str:
                    try:
                        activation = getattr(torch.nn,activation) 
                    except AttributeError:
                        activation = getattr(Activations,activation) 

                self.layers.append(activation().to(device))
            elif type(activation) is list :
                if type(activation[i+1]) == str:
                    try:
                        activation = getattr(torch.nn,activation[i+1]) 
                    except AttributeError:
                        activation = getattr(Activations,activation[i+1]) 
                
                self.layers.append(activation().to(device))
        

            norm_layer = None
            if type(batch_norm) is not list and batch_norm is not None:
                if batch_old:
                    channel = batch_channels
                else:
                    channel = out_channels
                if Group_norm:
                    norm_layer=  batch_norm(groups,channel,**batch_args).to(device)
                else:
                    norm_layer=  batch_norm(channel,**batch_args).to(device)
            
            elif type(batch_norm) is list :
                norm_layer = batch_norm[i+1](channel,**batch_args).to(device)
            
            if norm_layer is not None:
                self.layers.append(norm_layer)
            
    def forward(self,x):
        out = x
        # print()
        for layer in self.layers:
            out = layer(out)
        return out

class ResBlock(Module):
    '''
    Adapted From
    Li, B., Zhang, Y. & Sun, F. 
    Deep residual neural network based PointNet for 3D object part segmentation. 
    Multimed Tools Appl 81, 11933–11947 (2022). 
    https://doi.org/10.1007/s11042-020-09609-8
    '''
    def __init__(self, D, D1, D2, 
                kernel=1, kernel_pad="same",padding_mode="zeros",
                activation = None, norm = None):
        super(ResBlock, self).__init__()

        if type(norm) == str:
            norm = getattr(torch.nn,norm)
        
        if type(activation) == str:
            activation = getattr(torch.nn,activation) 

        self.block1 = torch.nn.Conv1d(D , D1, kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode).to(device)
        
        if activation is not None:
            self.act = activation().to(device)
        else:
            self.act = None
        
        if norm is not None:
            self.block1_norm = norm(D1)
        else:
            self.block1_norm = None

        self.block2 = torch.nn.Conv1d(D1, D2, kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode).to(device)
    

        self.block3 = torch.nn.Conv1d(D , D2, kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode).to(device)
        if norm is not None:
            self.block3_norm =  norm(D2)
        else:
            self.block3_norm = None
        
    
    def forward(self,x):
        x1 = self.block1(x)
        if self.act is not None:
            x1 = self.act(x1)
        if self.block1_norm is not None:
            x1 = self.block1_norm(x1)

        x1 = self.block2(x1)
        x = self.block3(x)
        if self.block3_norm is not None:
            x = self.block3_norm(x)

        x = x+x1
        if self.act is not None:
            x = self.act(x)

        return x

class ResPointNet(Module):
    '''
    Adapted From
    Li, B., Zhang, Y. & Sun, F. 
    Deep residual neural network based PointNet for 3D object part segmentation. 
    Multimed Tools Appl 81, 11933–11947 (2022). 
    https://doi.org/10.1007/s11042-020-09609-8
    '''
    def __init__(self, layer_sets, input_size=3,
                kernel=1, kernel_pad="same",padding_mode="zeros",
                activation = None, batch_norm = None,
                sym_function=SymMax, sym_args={},
                output_funct=None, output_funct_args = {}):
        super(ResPointNet,self).__init__()

        self.sym_function = sym_function(**sym_args)


        self.blocks = torch.nn.ModuleList()
        D = input_size
        assert(len(layer_sets) == 3)

        local_features = layer_sets[0][-1]

        for layer_i,layer in enumerate(layer_sets):
            self.blocks.append(torch.nn.ModuleList())
            for i in range(0,len(layer),2):
                D1 = layer[i]
                D2 = layer[i+1]
                if layer_i == 2 and i ==0:
                    D += local_features
                block = ResBlock(D,D1,D2,  kernel=kernel, kernel_pad=kernel_pad,padding_mode=padding_mode,  activation = activation, norm = batch_norm)
                self.blocks[layer_i].append(block)
                D = D2
        
        if output_funct is not None:
            self.output_funct = getattr(Output_Funtions,output_funct)(**output_funct_args)
        else:
            self.output_funct = None
                
    def forward(self,x):
        out = x
        for layer in self.blocks[0]:
            out = layer(out)
        local_features = out
        for layer in self.blocks[1]:
            out = layer(out)
        
        out = self.sym_function(out)
        N = x.shape[2]
        global_features = torch.Tensor.expand(out.unsqueeze_(2),-1,-1,N)
        out = torch.cat((local_features,global_features),dim=1)
        for layer in self.blocks[2]:
            out = layer(out)
        
        if self.output_funct is not None:
            out = self.output_funct(out)
        

        return out

class CNN(Module):
    def __init__(self,layers, conv_args= {}, 
                 channels_in = 4,
                 activation=None,
                 norm=None):
        
        super(CNN,self).__init__()
        self.layer_list = layers
        self.conv_args = conv_args
        self.out_size = 0

        self.channels_in = channels_in

        if activation is not None:
            try:
                self.activation = getattr(torch.nn,activation)()
            except AttributeError:
                self.activation = getattr(Activations,activation)()
    
        if norm is not None:
            self.norm = getattr(torch.nn,norm)
       
        self.layers = torch.nn.ModuleList()
       
        in_size = self.channels_in
        N = len(self.layers)
        for i,layer in enumerate(self.layer_list):
            self.out_size = layer
            self.layers.append(torch.nn.Conv2d(in_size,self.out_size,**self.conv_args).to(device))
            if activation is not None:
                self.layers.append(self.activation.to(device))
            if norm is not None:
                self.layers.append(self.norm(self.out_size).to(device))
            in_size = self.out_size
        

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
    
class F_CNN(Module):
    def __init__(self, CNN_args={},T=512, N=4, num_boards=2, sym_fun = torch.max, cnn=CNN ):
        super(F_CNN,self).__init__()
        self.T = T
        self.N = N #No. Points
        self.num_boards = num_boards
        self.cnn = cnn(**CNN_args)
        self.sym_fun = sym_fun
    
    def forward(self, x):

        # print(x.shape) #BxNx512
        B = x.shape[0]
        W = int(math.sqrt(x.shape[2] / self.num_boards))

        x = torch.reshape(x,(self.N,B,self.num_boards, W, W)) #NxBx2x16x16 -> 2 complex boards per batch per point
        # print(x.shape)
        # print(x[0,0,0,0,0:6])
        x = torch.view_as_real(x) #NxBx2x16x16x2 -> split real and complex images
        ''' Plot boards
        img1Re = x[0,0,0,:,:,0]
        img2Re = x[0,0,1,:,:,0]
        img1Im = x[0,0,0,:,:,1]
        img2Im = x[0,0,1,:,:,1]

        plt.subplot(2,2,1)
        plt.imshow(img1Re)
        plt.title("Board 1 Re")

        plt.subplot(2,2,2)
        plt.imshow(img2Re)
        plt.title("Board 2 Re")

        plt.subplot(2,2,3)
        plt.imshow(img1Im)
        plt.title("Board 1 Im")

        plt.subplot(2,2,4)
        plt.imshow(img2Im)
        plt.title("Board 2 Im")

        plt.show()

        input()
        '''

        # print(x.shape)
        x = torch.concat((x[:,:,:,:,:,0],x[:,:,:,:,:,1]),dim=2) #NxBx4x16x16 -> combine into channels
        # x = torch.reshape(x, (self.N,B,2*self.num_boards, W, W))
        out = torch.zeros((self.N,B, self.cnn.out_size, W, W)).to(device)
        for n,point in enumerate(x):
            p = self.cnn(point) #NxBx2x16x16 -> Run CNN's on each point
            out[n,:] = p 
        # print(out.shape)

        out = self.sym_fun(out,dim=0).values #Bx2x16x16 -> Pool 
        # print(out.shape)
        out = torch.reshape(out, (B,-1, 1)) #Bx1x512 -> hologram
        # print(out.shape)

        out = torch.exp(1j * out)
        # print(out.shape)

        return out



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__ (self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)



if __name__ == "__main__":

    from Utilities import forward_model, transducers
    from Solvers import wgs

    B = 2
    C = 4

    points1 = torch.FloatTensor(3,4).uniform_(-.06,.06).to(device)
    A1=forward_model(points1, transducers()).to(device).unsqueeze_(0)
    points2 = torch.FloatTensor(3,4).uniform_(-.06,.06).to(device)
    A2=forward_model(points2, transducers()).to(device).unsqueeze_(0)
    

    in_A = torch.concat((A1,A2),0) #add batch
    # print(A[0,:,0:6])


    layers = [4,10,10,10,2]
    args = {
        "kernel_size":3,
        "padding":"same"
    }

    F_cnn_args = {
        "layers":layers,
        "channels_in":C,
        "conv_args":args,
        "activation":"ReLU",
        "norm":"BatchNorm2d"
    }

    F_cnn = F_CNN(F_cnn_args)
    out = F_cnn(in_A)
    # print(out.shape)
    # _, _, x = wgs(A1,torch.ones(4,1).to(device)+0j,200)
    # print(x.shape)

    # p_wgs = A1@x
    # print(torch.abs(p_wgs).shape)
    p_cnn = in_A@out
    print(torch.abs(p_cnn))
    
