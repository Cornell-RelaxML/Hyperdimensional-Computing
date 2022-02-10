#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import sys
sys.path.insert(0, '../')
import scipy.stats
    
class LinearEncoder:
    def __init__(self, dim=10000, num=256):
        self.dim = dim
        self.num = num
    
    def get_hdv(self, dim=10000, num=1):
        assert num > 0, "[Error] Only support nonzero size in get_hdv()"
        if num == 1:
            # sample [-1,1] with uniform distribution
            return np.random.choice([-1,1], size=dim)
        else:
            result = np.random.randint(2, size=(num, dim))
            result = (result-0.5)*2
            return result.astype('int')
    
    def build_item_mem(self):
        assert self.num>1, "No need of this function if only one vector in the item memory."
        print("generating linear item memory...")
        item_mem = self.get_hdv(dim=self.dim, num=self.num)
        # generate the rest of the pixels based on the absolute distances among pixels.
        index = np.arange(self.dim//2)
        np.random.shuffle(index)
        interval = int((self.dim/2)/(self.num-1))
        pointer = 0
        for i in range(1,self.num):
            new_item = np.copy(item_mem[i-1])
            if i == self.num-1:
                new_item[index[pointer:]] *= -1
            else:
                new_item[index[pointer: pointer+interval]] *= -1
            pointer += interval
            item_mem[i] = new_item
        self.item_mem = torch.from_numpy(item_mem)
        return self.item_mem # {-1, 1}

    def encode_one_img(self, x):
        '''
        x: input image vector. shape=[#pixels,]
        item_mem: hd features for each pixel. shape=[#pixel values, HD dim]
        '''
        rv = self.item_mem[x[0]]
        for i in range(1, x.shape[0]):
            rv = torch.roll(rv, i)
            rv = -rv * self.item_mem[x[i]]
        return rv
#         return rv > 0 # store {0,1} vectors to reduce disk storage
    
    def encode_data_extract_labels(self, datast):
        n = len(datast) # number of examples in x
        rv = torch.zeros((n, self.dim))
        labels = torch.zeros(n).long()
        print('start encoding data here')
        for i in range(n):
            rv[i] = self.encode_one_img((255 * datast[i][0].view(-1)).int()) # transform to [0, 255]
            labels[i] = datast[i][1]
            if (i % 1000 == 999): print("{} images encoded".format(i+1))
        print('finish encoding data here')
        return rv, labels
    

class RandomFourierEncoder:
    def __init__(self, input_dim, gamma, binary = True, gorder=2, output_dim=10000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.binary = binary
        # kernel parameter
        self.gamma = gamma
        if binary:
            self.gorder = 2
            print('Binary HDC, set gorder to 2')
        else:
            self.gorder = gorder
        
    def pts_map(self, x, r=1.0):
        theta = 2.0 * np.pi / (1.0*self.gorder) * x
        pts = r * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
        return pts
        
    def GroupRFF(self, x, sigma):
        intervals = sigma * torch.tensor([scipy.stats.norm.ppf(i * 1.0/self.gorder) for i in range(1, self.gorder)])
        print('the threshold to discretize fourier features to group elements', intervals)
        group_index = torch.zeros_like(x)
        group_index[x <= intervals[0]] = 0
        group_index[x > intervals[-1]] = self.gorder - 1
        if self.gorder > 2:
            for i in range(1, self.gorder - 1):
                group_index[(x > intervals[i-1]) & (x <= intervals[i])] = i
        return group_index
    
    def build_item_mem(self): # create random fourier features for 256 pixel values
        # a correction factor for bias
        correction_factor = 1/1.4
        # covariance kernel
        x = np.linspace(0, 255, num=256)
        Cov = np.array([np.exp(-correction_factor * self.gamma**2 * ((x - y)/255.0)**2 / 2) for y in range(256)])
        k = Cov.shape[0]
        assert Cov.shape[1] == k, "Cov is not a square matrix."
        L = np.sin(Cov * np.pi / 2.0)
        ''' Eigen decomposition: L = eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T '''
        eigen_values, eigen_vectors = np.linalg.eigh(L)
        R = eigen_vectors @ np.diag(np.maximum(0, eigen_values)**0.5) @ eigen_vectors.T        
        item_mem = torch.from_numpy(np.random.randn(self.output_dim, k) @ R)
        if self.binary:
            self.item_mem = (item_mem >= 0).T *1.0
        else:
            self.item_mem = self.GroupRFF(item_mem, np.sqrt((R ** 2).sum(0).max())).T *1.0
        return self.item_mem
    
    def encode_one_img(self, x):
        '''
        x:   flattened input image. size=[#pixels,]
        rfs: random feature vectors for pixel values. size=[HDC Dim, #features]
        '''
        rv = self.item_mem[x[0]]
        for i in range(1, x.size(0)):
            rv = torch.roll(rv, i) # rv = np.roll(rv, 1) also works
            if self.binary:
                rv = torch.logical_xor(rv, self.item_mem[x[i]])
            else:
                rv = self.group_bind([rv, self.item_mem[x[i]]])
        return rv
    
    # returns an array of HD features for multiple inputs, together with label list
    def encode_data_extract_labels(self, datast):
        '''
        datast:   trainset or testset loaded via torch. tuple style, contains N (x,y) pair.
        rfs: random feature vectors for pixel values. shape=[HDC Dim, #features]
        return: rv -> hypervectors for images. shape=[N, HDC dim]
        '''
        channels = datast[0][0].size(0)
        n = len(datast) # number of examples in x
        rv = torch.zeros((n, channels * self.output_dim))
        labels = torch.zeros(n).long()
        print('Start encoding data')
        for i in range(n):
            for channel_id in range(channels):
                rv[i, channel_id*self.output_dim:(channel_id+1)*self.output_dim] = self.encode_one_img((255 * datast[i][0][channel_id].view(-1)).int()) # transform to [0, 255]
            labels[i] = datast[i][1]
            if (i % 1000 == 999): print("{} images encoded".format(i+1))
        print('Finish encoding data')
        return rv, labels
    
    def group_bind(self, lst):
        if len(lst)>300:
            results = lst[0]
            for ite in lst[1:]:
                results += ite
        else:
            results = torch.sum(torch.stack(lst), dim=0)
        return results  # torch.fmod(results, self.gorder) # mathematically same 

    def group_bundle(self, lst):
        intervals = torch.tensor([2*np.pi/self.gorder * i for i in range(self.gorder)]) + np.pi/self.gorder
        if len(lst)>300:
            pts = self.pts_map(lst[0])
            for ite in lst[1:]:
                pts += self.pts_map(ite)
        else:
            pts = torch.sum(self.pts_map(torch.stack(lst)), dim=0)
        raw_angles = 2*np.pi + torch.arctan(pts[:,1] / pts[:,0]) - np.pi * (pts[:,0]<0).float()
        angles = torch.fmod(raw_angles, 2*np.pi)
        return torch.floor(angles/(2.0*np.pi) * self.gorder + 1/2)   # torch.fmod( , self.gorder)
    
    def similarity(self, x, y):
        return torch.sum(torch.sum(self.pts_map(x) * self.pts_map(y), dim=-1), dim=-1) * (1.0/x.size(-1))
