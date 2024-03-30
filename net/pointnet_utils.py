'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tnet(nn.Module):
   """
   Introduce rotation invariance to the pointcloud. Takes in a pointcloud and 
   returns a rotation matrix to nullify the rotation
   Reference: https://arxiv.org/abs/1612.00593
   """
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      # initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()

      # add identity to the output
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class GlobalFeatureVectorModule(nn.Module):
   """
   Returns for an input point cloud
   1. global feature vector before flattening 
   2. low level feature layer  
   Reference: https://arxiv.org/abs/1612.00593
   """
   def __init__(self, k=6):
      super().__init__()

      self.k=k

      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,64,1)
      self.conv3 = nn.Conv1d(64,64,1)
      self.conv4 = nn.Conv1d(64,128,1)
      self.conv5 = nn.Conv1d(128,1024,1)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(64)
      self.bn3 = nn.BatchNorm1d(64)
      self.bn4 = nn.BatchNorm1d(128)
      self.bn5 = nn.BatchNorm1d(1024)
      
      nn.init.xavier_uniform_(self.conv1.weight)
      nn.init.xavier_uniform_(self.conv2.weight)
      nn.init.xavier_uniform_(self.conv3.weight)
      nn.init.xavier_uniform_(self.conv4.weight)
      nn.init.xavier_uniform_(self.conv5.weight)

   def forward(self, input):
      n_pts = input.size()[2]

      out1 = self.bn1(F.leaky_relu(self.conv1(input), negative_slope=0.01))
      out2 = self.bn2(F.leaky_relu(self.conv2(out1), negative_slope=0.01))
      out3 = self.bn3(F.leaky_relu(self.conv3(out2), negative_slope=0.01))
      out4 = self.bn4(F.leaky_relu(self.conv4(out3), negative_slope=0.01))
      out5 = self.conv5(out4)

      out6 = nn.MaxPool1d(out5.size(-1))(out5)

      return out6, out2, n_pts


class LWGlobalFeatureVectorModule(nn.Module):
   def __init__(self, k=6):
      super().__init__()

      self.k=k

      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      
      nn.init.xavier_uniform_(self.conv1.weight)
      nn.init.xavier_uniform_(self.conv2.weight)
      nn.init.xavier_uniform_(self.conv3.weight)

   def forward(self, input):
      n_pts = input.size()[2]

      out1 = self.bn1(F.leaky_relu(self.conv1(input), negative_slope=0.01))
      out2 = self.bn2(F.leaky_relu(self.conv2(out1), negative_slope=0.01))
      out3 = self.bn3(F.leaky_relu(self.conv3(out2), negative_slope=0.01))

      out4 = nn.MaxPool1d(out3.size(-1))(out3)

      return out4, out1, n_pts
