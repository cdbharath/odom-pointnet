'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.pointnet_utils import GlobalFeatureVectorModule, LWGlobalFeatureVectorModule
from config.config import train_inference_config


class MKPRankingModule(nn.Module):
   """
   MKP Selection Module takes in a batch of Matched Key Points as input and
   outputs top_n matches based on the calculated ranking  
   """
   def __init__(self, k=6, top_n=100):
      super().__init__()

      self.k=k
      self.global_feature_vector_module = GlobalFeatureVectorModule(k=self.k)
      self.top_n = top_n

      self.conv1 = nn.Conv1d(1088,512,1)
      self.conv2 = nn.Conv1d(512,256,1)
      self.conv3 = nn.Conv1d(256,128,1)
      self.conv4 = nn.Conv1d(128,1,1)

      self.bn1 = nn.BatchNorm1d(512)
      self.bn2 = nn.BatchNorm1d(256)
      self.bn3 = nn.BatchNorm1d(128)
      self.bn4 = nn.BatchNorm1d(1)

   def forward(self, input):
      _out4, _out1, n_pts = self.global_feature_vector_module(input)
      _out5 = nn.Flatten(1)(_out4).repeat(n_pts,1,1).transpose(0,2).transpose(0,1)

      _out6 = torch.cat([_out5, _out1], 1)

      out1 = F.relu(self.bn1(self.conv1(_out6)))
      out2 = F.relu(self.bn2(self.conv2(out1)))
      out3 = F.relu(self.bn3(self.conv3(out2)))
      out4 = F.relu(self.bn4(self.conv4(out3)))

      _, indices = torch.topk(out4, self.top_n)

      # filtering top n based on indices
      batch_content = []
      for i in range(input.shape[0]):
         shortlist_untransposed = torch.transpose(input[i], 0, 1)[indices[i].squeeze()]
         shortlist = torch.transpose(shortlist_untransposed, 0, 1)
         shortlist_unsqueezed = torch.unsqueeze(shortlist, 0)

         batch_content.append(shortlist_unsqueezed)

      if len(batch_content) > 0:
         top_matches = torch.cat(batch_content)
      else:
         top_matches = []

      return top_matches


class MKPSelectionModule(nn.Module):
   """
   MKP Selection Module takes in a batch of Matched Key Points as input and
   outputs top_n matches based on the calculated ranking  
   """
   def __init__(self, k=6):
      super().__init__()

      self.k=k
      self.global_feature_vector_module = GlobalFeatureVectorModule(k=self.k)

      self.conv1 = nn.Conv1d(1088,512,1)
      self.conv2 = nn.Conv1d(512,256,1)
      self.conv3 = nn.Conv1d(256,128,1)
      self.conv4 = nn.Conv1d(128,128,1)
      self.conv5 = nn.Conv1d(128,1,1)

      self.bn1 = nn.BatchNorm1d(512)
      self.bn2 = nn.BatchNorm1d(256)
      self.bn3 = nn.BatchNorm1d(128)
      self.bn4 = nn.BatchNorm1d(128)
      self.bn5 = nn.BatchNorm1d(1)

      nn.init.xavier_uniform_(self.conv1.weight)
      nn.init.xavier_uniform_(self.conv2.weight)
      nn.init.xavier_uniform_(self.conv3.weight)
      nn.init.xavier_uniform_(self.conv4.weight)
      nn.init.xavier_uniform_(self.conv5.weight)

   def forward(self, input):
      global_feature_vector, local_feature_vector, n_pts = self.global_feature_vector_module(input)
      global_feature_vectors = nn.Flatten(1)(global_feature_vector).repeat(n_pts,1,1).transpose(0,2).transpose(0,1)

      combined_feature_vector = torch.cat([global_feature_vectors, local_feature_vector], 1)

      out1 = self.bn1(F.leaky_relu(self.conv1(combined_feature_vector), negative_slope=0.01))
      out2 = self.bn2(F.leaky_relu(self.conv2(out1), negative_slope=0.01))
      out3 = self.bn3(F.leaky_relu(self.conv3(out2), negative_slope=0.01))
      out4 = self.bn4(F.leaky_relu(self.conv4(out3), negative_slope=0.01))
      out5 = self.conv5(out4)

      return out5


class RotationTranslationEstimationModule(nn.Module):
   """
   Estimates rotation or translation from MKP module's output
   """
   def __init__(self, k=6, do_rotation_augmentation=False, output_dim=3, normalize_output=False):
      super().__init__()

      if do_rotation_augmentation:
         # TODO Implement this
         raise NotImplementedError

      self.k=k
      self.normalize_output = normalize_output
      self.global_feature_vector_module = LWGlobalFeatureVectorModule(k=self.k)     

      self.fc2 = nn.Linear(1024,512)
      self.fc3 = nn.Linear(512,256)
      self.fc4 = nn.Linear(256,output_dim)

      self.bn2 = nn.BatchNorm1d(512)
      self.bn3 = nn.BatchNorm1d(256) 

      self.dropout = nn.Dropout(train_inference_config.DROPOUT)
      
      nn.init.xavier_uniform_(self.fc2.weight)
      nn.init.xavier_uniform_(self.fc3.weight)
      nn.init.xavier_uniform_(self.fc4.weight)

   def forward(self, input):
      out6, _, _ = self.global_feature_vector_module(input)
      xb = nn.Flatten(1)(out6)
      xb = self.bn2(F.leaky_relu(self.fc2(xb), negative_slope=0.01))
      xb = self.dropout(self.bn3(F.leaky_relu(self.fc3(xb), negative_slope=0.01)))
      xb = self.fc4(xb)
      
      if self.normalize_output:
         xb = F.normalize(xb, p=2, dim=1)
      
      return xb


class LWPoseEstimationModule(nn.Module):
   """
   Estimates rotation or translation from MKP module's output
   """
   def __init__(self, k=6, do_rotation_augmentation=False, output_dim=3, normalize_output=False):
      super().__init__()

      if do_rotation_augmentation:
         # TODO Implement this
         raise NotImplementedError

      self.k=k
      self.normalize_output = normalize_output
      self.global_feature_vector_module = GlobalFeatureVectorModule(k=self.k)     

      self.fc2 = nn.Linear(1024,512)
      self.fc3 = nn.Linear(512,256)
      self.fc4 = nn.Linear(256,output_dim)

      self.bn2 = nn.BatchNorm1d(512)
      self.bn3 = nn.BatchNorm1d(256) 

      self.dropout = nn.Dropout(train_inference_config.DROPOUT)
      
      nn.init.xavier_uniform_(self.fc2.weight)
      nn.init.xavier_uniform_(self.fc3.weight)
      nn.init.xavier_uniform_(self.fc4.weight)

   def forward(self, input):
      out6, _, _ = self.global_feature_vector_module(input)
      xb = nn.Flatten(1)(out6)
      xb = self.bn2(F.leaky_relu(self.fc2(xb), negative_slope=0.01))
      xb = self.dropout(self.bn3(F.leaky_relu(self.fc3(xb), negative_slope=0.01)))
      xb = self.fc4(xb)
      
      if self.normalize_output:
         xb = F.normalize(xb, p=2, dim=1)
      
      return xb


class AxisAngleLoss(nn.Module):
    def __init__(self):
        super(AxisAngleLoss, self).__init__()

    def forward(self, pred, gt):
        # Calculate RMSE loss
        rmse_loss = torch.sqrt(nn.MSELoss()(pred, gt))

        # Extract the slices [1:4] from pred
        sliced_pred = pred[:, 1:4]

        # Calculate the squared unit vector loss
        unit_vector_loss = torch.mean((torch.norm(sliced_pred, dim=1) - 1.0)**2)

        # Combine RMSE loss and squared unit vector loss
        loss = rmse_loss + torch.sqrt(unit_vector_loss)

        return loss


class Lodonet(nn.Module):
   def __init__(self, k=6, top_n=100, include_rot=True, do_rotation_augmentation=False, return_top_matches=False):
      super().__init__()
      self.include_rot = include_rot
      self.return_top_matches = return_top_matches

      self.mkp_selection_module = MKPSelectionModule(k=k, top_n=top_n)
      self.rotation_est_module = RotationTranslationEstimationModule(k=k, do_rotation_augmentation=do_rotation_augmentation)
      self.translation_est_module = RotationTranslationEstimationModule(k=k, do_rotation_augmentation=do_rotation_augmentation)

   def forward(self, input):
      top_matches = self.mkp_selection_module(input)
      
      translation = self.translation_est_module(top_matches)
      if self.include_rot: 
         rotation = self.rotation_est_module(top_matches)
      
      if not self.include_rot:
         return translation
      
      if self.return_top_matches:
         return rotation, translation, top_matches
      
      return rotation, translation


if __name__ == "__main__":
   mkp_selelction_module = MKPSelectionModule()
   rotation_estimation_module = RotationTranslationEstimationModule()

   x = torch.rand(2, 6, 10, dtype=torch.float, requires_grad=False)
   mkp_ranking = mkp_selelction_module(x)
   
   x = torch.rand(2, 6, 5, dtype=torch.float, requires_grad=False)
   rotation_est = rotation_estimation_module(x)
   
   print(f"Size format [batch_size, size_per_point, num_points]")
   print(f"Input size: {x.shape}")
   print(f"MKP module output size: {mkp_ranking.shape}")
   print(f"Rotation estimation module output size: {rotation_est.shape}")
