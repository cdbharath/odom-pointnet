'''
Author: Bharath Kumar Ramesh Babu
Email: kumar7bharath@gmail.com
'''

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset_utils import KittiLidarOdomDataset, shortlist_points_with_mkp_ranking
from net.lodonet import MKPSelectionModule, LWPoseEstimationModule, RotationTranslationEstimationModule
from net.lodonet2 import PointNet2
from config.config import dataset_config, log_config, train_inference_config
from utils.common_utils import ensure_path_exists

        
def clip_gradients(model, max_grad_norm = 0.1):
    # Get all the parameters that require gradients
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    
    # Clip gradients to the specified threshold
    nn.utils.clip_grad_norm_(parameters, max_grad_norm)

def quaterntion_to_axis_angle_torch(q):
    theta = 2 * torch.acos(q[0])
    u1 = q[1] / torch.sqrt(1 - q[0]**2)
    u2 = q[2] / torch.sqrt(1 - q[0]**2)
    u3 = q[3] / torch.sqrt(1 - q[0]**2)
    return theta, torch.tensor([u1, u2, u3])
    
def axis_angle_to_quaternion_torch(axis_angle):
    qw = torch.cos(axis_angle[0] / 2)
    qx = axis_angle[1] * torch.sin(axis_angle[0] / 2)
    qy = axis_angle[2] * torch.sin(axis_angle[0] / 2)
    qz = axis_angle[3] * torch.sin(axis_angle[0] / 2)
    return torch.tensor([qw, qx, qy, qz])
    
def axis_angle_to_quaternion_batch(axis_angle):
    qw = torch.cos(axis_angle[:, 0] / 2)
    qx = axis_angle[:, 1] * torch.sin(axis_angle[:, 0] / 2)
    qy = axis_angle[:, 2] * torch.sin(axis_angle[:, 0] / 2)
    qz = axis_angle[:, 3] * torch.sin(axis_angle[:, 0] / 2)
    return torch.stack([qw, qx, qy, qz], dim=1)
    
def train(model, loss_fn, dataloader, optim, device, augument, is_val=False, model_name='PointNet'):
    '''
    Implements alternating optimization 
    MKP module is optimized followed by rotation and translation module 
    '''
    mkp_model, r_model, t_model = model
    mkp_loss_fn, r_loss_fn, t_loss_fn = loss_fn
    mkp_optim, r_optim, t_optim = optim

    if not is_val:
        mkp_model.train()
        r_model.train()
        t_model.train()
    else:
        mkp_model.eval()
        r_model.eval()
        t_model.eval()

    pbar = tqdm(dataloader)

    batch_count = 0
    trans_running_loss = 0.0
    rot_running_loss = 0.0
    mkp_running_loss = 0.0

    for sample in pbar:
        batch_count += 1

        scans = sample[0].to(device)
        scans = scans.transpose(1, 2)

        gt = sample[1].to(device)
        rotation_gt = gt[:, 3:]
        translation_gt = gt[:, :3]

        mkp_gt = sample[2].to(device)
        mkp_gt = mkp_gt.transpose(1, 2)

        # if model_name == 'PointNet':
        #     mkp_ranking = mkp_model(scans)
        # else:
        #     mkp_ranking, _ = mkp_model(scans)

        # mkp_loss = mkp_loss_fn(mkp_ranking.squeeze(), mkp_gt.squeeze())
        # mkp_running_loss += mkp_loss.item()

        # if not is_val:
        #     mkp_optim.zero_grad()
        #     mkp_loss.backward()
        #     clip_gradients(mkp_model)
        #     mkp_optim.step()

        # sigmoid_ranking = torch.round(torch.sigmoid(mkp_ranking)).int()
        # if model_name != 'PointNet':
        #     sigmoid_ranking = sigmoid_ranking.unsqueeze(1)
        
        shortlist_points = shortlist_points_with_mkp_ranking(
            scans, mkp_gt)
        rotation_shortlist_points = shortlist_points
            
        if augument:
            aug_scans = sample[3].to(device)
            aug_scans = aug_scans.transpose(1, 2)
            aug_gt = sample[4].to(device)

            rotation_shortlist_points = shortlist_points_with_mkp_ranking(
                aug_scans, mkp_gt)
            rotation_gt = aug_gt[:, 3:]
            

        if model_name == 'PointNet':
            translation = t_model(shortlist_points)
            rotation = r_model(rotation_shortlist_points)
        else:
            translation, _ = t_model(shortlist_points)
            rotation, _ = r_model(rotation_shortlist_points)

        t_loss = torch.sqrt(t_loss_fn(translation, translation_gt))
        r_loss = torch.sqrt(r_loss_fn(rotation, rotation_gt))

        trans_running_loss += t_loss.item()
        rot_running_loss += r_loss.item()

        if not is_val:
            r_optim.zero_grad()
            r_loss.backward()
            clip_gradients(r_model)
            r_optim.step()

            t_optim.zero_grad()
            t_loss.backward()
            clip_gradients(t_model)
            t_optim.step()

        pbar.set_description(
            f"Avg. MKP Loss {mkp_running_loss/batch_count:.3f} | Trans Loss {trans_running_loss/batch_count:.3f} | Rot Loss {rot_running_loss/batch_count:.3f}"
        )

    return mkp_running_loss/batch_count, trans_running_loss/batch_count, rot_running_loss/batch_count


if __name__ == "__main__":
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    batch_size = train_inference_config.BATCH_SIZE
    num_workers = train_inference_config.NUM_WORKERS
    learning_rate_t = train_inference_config.LEARNING_RATE_T
    learning_rate_r = train_inference_config.LEARNING_RATE_R
    learning_rate_mkp = train_inference_config.LEARNING_RATE_MKP
    n_epochs = train_inference_config.N_EPOCHS
    val_every = train_inference_config.VAL_EVERY
    ckpt_every = log_config.CKPT_EVERY
    top_n = train_inference_config.TOP_N
    momentum = train_inference_config.MOMENTUM
    milestone_r = train_inference_config.MILESTONE_R
    milestone_t = train_inference_config.MILESTONE_T
    milestone_mkp = train_inference_config.MILESTONE_MKP
    lr_decay_r = train_inference_config.LEARNING_RATE_DECAY_R
    lr_decay_t = train_inference_config.LEARNING_RATE_DECAY_T
    lr_decay_mkp = train_inference_config.LEARNING_RATE_DECAY_MKP
    weight_decay_t = train_inference_config.WEIGHT_DECAY_T
    weight_decay_r = train_inference_config.WEIGHT_DECAY_R
    weight_decay_mkp = train_inference_config.WEIGHT_DECAY_MKP
    augument = train_inference_config.AUGUMENT
    train_cities = train_inference_config.TRAIN_CITIES
    val_cities = train_inference_config.VAL_CITIES
    model_name = train_inference_config.MODEL
    aug_angle = train_inference_config.AUG_ANGLE

    # Create a dictionary with the parameter names and their values
    parameters = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "learning_rate_t": learning_rate_t,
        "learning_rate_r": learning_rate_r,
        "learning_rate_mkp": learning_rate_mkp,
        "n_epochs": n_epochs,
        "val_every": val_every,
        "ckpt_every": ckpt_every,
        "top_n": top_n,
        "momentum": momentum,
        "milestone_r": milestone_r,
        "milestone_t": milestone_t,
        "milestone_mkp": milestone_mkp,
        "lr_decay_r": lr_decay_r,
        "lr_decay_t": lr_decay_t,
        "lr_decay_mkp": lr_decay_mkp,
        "weight_decay_t": weight_decay_t,
        "weight_decay_r": weight_decay_r,
        "weight_decay_mkp": weight_decay_mkp,
        "augument": augument,
        "train_cities": train_cities,
        "val_cities": val_cities,
        "aug_angle": aug_angle, 
        "model_name": model_name,
    }

    # Specify the output file path
    output_file = os.path.join(log_config.LOG_DIR, formatted_datetime, "parameters.txt")

    # Ensure the folders exixts
    ensure_path_exists(os.path.join(log_config.LOG_DIR, formatted_datetime))

    # Open the file in write mode and write the parameters
    with open(output_file, "w") as file:
        for param, value in parameters.items():
            file.write(f"{param} = {value}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_data = KittiLidarOdomDataset(dataset_config.LIDAR_PATH, dataset_config.GT_PATH, include_sequences=train_cities,
                                       pre_process=True, noise_std=0.0, augument=augument, augment_yaw_max=aug_angle, top_k=top_n)
    val_data = KittiLidarOdomDataset(dataset_config.LIDAR_PATH, dataset_config.GT_PATH, include_sequences=val_cities,
                                     pre_process=True, noise_std=0.0, top_k=top_n)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers, drop_last=True)

    if model_name == "PointNet":
        mkp_model = MKPSelectionModule()
        r_model = LWPoseEstimationModule(output_dim=4)
        t_model = LWPoseEstimationModule()
    else:
        mkp_model = PointNet2(num_class=200)
        r_model = PointNet2(num_class=3)
        t_model = PointNet2(num_class=3)

    mkp_model.to(device)
    r_model.to(device)
    t_model.to(device)

    mkp_model.double()
    r_model.double()
    t_model.double()

    # ckpt = torch.load(os.path.join(dataset_config.CWD, 'pre_trained/mkp_checkpoint_epoch270.pth'), map_location=device)
    # mkp_model.load_state_dict(ckpt, strict=True)

    mkp_loss = nn.BCEWithLogitsLoss()
    r_loss = nn.MSELoss()
    t_loss = nn.MSELoss()

    mkp_optim = torch.optim.Adam(mkp_model.parameters(), lr=learning_rate_mkp, weight_decay=weight_decay_mkp)
    r_optim = torch.optim.Adam(r_model.parameters(), lr=learning_rate_r, weight_decay=weight_decay_r)
    t_optim = torch.optim.Adam(t_model.parameters(), lr=learning_rate_t, weight_decay=weight_decay_t)

    mkp_sched = torch.optim.lr_scheduler.MultiStepLR(mkp_optim, milestones=milestone_mkp, gamma=lr_decay_mkp)
    r_sched = torch.optim.lr_scheduler.MultiStepLR(r_optim, milestones=milestone_r, gamma=lr_decay_r)
    t_sched = torch.optim.lr_scheduler.MultiStepLR(t_optim, milestones=milestone_t, gamma=lr_decay_t)

    trans_loss_accumulator = []
    rot_loss_accumulator = []
    trans_val_loss_accumulator = []
    rot_val_loss_accumulator = []
    mkp_loss_accumulator = []
    mkp_val_loss_accumulator = []

    for i in range(n_epochs):
        print(f"epoch {i}")

        mkp_sched.step()
        r_sched.step()
        t_sched.step()

        avg_mkp_loss, avg_trans_loss, avg_rot_loss = train([mkp_model, r_model, t_model], [mkp_loss, r_loss, t_loss],
                                                           train_dataloader, [mkp_optim, r_optim, t_optim], device, augument, is_val=False, model_name=model_name)
        print(f"avg training. mkp loss: {avg_mkp_loss} | trans loss: {avg_trans_loss} | rot loss: {avg_rot_loss}")

        trans_loss_accumulator.append(avg_trans_loss)
        rot_loss_accumulator.append(avg_rot_loss)
        mkp_loss_accumulator.append(avg_mkp_loss)

        if i % val_every == 0:
            with torch.no_grad():
                val_mkp_loss, val_trans_loss, val_rot_loss = train([mkp_model, r_model, t_model], [mkp_loss, r_loss, t_loss],
                                                                   val_dataloader, [mkp_optim, r_optim, t_optim], device, augument=False, is_val=True, model_name=model_name)
                trans_val_loss_accumulator.append(val_trans_loss)
                rot_val_loss_accumulator.append(val_rot_loss)
                mkp_val_loss_accumulator.append(val_mkp_loss)
                print(f"avg validation. mkp loss: {val_mkp_loss} | trans loss: {val_trans_loss} | rot loss: {val_rot_loss}")

        if i % ckpt_every == 0:
            print(f"Saving Checkpoint")
            ensure_path_exists(log_config.CKPT_PATH)
            # torch.save(mkp_model.state_dict(), os.path.join(log_config.LOG_DIR, formatted_datetime, "mkp_checkpoint_epoch" + str(i) + ".pth"))
            torch.save(r_model.state_dict(), os.path.join(log_config.LOG_DIR, formatted_datetime, "r_checkpoint_epoch" + str(i) + ".pth"))
            torch.save(t_model.state_dict(), os.path.join(log_config.LOG_DIR, formatted_datetime, "t_checkpoint_epoch" + str(i) + ".pth"))

        # save loss plots
        plt.figure(1)
        plt.title("Translation Loss")
        plt.plot(trans_loss_accumulator)
        plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "translation_loss.png"))

        plt.figure(2)
        plt.title("Rotation Loss")
        plt.plot(rot_loss_accumulator)
        plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "rotation_loss.png"))

        plt.figure(3)
        plt.title("Translation Validation Loss")
        plt.plot(trans_val_loss_accumulator)
        plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "translation_val_loss.png"))

        plt.figure(4)
        plt.title("Rotation Validation Loss")
        plt.plot(rot_val_loss_accumulator)
        plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "rotation_val_loss.png"))

        # plt.figure(5)
        # plt.title("MKP Loss")
        # plt.plot(mkp_loss_accumulator)
        # plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "mkp_loss.png"))

        # plt.figure(6)
        # plt.title("MKP Validation Loss")
        # plt.plot(mkp_val_loss_accumulator)
        # plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "mkp_val_loss.png"))

        plt.figure(7)
        plt.yscale('log')
        plt.title("Translation Log Loss")
        plt.plot(trans_loss_accumulator)
        plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "translation_log_loss.png"))

        plt.figure(8)
        plt.yscale('log')
        plt.title("Rotation Log Loss")
        plt.plot(rot_loss_accumulator)
        plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "rotation_log_loss.png"))

        plt.figure(9)
        plt.yscale('log')
        plt.title("Rotation Validation Log Loss")
        plt.plot(rot_val_loss_accumulator)
        plt.savefig(os.path.join(log_config.LOG_DIR, formatted_datetime, "rotation_val_log_loss.png"))
