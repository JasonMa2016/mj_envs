from pathlib import Path
import cv2
import glob
import hydra
from matplotlib import cm 
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip

import numpy as np
import os 
# from numpngw import write_apng

import torch
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image  

from r3m import load_r3m
from mj_envs.envs.env_base import load_gofar

torch_transforms = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])

# torch_transforms = T.Compose([T.Resize(224),
#                         T.ToTensor()])

dir_name = "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/"

def main():

    
    task_name = "heat_banana"
    tasks = [f'data_analysis/{task_name}_gofar', f'data_analysis/{task_name}_r3m']
    for task in tasks:
        if 'r3m' in task:
            model = load_r3m("resnet50")
        else:
            model = load_gofar()
        print(task)
        os.makedirs(task, exist_ok=True)
        vid_files = ["/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/heat_banana_expertrecover_20220822-201128_paths_Franka_wrist_*.mp4"]
        # vid_files = ["/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/stack_cup_expertrecover_20220821-175221_paths_Franka_wrist_*.mp4"]
        # vid_files = ["/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/fold_towel_expertrecover_20220821-115621_paths_Franka_wrist_*.mp4"]
        # vid_files = ["/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/stack_chocolate_cone_mixedrecover_20220820-133732_paths_Franka_wrist_*.mp4"]
        # vid_files = ["/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/push_plastic_bottle_expertrecover_20220820-112119_paths_Franka_wrist_*.mp4",
        # "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/push_plastic_bottle_suboptimalrecover_20220820-130936_paths_Franka_wrist_*.mp4"]
        
        # vid_files = ["/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/place_watermelon_expertrecover_20220819-141204_paths_Franka_wrist_*.mp4", 
        # "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/place_watermelon_suboptimalrecover_20220819-142835_paths_Franka_wrist_*.mp4"]
        vids = []
        breakpoints = -1
        for file in vid_files:
            vids.extend(sorted(glob.glob(file)))
            if breakpoints == -1:
                breakpoints = len(vids) # figure out which ones are expert 
        print(vids)

        # create goal embedding: average over unseen goals
        goal_embeddings = []
        goal_file = "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/heat_banana_expertrecover_20220822-201128_paths_Franka_wrist_*.mp4"
        # goal_file = "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/fold_towel_expertrecover_20220821-115621_paths_Franka_wrist_*.mp4"
        # goal_file = "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/stack_chocolate_cone_mixedrecover_20220820-133732_paths_Franka_wrist_*.mp4"
        # success_ids = [10,11,12,13,23,26,28,29] # stack_cone
        # success_trials = [0, 1, 7, 10, 11, 13, 15, 16, 17, 19] # stack_cup
        # goal_file = "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/push_plastic_bottle_expertrecover_20220820-112119_paths_Franka_wrist_*.mp4"
        # goal_file = "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/open_white_drawer_expertrecover_20220817-150331_paths_Franka_wrist_*.mp4"
        # goal_file = "/mnt/tmp_nfs_clientshare/jasonyma/robopen_dataaset/jasonyma_dataset/videos/place_watermelon_expertrecover_20220819-141204_paths_Franka_wrist_*.mp4"
        goal_vids = glob.glob(goal_file)
        for vid_id, vid in enumerate(goal_vids):
            # if vid_id not in success_trials:
            #     continue
            vid_name = vid.split('/')[-1].split('.')[0]
            # read the Robopen video
            vidcap = cv2.VideoCapture(vid)
            success,image = vidcap.read()
            imgs = []
            while True:
                # cv2.imwrite("frame%d.png" % count, image)     # save frame as JPEG file      
                success,image = vidcap.read()
                if not success:
                    break

                imgs.append(image)
            # transform images into correct format   
            for i in range(len(imgs)):
                imgs[i] = torch_transforms(Image.fromarray(imgs[i].astype(np.uint8)))
            imgs = torch.stack(imgs) * 255
            # assert torch.min(imgs) >=0  and torch.max(imgs) <= 255
            model.eval()
            with torch.no_grad():
                embeddings = model(imgs)
                embeddings = embeddings.cpu().numpy()
            # Embedding Goal Distance
            goal_embedding = embeddings[-1]
            goal_embeddings.append(goal_embedding)
        # Compute goal embedding!
        goal_embedding = np.array(goal_embeddings).mean(axis=0)


        distances_all = []
        for j, vid in enumerate(vids):
            vid_name = vid.split('/')[-1].split('.')[0]
            os.makedirs(f"{dir_name}/{task}/{vid_name}", exist_ok=True)
            print(vid_name)

            # read the Robopen video
            vidcap = cv2.VideoCapture(vid)
            success,image = vidcap.read()
            imgs = []
            while True:
                # cv2.imwrite("frame%d.png" % count, image)     # save frame as JPEG file      
                success,image = vidcap.read()
                if not success:
                    break

                imgs.append(image)
            cv2.imwrite(f'{dir_name}/{task}/{vid_name}/goal_frame.png', imgs[-1])
            cl = ImageSequenceClip(imgs, fps=20)
            cl.write_gif(f'{dir_name}/{task}/{vid_name}/traj.gif', fps=20)

            # transform images into correct format   
            for i in range(len(imgs)):
                imgs[i] = torch_transforms(Image.fromarray(imgs[i].astype(np.uint8)))

            save_image(imgs[-1], f'{dir_name}/{task}/{vid_name}/goal_frame_cropped.png')
            imgs = torch.stack(imgs) * 255
            # assert torch.min(imgs) >=0  and torch.max(imgs) <= 255
            print(imgs.shape)
            model.eval()
            with torch.no_grad():
                embeddings = model(imgs)
                embeddings = embeddings.cpu().numpy()
            print(embeddings.shape)
            # t-SNE transformation if not 2D
            # tsne = TSNE(n_components=2, verbose=1, random_state=123)

            # Embedding Goal Distance
            # if j == 0:
            #     # we assume the first demonstration is the expert
            #     goal_embedding = embeddings[-1]

            distances = [] 
            for i in range(embeddings.shape[0]):
                cur_embedding = embeddings[i]
                cur_distance = np.linalg.norm(goal_embedding-cur_embedding)
                distances.append(cur_distance)

            distances_all.append(distances)
            plt.title(f"Video {vid_name} Distance")
            plt.plot(np.arange(len(distances)), distances)
            plt.savefig(f"{dir_name}/{task}/{vid_name}/embedding_distance.png")
            plt.close() 
            
            # Embedding HeatMap
            pairwise_distance = -np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=2)
            plt.title(f"{vid_name} Heatmap")
            plt.imshow(pairwise_distance, cmap='hot')
            plt.colorbar() 
            plt.savefig(f"{dir_name}/{task}/{vid_name}/embedding_heatmap.png")
            plt.close()

            # Embedding Visualization (Assumes 2-D)
            plt.title(f"{vid_name} PCA")
            colors = cm.rainbow(np.linspace(0,1,imgs.shape[0]))
            pca = PCA(n_components=2)
            embeddings = pca.fit_transform(embeddings) 
            
            plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors)
            plt.savefig(f"{dir_name}/{task}/{vid_name}/embedding_vis.png")
            plt.close() 
        
        # compare all trajectories embedding to the true goal
        plt.title(f"Video {task} Distance")
        for i, distances in enumerate(distances_all):
            if i < breakpoints:
                label = "expert"
                color = "blue"
            else:
                label = "suboptimal"
                color = "red"
            plt.plot(np.arange(len(distances)), distances, label=label, c=color)
        # plt.legend(loc="best")
        plt.savefig(f"{dir_name}/{task}/embedding_distance_all.png")
        plt.close() 

if __name__ == '__main__':
    main()