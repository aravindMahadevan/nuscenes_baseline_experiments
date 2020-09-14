import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from nuscenes import NuScenes

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.eval.prediction.config import load_prediction_config
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

from nuscenes.prediction.models.physics import ConstantVelocityHeading, PhysicsOracle
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from PIL import Image
import os 
import random

class NuscenesDataset(Dataset):
    
    def __init__(self, 
                 nusc,
                 helper,
                 maps_dir,
                 save_maps_dataset = False, 
                 config_name = 'predict_2020_icra.json',
                 history=1, 
                 num_examples = None,
                 in_agent_frame=True):
        
        
        self.nusc = nusc
        self.helper = helper
        
        #initialize the data set 
        if maps_dir == 'maps_train':
            dataset_version = "train"
        elif maps_dir == 'maps':
            dataset_version = "train_val"
        elif maps_dir == 'maps_val':
            dataset_version = "val"
        
        #initialize maps directory where everything will be saved 
        self.maps_dir = os.path.join(os.getcwd(), maps_dir)
        self.data_set = get_prediction_challenge_split(dataset_version,dataroot=self.nusc.dataroot)
        
        if num_examples: 
            self.data_set = self.data_set[:num_examples]

        #initialize rasterizers for map generation
        self.static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        self.agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=history)
        self.mtp_input_representation = InputRepresentation(self.static_layer_rasterizer, self.agent_rasterizer, Rasterizer())

        self.in_agent_frame = in_agent_frame

        self.config = load_prediction_config(self.helper, config_name)

        self.valid_data_points = []
        
        self.save_maps_dataset = save_maps_dataset
        
        if self.save_maps_dataset: 
            self.save_maps()
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            
    def save_maps(self):
        '''
        Input: None 
        Output: None
        
        This method finds all the valid data points in the data set. We define a valid data point 
        where the velocity, acceleartion, and heading specified by token not NaN. 
        '''
        print("starting to save maps")
        for i, token in enumerate(self.data_set): 
            instance_token_img, sample_token_img = self.data_set[i].split('_')
            
            file_path = os.path.join(self.maps_dir, "maps_{0}.jpg".format(i))
            
            instance_token_img, sample_token_img = self.data_set[i].split('_')
            img = self.mtp_input_representation.make_input_representation(instance_token_img, sample_token_img)
            im = Image.fromarray(img)
            im.save(file_path)
        
            print("{0}/{1} image saved".format(i, len(self.data_set)))
        
        print("done saving maps")
        
        
        
    def __len__(self):
        return len(self.data_set)
    
    #return the image tensor, agent state vector, and the ground truth
    def __getitem__(self, index):
        token = self.data_set[index]
        instance_token_img, sample_token_img = self.data_set[index].split('_')
        
        velocity = self.helper.get_velocity_for_agent(instance_token_img, sample_token_img)
        acceleration = self.helper.get_acceleration_for_agent(instance_token_img, sample_token_img)
        heading = self.helper.get_heading_change_rate_for_agent(instance_token_img, sample_token_img)        

        #using a padding token of -1
        if np.isnan(velocity) or np.isnan(acceleration) or np.isnan(heading):
            velocity =  acceleration = heading = -1 

        #construct agent state vector 
        agent_state_vec = torch.Tensor([velocity, acceleration, heading])
        
        #change image from (3, N, N), will have data loader take care 
        #get image and construct tensor 
        file_path = os.path.join(self.maps_dir, "maps_{0}.jpg".format(index))
        
        im = Image.open(file_path)
        img = np.array(im)
        image_tensor = torch.Tensor(img).permute(2, 0, 1)
        
        #get ground truth 
        ground_truth = self.helper.get_future_for_agent(instance_token_img, 
                                                        sample_token_img,
                                                        self.config.seconds, 
                                                        in_agent_frame=self.in_agent_frame)
        
        ground_truth = torch.Tensor(ground_truth).unsqueeze(0)
        return image_tensor, agent_state_vec, ground_truth, token 
