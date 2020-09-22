from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.models.backbone import ResNetBackbone, MobileNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss
from nuscenes.eval.prediction.data_classes import Prediction


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from NuscenesDataset import NuscenesDataset
import numpy as np 
import time
import os 
import matplotlib
import matplotlib.pyplot as plt 
import pickle
import random
import json

class Nuscenes_Baseline_Experiment():
    def __init__(self, 
                 output_dir,
                 data_version = 'v1.0-trainval',
                 M = 3, 
                 num_epochs = 150, 
                 num_workers = 4,
                 num_training_examples = None,
                 num_validation_examples = None, 
                 model = 'MTP',
                 lr = 1e-4, 
                 decay_factor = 0.9,
                 batch_size=16, 
                 training_maps = 'maps_train',
                 validation_maps = 'maps_val',
                 data_root = '../full_data/sets/nuscenes',
                 trajectory_set_path = None, 
                 backbone_architecture='mobilenet_v2'):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        #init experiment parameters
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint.pth.tar")
        self.config_path = os.path.join(self.output_dir, "config.txt")
      
                
        self.num_epochs = num_epochs
        self.batch_size = batch_size 
        self.lr = lr 
        self.backbone_architecture = backbone_architecture
        
        #init network and criterion
        self.num_modes = M
        self.backbone = MobileNetBackbone(backbone_architecture)
        
        if model == 'MTP':
            self.model = MTP(self.backbone, num_modes=self.num_modes).to(self.device) 
            self.criterion = MTPLoss(self.num_modes)
        elif model == 'CoverNet':
            self.model = CoverNet(self.backbone, num_modes=self.num_modes).to(self.device)
            if not trajectory_set_path or not os.path.isfile(trajectory_set_path): 
                raise ValueError("Invalid trajectory set path for CoverNet. Check path.")
            trajectories =pickle.load(open(trajectory_set_path, 'rb'))
            trajectories = torch.Tensor(trajectories)
            self.criterion = ConstantLatticeLoss(trajectories).to(self.device)
        else:
            raise ValueError("Invalid model specification")           
        
        #init data set and data loader 
        self.nusc = NuScenes(version=data_version, dataroot=data_root, verbose=True)
        self.helper = PredictHelper(self.nusc)
        
        self.training_set = NuscenesDataset(self.nusc, 
                                            self.helper, 
                                            maps_dir = training_maps, 
                                            num_examples = num_training_examples)
        
        self.validation_set = NuscenesDataset(self.nusc, 
                                             self.helper, 
                                             maps_dir = validation_maps, 
                                             num_examples = num_validation_examples)      
        
        print(len(self.validation_set)) 
        print(len(self.training_set))
        
        self.train_loader = DataLoader(self.training_set, 
                                       batch_size=self.batch_size, 
                                       shuffle = True,
                                       pin_memory = True,
                                       num_workers = num_workers)
        
        
        self.val_loader = DataLoader(self.validation_set, 
                                     batch_size=self.batch_size, 
                                     shuffle = True,
                                     pin_memory = True,
                                     num_workers = num_workers)        
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=decay_factor)

        
        self.train_loss = []
        self.val_loss = []
        self.val_epochs = []
        
        #check if already created experiment
        if os.path.isfile(self.config_path):
            print("Reloading previous experiment")
            self.load()
        else: 
            #create directory
            os.makedirs(output_dir, exist_ok=True)
            self.save()
                    
    @property
    def epoch(self):
        return len(self.train_loss)

    def setting(self):
        return {'Model' : self.model,
                'Optimizer' : self.optimizer,
                'Number of Modes' : self.num_modes,
                'Backbone architecture' : self.backbone_architecture, 
                'Batch Size' : self.batch_size,
                'Learning Rate' : self.lr, 
                'Number of Epochs' : self.num_epochs}

    def __repr__(self):
        string = ''
        for key, val in self.setting().items():
            string += '{} : {}\n'.format(key, val)
        return string

    def state_dict(self):
        return {'Model' : self.model.state_dict(),
                'Optimizer' : self.optimizer.state_dict(),
                'TrainLoss' : self.train_loss, 
                'ValLoss': self.val_loss, 
                'ValEpochs': self.val_epochs, 
                'Scheduler': self.scheduler}
        
    def save_fig(self):
        fig, ax = plt.subplots(1)
        ax.plot(range(len(self.train_loss)), self.train_loss, 'r', label='Train Loss')
        ax.plot(self.val_epochs, self.val_loss, 'b', label='Validation Loss')
        ax.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('MTP Training and Validation Loss')
        fig.savefig(os.path.join(self.output_dir, 'mtp_train_loss_epoch'))
        plt.close()
        
    def save(self):
        self.save_fig()
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['Model'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.train_loss = checkpoint['TrainLoss']
        self.val_loss = checkpoint['ValLoss']
        self.val_epochs = checkpoint['ValEpochs']
        self.scheduler = checkpoint['Scheduler']
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
         
    def load(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint
    
    def run(self):
        
        starting_epoch = len(self.train_loss)
        print("Starting/Continuing Training at epoch {0}!".format(starting_epoch))
        for epoch in range(starting_epoch, self.num_epochs): 
            self.train(epoch)
            
            self.val_epochs.append(epoch)
            print('calculating validation loss')
            self.compute_validation_loss(epoch)
            print('saving weights')
            self.save()
            self.scheduler.step()

#        self.val_epochs.append(epoch)
#        print("computing final validation loss")
#        self.compute_validation_loss(epoch)
#        print('saving final weights')
        self.save()
        print("Done training!")

    def compute_validation_loss(self, epoch): 
        epoch_loss = 0
        count=0
        with torch.no_grad():
            for image_tensor, agent_vec, ground_truth, _ in self.val_loader:
                output = self.model(image_tensor.to(self.device), agent_vec.to(self.device))
                loss = self.criterion(output, ground_truth.to(self.device))
                count+=1
                print("{0}/{1} validation batches finished".format(count, len(self.val_loader)))
                epoch_loss += loss.item()

        average_loss = epoch_loss/(len(self.val_loader))
        print("Epoch Validation {0}/{1} complete. Loss at epoch: {2}".format(epoch, self.num_epochs - 1, average_loss))
        print("Ending Validation Epoch {0} at {1}".format(epoch, str(time.strftime("%H:%M:%S", time.localtime()))))
        self.val_loss.append(average_loss)     
          
    def train(self, epoch): 
        epoch_loss = 0
        count=0
        print("Starting Epoch {0} at {1}".format(epoch, str(time.strftime("%H:%M:%S", time.localtime()))))
        for image_tensor, agent_vec, ground_truth, _ in self.train_loader:
            output = self.model(image_tensor.to(self.device), agent_vec.to(self.device))
            loss = self.criterion(output, ground_truth.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            count+=1
            print("{0}/{1} training batches finished".format(count, len(self.train_loader)))
            epoch_loss += loss.item()
            
        average_loss = epoch_loss/(len(self.train_loader))
        print("Epoch {0}/{1} complete. Loss at epoch: {2}".format(epoch, self.num_epochs - 1, average_loss))
        print("Ending Epoch {0} at {1}".format(epoch, str(time.strftime("%H:%M:%S", time.localtime()))))
        self.train_loss.append(average_loss)

    def generate_predictions_from_validation(self, json_file_name = 'mtp_preds.json'): 
        mtp_output = []
        val_loader = DataLoader(self.validation_set, 
                                     batch_size=1, 
                                     shuffle = True,
                                     pin_memory = True,
                                     num_workers = 0)  
        
        print('starting validation predictions')
        prediction_output_path = os.path.join(self.output_dir,json_file_name)
        with torch.no_grad():
            count = 0
            for image_tensor, agent_state_vec, ground_truth, token  in val_loader:
                config = self.validation_set.config
                instance_token_img, sample_token_img = token[0].split('_')
                
                output = self.model(image_tensor.to(self.device), agent_state_vec.to(self.device))
                prediction = output[:,:-self.num_modes].cpu().numpy()
                probabilites = output[:,-self.num_modes:].squeeze(0).cpu().numpy()
                prediction = prediction.reshape(self.num_modes, config.seconds * 2, 2)

                serialized_pred = Prediction(instance_token_img, sample_token_img, prediction, probabilites).serialize()
                mtp_output.append(serialized_pred)
                print("{0}/{1} predictions saved".format(count, len(val_loader)))
                count+=1
                    
        print("saving predictions")
        json.dump(mtp_output, open(prediction_output_path, "w"))
            
            
if __name__ == "__main__":
    exp = Nuscenes_Baseline_Experiment(output_dir = 'exp15', model = 'MTP')
    exp.run()
    exp.generate_predictions_from_validation()


