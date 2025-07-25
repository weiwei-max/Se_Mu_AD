import torch
from torch import nn

from sklearn.cluster import AffinityPropagation
import random

import yaml
from easydict import EasyDict as edict
import cv2

def parse_task_dictionary(db_name, task_dictionary):
    """ 
        Return a dictionary with task information. 
        Additionally we return a dict with key, values to be added to the main dictionary
        
      loss_target_speed: 0.1
      loss_checkpoint: 0.1
      loss_semantic: 0.1
      loss_bev_semantic: 0.1
      loss_depth: 0.1
      loss_center_heatmap: 0.1
      loss_wh: 0.1
      loss_offset: 0.1
      loss_yaw_class: 0.1
      loss_yaw_res: 0.1
    """

    task_cfg = edict()
    task_cfg.NAMES = []
    task_cfg.NAMES.append('loss_target_speed')
    task_cfg.NAMES.append('loss_checkpoint')
    task_cfg.NAMES.append('loss_semantic')
    task_cfg.NAMES.append('loss_bev_semantic')
    task_cfg.NAMES.append('loss_depth')
    task_cfg.NAMES.append('loss_bbox')
    # task_cfg.NAMES.append('loss_center_heatmap')
    # task_cfg.NAMES.append('loss_wh')
    # task_cfg.NAMES.append('loss_offset')
    # task_cfg.NAMES.append('loss_yaw_class')
    # task_cfg.NAMES.append('loss_yaw_res')
    
    return task_cfg

def create_config(exp_file, params):
    
    with open(exp_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Copy all the arguments
    cfg = edict()
    for k, v in config.items():
        cfg[k] = v

    cfg.TASKS = parse_task_dictionary(cfg['train_db_name'], cfg['task_dictionary'])

    return cfg

class MTLoss_affinity(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MTLoss_affinity, self).__init__()
        self.p = p
        self.tasks = tasks
        self.loss_weights = loss_weights
        self.group = None
    
    def forward(self, losses, tasks):
        # out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}
        out = losses
        
        for group, comp in self.group.items():
            out[group] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in comp]))

        return out
    

class form_task_affinity():
    def __init__(self, p):
        self.tasks = p.TASKS.NAMES
        self.group = p.group
        self.affin_decay = p.affin_decay
        if p.preference == 'None': self.preference = None
        else: self.preference=p.preference
        self.affinity_map = torch.zeros(len(self.tasks), len(self.tasks))
        self.pre_loss = {task: -1 for task in p.TASKS.NAMES}
        self.num=0
        self.convergence_iter = 50
        self.index = 0
    def update(self, group, loss_dict):
        for task_s in self.group[group]:
            for task_t in self.tasks:
                if self.pre_loss[task_t]<=0: continue
                if task_t in self.group[group]:
                    
                    if self.pre_loss[task_s]<=0: continue
                    affin_t = 1 - loss_dict[task_t].item()/self.pre_loss[task_t]
                    affin_t /= len(self.group[group])
                    affin_s = 1 - loss_dict[task_s].item()/self.pre_loss[task_s]
                    affin_s /= len(self.group[group])
                    
                    if task_t==task_s:
                        if affin_t < 0: pass
                        else: self.affin_update(affin_t, task_s, task_t)
                    elif affin_t * affin_s >=0:
                        self.affin_update(affin_t, task_s, task_t)
                    else:
                        self.affin_update(-max(affin_t, affin_s), task_s, task_t)
                    
                else:
                    affin = 1 - loss_dict[task_t].item()/self.pre_loss[task_t]
                    affin /= len(self.group[group])
                    self.affin_update(affin, task_s, task_t)
        
        for task in self.tasks: self.pre_loss[task] = loss_dict[task].item()
                
    def affin_update(self, affin, task_s, task_t):
        task_s_i, task_t_i = self.tasks.index(task_s), self.tasks.index(task_t)
        self.affinity_map[task_s_i, task_t_i] = (1-self.affin_decay)*self.affinity_map[task_s_i, task_t_i] + self.affin_decay*affin
        
    def init_pre_loss(self):
        for task in self.tasks: self.pre_loss[task]=-1
        
    def next_group(self):
        convergence_iter = self.convergence_iter
        
        X = self.affinity_map.clone()
        for i in range(len(X)): X[:,i] /= X[i,i]
        X = (X + X.T)/2
        
        for _ in range(10):
            cluster = AffinityPropagation(preference=self.preference, affinity='precomputed', convergence_iter=convergence_iter)
            cls = cluster.fit_predict(X)
            cluster_centers_indices = cluster.cluster_centers_indices_
            labels = cluster.labels_
            n_clusters = len(cluster_centers_indices)
            
            res={}
            for i, center in enumerate(cluster_centers_indices): 
                res['group%d'%(i+1)] = [task for j, task in enumerate(self.tasks) if labels[j]==i]
            
            if len(res) == 0: convergence_iter += 100
            else: break
        
        if len(res) != 0: self.group = res 
        # 存储当前的group和affinity_map
        torch.save(self.group, f'/home/wei/carla_garage/vis_groups2/group_{self.index}.pt')
        torch.save(self.affinity_map, f'/home/wei/carla_garage/vis_groups2/affinity_map_{self.index}.pt')
        self.index += 1
        train_group = [*self.group.keys()]
        random.shuffle(train_group)
        return train_group
