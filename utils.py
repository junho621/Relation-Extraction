import random
import numpy as np
import torch
import datetime
from sklearn.metrics import accuracy_score
import yaml
from easydict import EasyDict


def get_timestamp():
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=KST)
    now2str = now.strftime("%Y/%m/%d %H:%M")
    return now2str

def seed_everything(seed):
    np.random.seed(seed) # numpy 관련 연산 무작위 고정 
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed) # cpu 연산 무작위 고정 
    torch.cuda.manual_seed(seed) # gpu 연산 무작위 고정 
    torch.cuda.manual_seed_all(seed)  # 멀티 gpu 연산 무작위 고정 
    #torch.backends.cudnn.enabled = False # cudnn library를 사용하지 않게 만듬

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return acc

# Set Config
class YamlConfigManager:
  def __init__(self, config_file_path, config_name):
    super().__init__()
    self.values = EasyDict()        
    if config_file_path:
      self.config_file_path = config_file_path
      self.config_name = config_name
      self.reload()
  
  def reload(self):
    self.clear()
    if self.config_file_path:
      with open(self.config_file_path, 'r') as f:
        self.values.update(yaml.safe_load(f)[self.config_name])

  def clear(self):
    self.values.clear()
    
  def update(self, yml_dict):
    for (k1, v1) in yml_dict.items():
      if isinstance(v1, dict):
        for (k2, v2) in v1.items():
          if isinstance(v2, dict):
            for (k3, v3) in v2.items():
              self.values[k1][k2][k3] = v3
          else:
            self.values[k1][k2] = v2
      else:
        self.values[k1] = v1

  def export(self, save_file_path):
    if save_file_path:
      with open(save_file_path, 'w') as f:
        yaml.dump(dict(self.values), f)

