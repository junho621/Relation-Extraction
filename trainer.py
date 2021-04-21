import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import wandb
from utils import *
from loss import *
from load_data import *

class Trainer(object):
    def __init__(self, cfg, model, epoch=None, optimizer=None, scheduler=None, train_dataset=None, test_dataset=None):
        self.cfg = cfg
        self.epoch = epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = create_criterion(self.cfg.values.train_args.criterion)

    def train(self):
        train_dataset = DataLoader(self.train_dataset, batch_size=self.cfg.values.train_args.train_batch_size, shuffle = True)
        global_step = 0
        epoch_loss, epoch_acc = 0.0, 0.0
        self.model.train()
        with tqdm(train_dataset, total = len(train_dataset), unit = 'batch') as train_bar:
            for step, batch in enumerate(train_bar):
                inputs = {key : value.to(self.device) for key, value in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(**inputs)
                loss = self.criterion(pred.logits, labels)
                loss.backward()
                correct = (torch.argmax(pred.logits, dim=1)  == labels).sum().item()
                acc = correct / len(batch['input_ids']) 
                epoch_acc += acc

                epoch_loss += loss.item()
                if (step + 1) % self.cfg.values.train_args.gradient_accumulation_steps == 0:
                    global_step += 1
                    self.optimizer.step()
                    if self.cfg.values.train_args.scheduler_name != 'ReduceLROnPlateau':
                        self.scheduler.step()  # Update learning rate schedule
                current_lr = get_lr(self.optimizer)
                
                # update progress bar
                train_bar.set_description(f'Training Epoch [{self.epoch + 1} / {self.cfg.values.train_args.num_epochs}]')
                train_bar.set_postfix(loss = loss.item(), acc = acc*100, current_lr = current_lr)

        return epoch_loss / global_step, epoch_acc / len(train_dataset)

    def evaluate(self, mode):
        self.model.eval()
        if mode == 'train':
            test_dataset = DataLoader(self.test_dataset, batch_size=self.cfg.values.train_args.eval_batch_size, shuffle=True)
            eval_loss, eval_acc = 0.0, 0.0
            nb_eval_steps = 0
            with tqdm(test_dataset, total = len(test_dataset), unit = 'Evaluating') as eval_bar:
                with torch.no_grad():
                    for batch in eval_bar:
                        inputs = {key : value.to(self.device) for key, value in batch.items() if key != 'labels'}
                        labels = batch['labels'].to(self.device)
                        output = self.model(**inputs)
                        pred = output.logits
                        loss = self.criterion(pred, labels)
                        
                        # val acc 계산
                        correct = (torch.argmax(pred, dim=1)  == labels).sum().item()
                        acc = correct / len(batch['input_ids']) 
                        eval_acc += acc

                        # 전체 손실 값 계산
                        eval_loss += loss.item()

                        # update progress bar
                        eval_bar.set_description(f'Evaluating [{self.epoch + 1} / {self.cfg.values.train_args.num_epochs}]')
                        eval_bar.set_postfix(loss = loss.item(), acc = eval_acc)

            return eval_loss / len(test_dataset), eval_acc / len(test_dataset)
        
        elif mode == 'inference':
            inference_dataset = DataLoader(self.test_dataset, batch_size=1, shuffle=False)
            output_pred = []

            with tqdm(inference_dataset, total = len(inference_dataset), unit = 'inference') as inference_bar:
                with torch.no_grad():
                    for batch in inference_bar:
                        inputs = {key : value.to(self.device) for key, value in batch.items() if key != 'labels'}
                        pred = self.model(**inputs)
                        result = np.argmax(pred.logits.detach().cpu().numpy(), axis=-1) 
                        output_pred.append(result)
            
            return np.array(output_pred).flatten()

