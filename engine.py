from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoConfig, AdamW, get_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CyclicLR

from transformers import RobertaForSequenceClassification

from load_data import *
from trainer import Trainer
from utils import *
import time
import wandb
import pandas as pd

def engine(cfg, args):
    seed_everything(cfg.values.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.values.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[S_ENT1]", "[E_ENT1]", "[S_ENT2]", "[E_ENT2]", 
                                                                "[S_NER1]", "[E_NER1]", "[S_NER2]", "[E_NER2]"]})
    
    ### 모델 정의
    config = AutoConfig.from_pretrained(cfg.values.model_name, num_labels = 42)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.values.model_name, config = config)
    model.resize_token_embeddings(len(tokenizer))
    
    if args.mode == 'train':
        dataset = load_train_data("/opt/ml/input/data/train/train_c.tsv")
        train_df, val_df = train_test_split(dataset, test_size = cfg.values.val_args.test_size, random_state = cfg.values.seed)

        # tokenizing
        tokenized_train = TEM_tokenized_dataset(train_df, tokenizer)
        tokenized_val = TEM_tokenized_dataset(val_df, tokenizer)

        train_dataset = RE_Dataset(tokenized_train, labels = train_df['label'].values)
        val_dataset = RE_Dataset(tokenized_val, labels = val_df['label'].values)

        ### optimizer, scheduler 정의
        # applying weight decay to all parameters other than bias and layer normalization terms
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.values.train_args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=cfg.values.train_args.lr,
            eps=cfg.values.train_args.adam_epsilon,
            )

        if cfg.values.train_args.scheduler_name == 'steplr':
            scheduler = StepLR(
                optimizer, 
                step_size = (train_dataset.__len__() // cfg.values.train_args.train_batch_size) * cfg.values.train_args.warmup_epoch, 
                gamma = cfg.values.train_args.steplr_gamma
                )

        elif cfg.values.train_args.scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, 'max', factor = cfg.values.train_args.steplr_gamma, patience = 2, cooldown = 0)

        else:
            scheduler = get_scheduler(
                cfg.values.train_args.scheduler_name, optimizer, 
                num_warmup_steps = (train_dataset.__len__() // cfg.values.train_args.train_batch_size) * cfg.values.train_args.warmup_epoch, 
                num_training_steps = train_dataset.__len__() * cfg.values.train_args.num_epochs
                )

        best_valid_loss = float('inf')
        best_valid_acc, best_model_saved_epoch = 0, 1
        for epoch in range(cfg.values.train_args.num_epochs):
            start_time = time.time() # 시작 시간 기록

            trainer = Trainer(cfg, model, epoch, optimizer, scheduler, train_dataset, val_dataset)
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.evaluate(args.mode)

            if cfg.values.train_args.scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(valid_acc)

            end_time = time.time() # 종료 시간 기록
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

            print(f'Time Spent : {elapsed_mins}m {elapsed_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {round(train_acc*100, 2)}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {round(valid_acc*100, 2)}%')

            current_lr = get_lr(optimizer)
            wandb.log({"Train Acc": round(train_acc*100, 2), "Validation Acc": round(valid_acc*100, 2),
                    "Train Loss": train_loss, "Validation Loss": valid_loss, "Learning Rate" : current_lr})

            if valid_acc > best_valid_acc:
                best_valid_acc, best_model_saved_epoch = valid_acc, epoch + 1
                torch.save(model.state_dict(), f'/opt/ml/my_code/results/{args.config}.pt')
                print('\tBetter model found!! saving the model')
        print()
        print('='*50 + f' Model Last saved from Epoch : {best_model_saved_epoch} ' + '='*50)
        print('='*50 + ' Training finished ' + '='*50)

    elif args.mode == 'inference':
        ### load my model
        model.load_state_dict(torch.load(f"/opt/ml/my_code/results/{args.config}.pt"))
        # load test datset
        test_dataset, test_label = load_test_dataset("/opt/ml/input/data/test/test.tsv", tokenizer)
        test_dataset = RE_Dataset(test_dataset, test_label)

        trainer = Trainer(cfg, model, test_dataset = test_dataset)
        pred_answer = trainer.evaluate(args.mode)

        output = pd.DataFrame(pred_answer, columns=['pred'])
        output.to_csv(f'/opt/ml/my_code/results/{args.config}_submission.csv', index=False)
        print()
        print('='*50 + ' Inference finished ' + '='*50)



