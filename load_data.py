import pickle as pickle
import os
import pandas as pd
import torch
import time

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
    #item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2], 
                              'entity_01_start':dataset[3], 'entity_01_end':dataset[4],
                              'entity_02':dataset[5], 'entity_02_start' : dataset[6],
                              'entity_02_end' : dataset[7], 'label':label,})
  return out_dataset


# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
def tokenized_dataset(dataset, tokenizer):
  sep = tokenizer.special_tokens_map["sep_token"]
  concat_entity = list(dataset['entity_01'] + sep + dataset['entity_02'])
  tokenized_sentences = tokenizer(
      concat_entity, 
      list(dataset['sentence']),
      return_tensors="pt",
      add_special_tokens=True,
      max_length=100,
      padding='max_length',
      truncation='only_second',
      )
  return tokenized_sentences


from pororo import Pororo
def TEM_tokenized_dataset(dataset, tokenizer):
  ner = Pororo(task = 'ner', lang = 'ko')
  TEM_preprocess_sent, concat_entity = [], []

  print('----Start TEM_token_processeing----')
  start_time = time.time() # 시작 시간 기록

  sep = tokenizer.special_tokens_map["sep_token"]
  for sent, ent1, ent2, start1, end1, start2, end2 in zip(dataset['sentence'], dataset['entity_01'], dataset['entity_02'],
                                                          dataset['entity_01_start'], dataset['entity_01_end'],
                                                          dataset['entity_02_start'], dataset['entity_02_end']):

    ner_01 = "[S_NER1]"+ner(ent1)[0][1].lower()+"[E_NER1]"
    ner_02 = "[S_NER2]"+ner(ent2)[0][1].lower()+"[E_NER2]"

    entity_01_start, entity_01_end = int(start1), int(end1)
    entity_02_start, entity_02_end = int(start2), int(end2)

    if entity_01_start < entity_02_start:
      sent = sent[:entity_01_start] + "[S_ENT1]" + ner_01 + ent1 +"[E_ENT1]" + sent[entity_01_end+1:entity_02_start]+\
             "[S_ENT2]" + ner_02 + ent2 + "[E_ENT2]" + sent[entity_02_end+1:]
    else:
      sent = sent[:entity_02_start]+"[S_ENT2]"+ner_02+sent[entity_02_start:entity_02_end+1]+"[E_ENT2]"+sent[entity_02_end+1:entity_01_start]+\
             "[S_ENT1]"+ner_01+sent[entity_01_start:entity_01_end+1]+"[E_ENT1]"+sent[entity_01_end+1:]

    concat_entity.append("[S_ENT1]" + ner_01 + ent1 + "[E_ENT1]" + sep + "[S_ENT2]" + ner_02 + ent2 + "[E_ENT2]") 
    TEM_preprocess_sent.append(sent)

  #sep = tokenizer.special_tokens_map["sep_token"]
  #concat_entity = list("[S_ENT1]" + dataset['entity_01'] + "[E_ENT1]" + sep + "[S_ENT2]" + dataset['entity_02'] + "[E_ENT2]")

  tokenized_sentences = tokenizer(
      concat_entity,
      TEM_preprocess_sent,
      return_tensors="pt",
      add_special_tokens=True,
      max_length=128,
      padding='max_length',
      truncation='only_second',
      )

  end_time = time.time() # 종료 시간 기록
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

  print(f'\tTime Spent : {elapsed_mins}m {elapsed_secs}s')
  print('----Finished TEM_token_processeing----')

  return tokenized_sentences
                                                                                                               

# tsv 파일을 불러옵니다.
def load_train_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  return dataset


def load_test_dataset(dataset_path, tokenizer):
  # load dataset
  dataset = pd.read_csv(dataset_path, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, None)
  test_label = dataset['label'].values
  tokenized_test = TEM_tokenized_dataset(dataset, tokenizer)
  return tokenized_test, test_label