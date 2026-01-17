import json
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

'''pytorch dataset은 dict같은 계층구조를 이해하지 못한다. 인덱스 하나에 환자 한명이 들어가는 구조로 되어야 datset이 이해한다

평탄화는 딥러닝이 이해할 수 있게 구조를 단순화하는 과정이다'''


def load_jsonl_as_dict(path):
    data_by_diag = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)  # row: {"id":..., "text":..., "label":...}

            diag = row["label"]
            pid = row["id"]

            data_by_diag[diag][pid] = {
                "text": row["text"],
                "label": diag
            }

    return dict(data_by_diag)


train_data = load_jsonl_as_dict("./json/train_jsonl.jsonl")
test_data  = load_jsonl_as_dict("./json/test_jsonl.jsonl")    

def flatten_label_patient_id_dict(data):
    patient_ids = []
    patient_texts = []
    patient_labels = []
    le = LabelEncoder()
    
    for label, patient_id_dict in data.items():
        for patient_id, content in patient_id_dict.items():

            # content가 dict인 경우 (정상)
            if isinstance(content, dict):
                text = content.get("text", "")
            # content가 바로 문자열인 경우도 대비
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)

            # text가 list이면 문자열로 합치기 (혹시 남아있을 경우 대비)
            if isinstance(text, list):
                text = " ".join(text)
         
            patient_ids.append(patient_id)
            patient_texts.append(text)      # ✅ 여기서 굳이 [text]로 감싸지 말자
           
            patient_labels.append(label)

    patient_labels = le.fit_transform(patient_labels)
    
    return patient_texts, patient_labels, patient_ids


class Dataset(Dataset) :
    def __init__(self, texts, labels, tokenizer_name = 'klue/bert-base', max_len = 256) :
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        if self.tokenizer.pad_token_id is None :
            raise ValueError("이 tokenizer는 pad_token_id가 없어요. (예: GPT2 계열) BERT/RoBERTa 계열로 쓰거나 pad 토큰을 설정해야 합니다.")

    def __len__(self) :
        return len(self.texts)

    def __getitem__(self, idx) :
        encoding = self.tokenizer(
            self.texts[idx],
            truncation = True,
            padding = 'max_length',
            max_length = self.max_len,
            return_tensors = 'pt')

        return {'input_ids' : encoding['input_ids'].squeeze(0),
                'attention_mask' : encoding['attention_mask'].squeeze(0),
                'labels' : torch.tensor(int(self.labels[idx]), dtype=torch.long)}
        

def Dataloader() :

    train_patient_texts, train_patient_labels, train_patient_id = flatten_label_patient_id_dict(train_data)
    test_patient_texts, test_patient_labels, test_patient_id = flatten_label_patient_id_dict(test_data)       
    
    train_dataset = Dataset(train_patient_texts, train_patient_labels, max_len = 256)
    test_dataset = Dataset(test_patient_texts, test_patient_labels, max_len = 256)


    train_loader = DataLoader(
        train_dataset,
        batch_size = 32, 
        shuffle=True,
        num_workers = 0,
        pin_memory=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle = False,
        num_workers = 0,
        pin_memory = True

)

    return train_loader, test_loader        