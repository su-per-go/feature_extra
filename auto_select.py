import json
import random
from enum import Enum
from urllib.parse import urlparse
import math

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from string import printable
import torch.optim as optim
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

seed = 43
torch.manual_seed(seed)
random.seed(seed)
# 在使用GPU时，你也需要设置CUDA随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU


class SelectConv(nn.Module):
    def __init__(self, max_vocab_len, input_dim, output_dim, max_len):
        super(SelectConv, self).__init__()
        self.embedding = nn.Embedding(max_vocab_len, input_dim)
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=5)
        self.max_pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(self.conv.out_channels * ((max_len - self.conv.kernel_size[0] + 1) // 2), 1)
        self.dropout1 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        output_conv = self.conv(x)
        output_conv = self.max_pool(output_conv)
        the_output = output_conv.view(output_conv.size(0), -1)
        return self.sigmoid(self.fc1(the_output))


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.int)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def split_url(url):
    parsed_url = urlparse(url)
    domain_name = parsed_url.netloc
    # 提取路径
    path = parsed_url.path
    if len(domain_name) == 0:
        path = path.lstrip("/")
        sp_url = path.split("/", 1)
        prefix = sp_url[0]
        suffix = sp_url[1] if len(sp_url) > 1 else ""
    else:
        prefix = parsed_url.scheme + "://" + parsed_url.netloc
        suffix = parsed_url.path + parsed_url.params + parsed_url.query + parsed_url.fragment
    return prefix, suffix


class CreateDataset:
    def __init__(self, legal_url_ls, phishing_url_ls, extra_ratio=0.1, random_state=42, batch_size=64,
                 num_folds=5):
        if len(legal_url_ls) < len(phishing_url_ls):
            max_len = math.floor(len(legal_url_ls) * extra_ratio)
        else:
            max_len = math.floor(len(phishing_url_ls) * extra_ratio)
        self.pre_url_ls = []
        self.suf_url_ls = []
        self.pre_max = 0
        self.suf_max = 0
        for url in random.sample(legal_url_ls, max_len) + random.sample(phishing_url_ls, max_len):
            result = split_url(url)
            if len(result[0]) > self.pre_max:
                self.pre_max = len(result[0])
            if len(result[1]) > self.suf_max:
                self.suf_max = len(result[1])
            self.pre_url_ls.append(result[0])
            self.suf_url_ls.append(result[1])
        self.label = [Label.LEGAL.value] * max_len + [Label.PHISHING.value] * max_len
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_folds = num_folds

        self.pre_url_ls = [self.encoded(self.pad_url(url, self.pre_max)) for url in self.pre_url_ls]
        self.suf_url_ls = [self.encoded(self.pad_url(url, self.suf_max)) for url in self.suf_url_ls]

    @staticmethod
    def encoded(url):
        return [printable.index(str_) for str_ in url]

    def split_train_test(self, data, label, random_state=42, fold=0):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(data))
        train_indices, test_indices = splits[fold]
        train_data, test_data = [data[i] for i in train_indices], [data[i] for i in test_indices]
        train_labels, test_labels = [label[i] for i in train_indices], [label[i] for i in test_indices]
        return train_data, test_data, train_labels, test_labels

    @staticmethod
    def loader_data(train_data, test_data, train_labels, test_labels, batch_size=64):
        train_dataset = CustomDataset(train_data, train_labels)
        verify_dataset = CustomDataset(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        verify_loader = DataLoader(verify_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, verify_loader

    @staticmethod
    def pad_url(url, max_length, padding_char=' '):
        if len(url) >= max_length:
            return url[:max_length]
        else:
            return url + padding_char * (max_length - len(url))

    def get_pre_load(self, fold):

        pre_split = self.split_train_test(self.pre_url_ls, self.label, self.random_state, fold)
        return self.loader_data(*pre_split, self.batch_size)

    def get_suf_load(self, fold):

        suf_split = self.split_train_test(self.suf_url_ls, self.label, self.random_state, fold)
        return self.loader_data(*suf_split, self.batch_size)

    def get_max_len(self):
        return self.pre_max, self.suf_max

    def get_num_folds(self):
        return self.num_folds


class AutoSelect:
    def __init__(self, the_dataset):
        self.the_dataset = the_dataset
        self.pre_embedding_len = 10
        self.suf_embedding_len = 10

    @staticmethod
    def train_early_stopping(model, optimizer, criterion, train_data_loader, val_data_loader, device,
                             max_epochs=200, patience=10, embedding_len=10):  # 训练
        best_val_loss = float('inf')
        no_improvement_count = 0
        f1, accuracy, recall, precision = 0, 0, 0, 0
        model.to(device)
        criterion.to(device)
        for epoch in tqdm(range(1, max_epochs + 1), desc='Training', unit='epoch'):
            for inputs, labels in train_data_loader:
                inputs = inputs[:, :embedding_len]
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_loss = 0.0
            all_val_preds = []
            all_val_labels = []
            with torch.no_grad():
                for val_inputs, val_labels in val_data_loader:
                    val_inputs = val_inputs[:, :embedding_len]
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs.squeeze(), val_labels.float()).item()
                    all_val_preds.extend((val_outputs >= 0.5).int().squeeze().cpu().numpy())
                    all_val_labels.extend(val_labels.cpu().numpy())
            val_loss /= len(val_data_loader)

            f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
            accuracy = accuracy_score(all_val_labels, all_val_preds)
            recall = recall_score(all_val_labels, all_val_preds)
            precision = precision_score(all_val_labels, all_val_preds)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if no_improvement_count >= patience:
                break
        return best_val_loss, f1, accuracy, recall, precision

    def find_best_embedding_len(self, dataset, max_select_len):
        best_val_loss = float('inf')
        no_improvement_count = 0
        best_embedding_len = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        patience = 10

        for the_len in range(10, max_select_len):
            model = SelectConv(max_vocab_len=101, input_dim=16, output_dim=64, max_len=the_len)
            optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.90, 0.99), weight_decay=1e-4)
            criterion = nn.BCELoss()
            best_embedding_len = the_len
            val_loss = self.train_early_stopping(model, optimizer, criterion, dataset[0], dataset[1], device,
                                                 embedding_len=the_len)
            if val_loss[0] < best_val_loss:
                best_val_loss = val_loss[0]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                break
        return best_embedding_len - 10

    def get_pre_best_embedding_len(self, max_len, num_folds):
        best_lengths = []
        for fold in range(num_folds):
            pre_dataset = self.the_dataset.get_pre_load(fold)
            best_length = self.find_best_embedding_len(pre_dataset, max_len)
            best_lengths.append(best_length)
        return best_lengths

    def get_suf_best_embedding_len(self, max_len, num_folds):
        best_lengths = []
        for fold in range(num_folds):
            suf_dataset = self.the_dataset.get_suf_load(fold)
            best_length = self.find_best_embedding_len(suf_dataset, max_len)
            best_lengths.append(best_length)
        return best_lengths


class Label(Enum):
    LEGAL = 0
    PHISHING = 1


class Config:
    def __init__(self, max_vocab_len, input_dim, output_dim, max_len, embedding_len, seed, batch_size):
        self.max_vocab_len = max_vocab_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.embedding_len = embedding_len
        self.seed = seed
        self.batch_size = batch_size


def auto_select_embedding_len(legal_ls, phishing_ls):
    config = Config(max_vocab_len=101, input_dim=16, output_dim=64, max_len=100, embedding_len=10, seed=42,
                    batch_size=64)

    the_dataset = CreateDataset(legal_ls, phishing_ls, batch_size=config.batch_size)
    auto_selector = AutoSelect(the_dataset=the_dataset)
    pre_best_embedding_len = auto_selector.get_pre_best_embedding_len(max_len=config.max_len, num_folds=5)
    suf_best_embedding_len = auto_selector.get_suf_best_embedding_len(max_len=config.max_len, num_folds=5)
    return sum(pre_best_embedding_len) // len(pre_best_embedding_len), sum(suf_best_embedding_len) // len(
        suf_best_embedding_len)


if __name__ == "__main__":
    with open("pilot_process/graph_info.json") as f:
        the_legal_ls = json.load(f)["after_legal_url_ls"]
    the_phishing_ls = pd.read_csv("E:/Crawling/dataset/phishing/url_info.csv")["request_url"].tolist()
    auto_select_embedding_len(the_legal_ls, the_phishing_ls)
