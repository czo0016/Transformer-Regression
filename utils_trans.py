import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler




def get_data():
    # train logs contains about 5000 logs without labels
    train_logs = pd.read_csv("./linking-writing-processes-to-writing-quality/train_logs.csv")
    # train scores only contain labels for each id
    train_scores = pd.read_csv("./linking-writing-processes-to-writing-quality/train_scores.csv")
    test_logs = pd.read_csv("./linking-writing-processes-to-writing-quality/test_logs.csv")

    return train_logs, train_scores, test_logs



def preprocess(train_logs, train_scores, test_logs):
    # set index using id
    train_scores = train_scores.set_index(["id"])

    train_logs = train_logs.fillna('X')

    train_logs['pause_time'] = train_logs.groupby('id')['down_time'].diff()

    train_logs['pause_time'] = train_logs['pause_time'].fillna(0)

    # remove categorical attributes
    train_logs = train_logs.drop(columns=["down_time", "up_time"])

    # replace move with something simplier
    categories_in_activity = ["Nonproduction", "Input", "Remove/Cut", "Replace", "Paste"]
    train_logs.loc[~train_logs["activity"].isin(categories_in_activity), "activity"] = "Move"

    # set index using id attribute
    train_logs = train_logs.set_index(["id"])

    return train_logs, train_scores, test_logs

class TrainingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = self._generate_senquence_(data)
        self.labels = labels.values

    def _generate_senquence_(self, data):
        sequences_data = []
        grouped_data = data.groupby(["id"])
        for _, x in grouped_data:
            sequences_data.append(x.values)
        return sequences_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = torch.tensor(self.labels[idx])
        return data, labels

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return sequences, torch.stack(labels)

def padding(sequence, max_length):
    pad_zeros = torch.zeros([max_length - len(sequence), sequence.shape[1]], dtype=float)
    padded_seq = torch.cat((pad_zeros, sequence))
    return padded_seq

def create_training_dataloader(batch_size):
    train_logs, train_scores, test_logs = get_data()

    train_logs, train_scores, test_logs = preprocess(train_logs, train_scores, test_logs)

    training_dataset = TrainingDataset(train_logs, train_scores)

    # Define the sizes for training and validation sets
    dataset_size = len(training_dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = dataset_size - train_size  # 20% for validation

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(training_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Now you can create DataLoader instances for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, len(training_dataset)
