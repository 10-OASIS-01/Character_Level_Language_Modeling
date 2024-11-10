# data/dataset.py

import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def get_vocab_size(self):
        return max(self.data) + 1  # Assuming data contains indices starting from 0

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        # Get input and target sequences
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

def create_datasets(input_file, block_size):
    # Read the input text file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    # Get all unique characters in the text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Number of characters in the dataset: {len(data)}")
    print(f"Number of unique characters in the vocabulary: {vocab_size}")
    print("Vocabulary:")
    print(''.join(chars))

    # Create mappings from characters to integers and vice versa
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode the entire dataset
    data_idx = [stoi[ch] for ch in data]

    # Split the data into train and test sets
    n = int(0.9 * len(data_idx))
    train_data = data_idx[:n]
    test_data = data_idx[n:]

    # Wrap in dataset objects
    train_dataset = CharDataset(train_data, block_size)
    test_dataset = CharDataset(test_data, block_size)

    # Save the vocabulary mappings in the dataset for decoding
    train_dataset.stoi = stoi
    train_dataset.itos = itos
    test_dataset.stoi = stoi
    test_dataset.itos = itos

    return train_dataset, test_dataset
