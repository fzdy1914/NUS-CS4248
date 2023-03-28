# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

word_idx_map = {'_UNK_': 0}
char_idx_map = dict()
tag_idx_map = dict()

WORD_EMBEDDING_DIM = 128

CHAR_EMBEDDING_DIM = 128
CHAR_KERNEL_SIZE = 3
CHAR_KERNEL_NUM = 128

LSTM_HIDDEN_DIM = 128
LSTM_N_LAYERS = 2
DROPOUT_RATE = 0.2

NUM_EPOCHS = 10
BATCH_SIZE = 64


class POSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class POSTagger(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_dim, word_pad_idx,
                 char_vocab_size, char_embedding_dim, kernel_size, kernel_num, char_pad_idx,
                 lstm_hidden_dim, lstm_output_dim, lstm_n_layers,
                 dropout_rate):
        super().__init__()

        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=word_pad_idx)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=char_pad_idx)
        self.char_embedding_dim = char_embedding_dim
        self.kernel_num = kernel_num

        self.conv = nn.Conv1d(in_channels=char_embedding_dim, out_channels=kernel_num, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2)

        self.lstm = nn.LSTM(word_embedding_dim + kernel_num, lstm_hidden_dim, num_layers=lstm_n_layers,
                            bidirectional=True)

        self.fc = nn.Linear(lstm_hidden_dim * 2, lstm_output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, words, chars_words):
        words_embedded = self.dropout(self.word_embedding(words))

        chars_words_len = chars_words.shape[0]
        batch_size = chars_words.shape[1]
        chars_word_len = chars_words.shape[2]

        chars_words_embedded = self.char_embedding(chars_words)
        chars_words_embedded = chars_words_embedded.view(-1, chars_word_len, self.char_embedding_dim).permute(0, 2, 1)

        chars_words_convolved = self.conv(chars_words_embedded)
        chars_words_convolved = chars_words_convolved.view(chars_words_len, batch_size, self.kernel_num, chars_word_len)

        chars_embedded, _ = torch.max(chars_words_convolved, dim=-1)
        chars_embedded = self.dropout(chars_embedded)

        full_embedded = torch.cat((words_embedded, chars_embedded), dim=-1)
        lstm_outputs, _ = self.lstm(full_embedded)

        outputs = self.fc(self.dropout(lstm_outputs))
        return outputs


def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for words, chars_words, tags in data_loader:
        optimizer.zero_grad()

        outputs = model(words, chars_words)
        outputs = outputs.view(-1, outputs.shape[-1])
        tags = tags.view(-1)

        loss = criterion(outputs, tags)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0

    for words, chars_words, tags in data_loader:
        outputs = model(words, chars_words)
        outputs = outputs.view(-1, outputs.shape[-1])
        tags = tags.view(-1)

        loss = criterion(outputs, tags)
        total_loss += loss.item()

    return total_loss / len(data_loader)


def collate_fn(batch):
    word_pad_idx = len(word_idx_map)
    char_pad_idx = len(char_idx_map)
    tag_pad_idx = len(tag_idx_map)

    max_words_len = 0
    max_chars_word_len = 0
    for words, chars_words, _ in batch:
        if len(words) > max_words_len:
            max_words_len = len(words)
        for chars_word in chars_words:
            if len(chars_word) > max_chars_word_len:
                max_chars_word_len = len(chars_word)

    words_list = list()
    chars_words_list = list()
    tags_list = list()

    chars_word_padding = [char_pad_idx for _ in range(max_chars_word_len)]
    for words, chars_words, tags in batch:
        words_idx_list = list()
        chars_words_idx_list = list()
        tags_idx_list = list()

        for word in words:
            words_idx_list.append(word_idx_map[word])
        words_idx_list.extend([word_pad_idx] * (max_words_len - len(words_idx_list)))
        words_list.append(words_idx_list)

        for chars_word in chars_words:
            char_idx_list = list()
            for char in chars_word:
                char_idx_list.append(char_idx_map[char])
            char_idx_list.extend([char_pad_idx] * (max_chars_word_len - len(char_idx_list)))
            chars_words_idx_list.append(char_idx_list)
        chars_words_idx_list.extend([chars_word_padding] * (max_words_len - len(chars_words_idx_list)))
        chars_words_list.append(chars_words_idx_list)

        for tag in tags:
            tags_idx_list.append(tag_idx_map[tag])
        tags_idx_list.extend([tag_pad_idx] * (max_words_len - len(tags_idx_list)))
        tags_list.append(tags_idx_list)

    words_tensor = torch.tensor(words_list).permute(1, 0).long().to(device)
    chars_words_tensor = torch.tensor(chars_words_list).permute(1, 0, 2).long().to(device)
    tags_tensor = torch.tensor(tags_list).permute(1, 0).long().to(device)

    return words_tensor, chars_words_tensor, tags_tensor


def train_model(train_file, model_file):
    f = open(train_file, 'r')
    lines = f.readlines()
    f.close()

    training_data = list()
    word_frequency_map = dict()
    char_set = set()
    tag_set = set()

    for line in lines:
        words = list()
        chars_words = list()
        tags = list()
        pairs = line.strip().split(' ')

        for pair in pairs:
            word, tag = pair.rsplit('/', 1)
            if word not in word_frequency_map:
                word_frequency_map[word] = 1
            else:
                word_frequency_map[word] += 1

            words.append(word)

            chars_word = list()
            for char in word:
                chars_word.append(char)
                char_set.add(char)
            chars_words.append(chars_word)

            tags.append(tag)
            tag_set.add(tag)

        training_data.append((words, chars_words, tags))

    filtered_training_data = list()
    for words, chars_words, tags in training_data:
        filtered_words = list()
        for word in words:
            if word_frequency_map[word] == 1:
                filtered_words.append('_UNK_')
            else:
                filtered_words.append(word)
        filtered_training_data.append((filtered_words, chars_words, tags))

    for word in word_frequency_map:
        word_idx_map[word] = len(word_idx_map)

    for char in char_set:
        char_idx_map[char] = len(char_idx_map)

    for tag in tag_set:
        tag_idx_map[tag] = len(tag_idx_map)

    training_data = filtered_training_data[:int(len(filtered_training_data) * 0.9)]
    training_data = sorted(training_data, reverse=True, key=lambda x: len(x[0]))
    train_data_loader = DataLoader(POSDataset(data=training_data), batch_size=BATCH_SIZE, collate_fn=collate_fn)

    validation_data = filtered_training_data[int(len(filtered_training_data) * 0.9):]
    validation_data = sorted(validation_data, reverse=True, key=lambda x: len(x[0]))
    validation_data_loader = DataLoader(POSDataset(data=validation_data), batch_size=BATCH_SIZE, collate_fn=collate_fn)

    word_pad_idx = len(word_idx_map)
    word_vocab_size = word_pad_idx + 1

    char_pad_idx = len(char_idx_map)
    char_vocab_size = char_pad_idx + 1

    tag_pad_idx = len(tag_idx_map)
    lstm_output_dim = tag_pad_idx + 1

    model = POSTagger(word_vocab_size, WORD_EMBEDDING_DIM, word_pad_idx,
                      char_vocab_size, CHAR_EMBEDDING_DIM, CHAR_KERNEL_SIZE, CHAR_KERNEL_NUM, char_pad_idx,
                      LSTM_HIDDEN_DIM, lstm_output_dim, LSTM_N_LAYERS,
                      DROPOUT_RATE)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=tag_pad_idx)

    for name, param in model.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)
    model = model.to(device)

    best_validation_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_data_loader, optimizer, criterion)
        validation_loss = evaluate(model, validation_data_loader, criterion)
        print(epoch, ", ", train_loss, ", ", validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save((word_idx_map, char_idx_map, tag_idx_map, model.state_dict()), model_file)

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
