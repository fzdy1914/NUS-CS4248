# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import sys

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORD_EMBEDDING_DIM = 128

CHAR_EMBEDDING_DIM = 128
CHAR_KERNEL_SIZE = 3
CHAR_KERNEL_NUM = 128

LSTM_HIDDEN_DIM = 128
LSTM_N_LAYERS = 2
DROPOUT_RATE = 0.2


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


def tag_sentence(test_file, model_file, out_file):
    f = open(test_file, 'r')
    lines = f.readlines()
    f.close()

    word_idx_map, char_idx_map, tag_idx_map, state_dict = torch.load(model_file)

    word_pad_idx = len(word_idx_map)
    word_vocab_size = word_pad_idx + 1

    char_pad_idx = len(char_idx_map)
    char_vocab_size = char_pad_idx + 1

    tag_pad_idx = len(tag_idx_map)
    lstm_output_dim = tag_pad_idx + 1

    idx_tag_map = {}
    for tag in tag_idx_map:
        idx_tag_map[tag_idx_map[tag]] = tag

    model = POSTagger(word_vocab_size, WORD_EMBEDDING_DIM, word_pad_idx,
                      char_vocab_size, CHAR_EMBEDDING_DIM, CHAR_KERNEL_SIZE, CHAR_KERNEL_NUM, char_pad_idx,
                      LSTM_HIDDEN_DIM, lstm_output_dim, LSTM_N_LAYERS,
                      DROPOUT_RATE)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    result = ''
    for line in lines:
        raw_words = line.strip().split(' ')
        filtered_words = list()
        chars_words = list()

        max_chars_word_len = 0
        for word in raw_words:
            if len(word) > max_chars_word_len:
                max_chars_word_len = len(word)

        for word in raw_words:
            chars_word = list()

            if word not in word_idx_map:
                filtered_words.append(word_idx_map['_UNK_'])
            else:
                filtered_words.append(word_idx_map[word])

            for char in word:
                if char not in char_idx_map:
                    chars_word.append(char_pad_idx)
                else:
                    chars_word.append(char_idx_map[char])
            chars_word.extend([char_pad_idx] * (max_chars_word_len - len(chars_word)))
            chars_words.append(chars_word)

        words_tensor = torch.tensor(filtered_words).long().view(-1, 1).to(device)
        chars_tensor = torch.tensor(chars_words).long().view(-1, 1, max_chars_word_len).to(device)

        outputs = model(words_tensor, chars_tensor)
        outputs = outputs.view(-1, outputs.shape[-1])

        word_tag_idx = torch.argmax(outputs, dim=1)
        for i in range(len(word_tag_idx)):
            word = raw_words[i]
            tag = idx_tag_map[word_tag_idx[i].item()]
            result += word + '/' + tag + ' '
        result += '\n'

    with open(out_file, 'w') as f:
        f.write(result)
    f.close()

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
