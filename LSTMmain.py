import torch
import torch.nn as nn
import pandas as pd
import csv
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import numpy as np






# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs, dtype=torch.long)
#
#
# training_data = [
#     # Tags are: DET - determiner; NN - noun; V - verb
#     # For example, the word "The" is a determiner
#     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
#     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# ]
# word_to_ix = {}
# # For each words-list (sentence) and tags-list in each tuple of training_data
# for sent, tags in training_data:
#     print(sent, tags)
#     for word in sent:
#         if word not in word_to_ix:  # word has not been assigned an index yet
#             word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
# print(word_to_ix)
# tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index
#
# # These will usually be more like 32 or 64 dimensional.
# # We will keep them small, so we can see how the weights change as we train.
# EMBEDDING_DIM = 6
# HIDDEN_DIM = 6
#
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(13, 20)

        # The linear layer that maps from hidden state space to tag space
        #self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence)
        #lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        lstm_out, _ = self.lstm(sentence)
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(lstm_out, dim=1)
        return tag_scores
#
#

# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# # Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)
#
# for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in training_data:
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         model.zero_grad()
#
#         # Step 2. Get our inputs ready for the network, that is, turn them into
#         # Tensors of word indices.
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         targets = prepare_sequence(tags, tag_to_ix)
#
#         # Step 3. Run our forward pass.
#         tag_scores = model(sentence_in)
#
#         # Step 4. Compute the loss, gradients, and update the parameters by
#         #  calling optimizer.step()
#         loss = loss_function(tag_scores, targets)
#         loss.backward()
#         optimizer.step()
#
# # See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#
#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(tag_scores)

#data_Reader()
EMBEDDING_DIM = 13
HIDDEN_DIM = 13

data_frame = pd.read_csv('./AfterProcessingData.csv')
print(data_frame.head())

# 定义窗口大小和 LSTM 模型的参数
window_size = 3
num_features = 15
batch_size = 1
epochs = 100

# 将时间序列数据转换为监督式学习数据集
X, y = [], []
for i in range(window_size, len(data_frame)):
    X.append(data_frame[i-window_size:i])
    y.append(data_frame[i])
X, y = np.array(X), np.array(y)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()


dataset = torch.tensor(data_frame.values, dtype=torch.float32)
training_data = dataset[0:7000]
test_data = dataset[7001:-1]

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(dataset[0]), len(dataset[1]))  # 0 -> input 1-> GT
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
input = training_data[1].unsqueeze(dim=0).unsqueeze(dim=0)
print(input.size())
h0 = torch.zeros(1, 1, 13)
c0 = torch.zeros(1, 1, 13)
model = nn.LSTM(13, 13, 1)
prediction, _ = model(input, (h0, c0))


for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
    for index, line in enumerate(training_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        if index + 1 < len(training_data):
            model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.

        # Step 3. Run our forward pass.
            print(line[1])
            lstm_out, _ = model(line.unsqueeze(dim=0).unsqueeze(dim=0))
            #tag_scores = F.log_softmax(lstm_out, dim=1)

            # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
            target = training_data[index+1].unsqueeze(dim=0)
            loss = loss_function(lstm_out, target)
            loss.backward()
            optimizer.step()
            #print(loss)

prediction, _ = model(test_data[0].unsqueeze(dim=0).unsqueeze(dim=0), (h0, c0))
print(prediction)
# rnn = nn.LSTM(10, 20, 1)
# input = torch.randn(5, 1, 10) # (sequence length, batch_size, input_size), i guess it means for 1 inputs, it has 5 sequence
# h0 = torch.randn(1, 1, 20) # (layer_number, batch_size, hidden_size)
# c0 = torch.randn(1, 1, 20)
# output, (hn, cn) = rnn(input, (h0, c0))

#with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)
#lstm = nn.LSTM(13, 13)
# hidden = (torch.randn(1, 1, 13), torch.randn(1, 1, 13))
# for i in inputs:
#     i = i.float()
#     out, hidden = lstm(i.view(1, 1, -1), hidden)
#
# inputs = inputs[None, :]
# # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# N, W, H = inputs.size()
#
# hidden = (torch.randn(1, W, 13), torch.randn(1, W, 13))  # clean out hidden state
# out, hidden = lstm(inputs, hidden)
# print(out)
# # print(hidden)