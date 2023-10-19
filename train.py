import json
from nltk_utils import tokenize, stem, bag_of_words
import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

#coleta de padrões e atribui rotulos a partir da tokenização
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        wordsTokenized = tokenize(pattern)
        all_words.extend(wordsTokenized)
        xy.append((wordsTokenized, tag))

ignore_words = ['?', '!', '.', ',']

#fazendo derivação
all_words = [stem(wordsTokenized) for wordsTokenized in all_words if wordsTokenized not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)

X_train = []
y_train = []

#criação de bag_of_words
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #CrossEntropyLoss -> 1 hot encoded

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


#Hiperparâmetros
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0]) #bag of words que foi criado -> mesmo que all_words
hidden_size = 8
output_size = len(tags) #n° de diferentes classes e tags

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0) #para iterar automaticamente em cima disso

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #verifica se tem GPU disponível
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#carregar treinamento
for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch + 1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
print(f'final loss, loss={loss.item():.4f}')