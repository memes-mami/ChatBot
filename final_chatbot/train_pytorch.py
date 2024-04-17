import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

import matplotlib.pyplot as plt

with open('intents2.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # Add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = tokenize(pattern)
        # Add to our words list
        all_words.extend(w)
        # Add to xy pair
        xy.append((w, tag))

# Stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: Bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # We can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store the loss and accuracy values during training
losses = []
accuracies = []

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Calculate training accuracy
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Training Accuracy: {acc:.2f}%')

    # Append current loss and accuracy to the lists
    losses.append(loss.item())
    accuracies.append(acc)

print(f'Final Loss: {loss.item():.4f}')

# Plot loss and accuracy
plt.figure(figsize=(10,5))
plt.plot(losses, label='Loss')
plt.plot(accuracies, label='Accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
