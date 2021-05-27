import pandas as pd
import numpy as np
import torch
import torch.nn as nn



df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
print(df_train.isna().sum())


class NeuralNetwork(nn.Module):
    def __init__(self, input_dimensions, num_of_neurons, num_of_neurons2):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dimensions, num_of_neurons)
        self.sigmoid = nn.ReLU()
        self.linear2 = nn.Linear(num_of_neurons, num_of_neurons2)
        self.linear3 = nn.Linear(num_of_neurons2, 1)

    def forward(self, x):
        l1 = self.linear1(x)
        activation = self.sigmoid(l1)
        l2= self.linear2(activation)
        activation = self.sigmoid(l2)
        l3 = self.linear3(activation)
        output = self.sigmoid(l3)
        #print(f"Linear 1: {l1}, Activation: {activation}, Linear 2: {l2}, output: {output}!")
        return output

X = np.random.rand(1000,10)

#print(X)
Y = np.random.randint(0, 2, 1000)
#print(Y)

def train_network(x, y, model, loss, lr, num_epochs):
    x_tensor = torch.tensor(x).float()
    print("HERE")
    print(x_tensor.shape)
    y_true_tensor = torch.tensor(y).float().view(1000,1)
    print(y_true_tensor.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):

        optimizer.zero_grad()
        y_pred_tensor = model(x_tensor)
        loss_value = loss(y_pred_tensor, y_true_tensor)
        loss_value.backward()
        optimizer.step()
        print(f"EPOCH NUMBER >>>> {epoch}, loss >>> {loss_value.item()} ")

    return model

m = NeuralNetwork(10, 5, 3)
l = nn.BCELoss()
model = train_network(X, Y, m, l, 0.01, 10000)
