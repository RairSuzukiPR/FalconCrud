import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.out = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

    # def save(self, file_name='model.pth'):
    #     model_folder_path = './model'
    #     if not os.path.exists(model_folder_path):
    #         os.makedirs(model_folder_path)
    #
    #     file_name = os.path.join(model_folder_path, file_name)
    #     torch.save(self.state_dict(), file_name)


class Trainer:
    def __init__(self, model, lr, epochs):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        torch.manual_seed(22)

        dados = pd.read_csv('training_data_tracker2.csv', sep=',', header=0)
        X = dados.drop('comportamento', axis=1)
        y = dados['comportamento']

        X = X.values
        y = y.values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=22)

        # Convert X features to float tensors
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)

        # Convert y labels to float tensors
        self.y_train = torch.LongTensor(self.y_train)
        self.y_test = torch.LongTensor(self.y_test)

        # Train the model
        losses = []

        for i in range(self.epochs):
            # Forward pass
            y_pred = self.model.forward(self.X_train)

            # Compute loss
            loss = self.criterion(y_pred, self.y_train)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track losses
            losses.append(loss.item())

            if i % (self.epochs * 0.1) == 0:
                print(f'Epoch: {i} Loss: {loss.item()}')

        with torch.no_grad():  # turn off backpropogation
            y_eval = self.model.forward(self.X_test)  # X_test are features from our test set, y_eval will be predictions
            loss = self.criterion(y_eval, self.y_test)

        correct = 0
        wrong = 0
        with torch.no_grad():
            for i, data in enumerate(self.X_test):
                y_val = self.model.forward(data)
                if y_val.argmax().item() == self.y_test[i]:
                    correct += 1
                else:
                    wrong += 1

        print(f'We got {correct} correct and {wrong} wrong!, with an accuracy of {(correct / (correct + wrong)) * 100}')
        return f'We got {correct} correct and {wrong} wrong!, with an accuracy of {(correct / (correct + wrong)) * 100}'
