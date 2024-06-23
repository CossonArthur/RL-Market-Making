import torch
import torch.nn as nn
import torch.optim as optim

# Example data dimensions
input_dim = 20  # dimension of input data
num_classes = 5  # number of classes for classification


# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


# Instantiate the model
model = Net(input_dim, num_classes)

# Print a summary of the model architecture
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Print the summary of the model architecture
print(model)
