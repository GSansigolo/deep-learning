import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define LSTM classifier model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Generate sample data
X_train = np.random.rand(100, 10, 50)  # 100 sequences of length 10 with 50 features each
y_train = np.random.randint(2, size=(100,))  # Binary labels (0 or 1)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
print(X_train_tensor.shape)

# Define model parameters
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 2
output_size = 2

# Instantiate the model
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Example of making predictions (not included in the final blog post)
X_test = np.random.rand(10, 10, 50)  # Test data with 10 sequences
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)
    print("Predicted Labels:", predicted_labels)