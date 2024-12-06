# reptile_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Define Reptile model
class ReptileModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(ReptileModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

def reptile_train(model, data, input_columns, target_columns, epochs, learning_rate, num_tasks):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    for epoch in range(epochs):
        meta_loss = 0.0

        for _ in range(num_tasks):
            # Simulate a task
            task_data = data.sample(frac=0.2)  # Sample 20% of data for this task
            inputs = torch.tensor(task_data[input_columns].values, dtype=torch.float32)
            targets = torch.tensor(task_data[target_columns].values, dtype=torch.float32)

            # Forward pass
            predictions = model(inputs)
            task_loss = loss_function(predictions, targets)

            # Backpropagation
            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()

            # Aggregate loss for reporting
            meta_loss += task_loss.item()

        # Log epoch progress
        print(f"Epoch {epoch+1}/{epochs}, Meta-Loss: {meta_loss / num_tasks:.4f}")

    return model

