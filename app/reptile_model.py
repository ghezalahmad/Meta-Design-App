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
    """
    Train the Reptile model using all labeled data.

    Args:
        model (torch.nn.Module): The Reptile model.
        data (pd.DataFrame): The dataset with both labeled and unlabeled samples.
        input_columns (list): List of input feature column names.
        target_columns (list): List of target property column names.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        num_tasks (int): Number of simulated tasks (not used in this implementation).

    Returns:
        torch.nn.Module: The trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    # Use all labeled data
    labeled_data = data.dropna(subset=target_columns).sort_index()


    # Prepare labeled inputs and targets
    inputs = torch.tensor(labeled_data[input_columns].values, dtype=torch.float32)
    targets = torch.tensor(labeled_data[target_columns].values, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()  # Ensure the model is in training mode
        predictions = model(inputs)  # Forward pass
        loss = loss_function(predictions, targets)  # Compute loss

        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        # Log progress
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model



