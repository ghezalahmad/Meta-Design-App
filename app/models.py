
import torch
import torch.optim as optim  # Import PyTorch's optimizer module


# Define MAML model
class MAMLModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(MAMLModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)
    
def meta_train(meta_model, data, input_columns, target_columns, epochs, inner_lr, outer_lr, num_tasks=5, hidden_size=128):
    """
    Meta-train the MAML model using simulated tasks.

    Args:
        meta_model (MAMLModel): The MAML model to train.
        data (pd.DataFrame): The dataset containing input and target columns.
        input_columns (list): Columns used as input features.
        target_columns (list): Columns used as target properties.
        epochs (int): Number of meta-training epochs.
        inner_lr (float): Learning rate for the inner loop.
        outer_lr (float): Learning rate for the outer loop.
        num_tasks (int): Number of tasks to simulate.

    Returns:
        MAMLModel: The trained meta-model.
    """
    optimizer = optim.Adam(meta_model.parameters(), lr=outer_lr)
    loss_function = torch.nn.MSELoss()

    for epoch in range(epochs):
        meta_loss = 0.0

        for task in range(num_tasks):
            # Simulate a task by sampling a subset of the data
            task_data = data.sample(frac=0.2)  # Use 20% of the data for this task
            inputs = torch.tensor(task_data[input_columns].values, dtype=torch.float32)
            targets = torch.tensor(task_data[target_columns].values, dtype=torch.float32)

            # Split into support set (inner loop) and query set (outer loop)
            num_support = int(len(inputs) * 0.8)
            support_inputs, query_inputs = inputs[:num_support], inputs[num_support:]
            support_targets, query_targets = targets[:num_support], targets[num_support:]

            # Inner loop: Task-specific adaptation
            task_model = MAMLModel(len(input_columns), len(target_columns), hidden_size=hidden_size)
            task_model.load_state_dict(meta_model.state_dict())  # Clone the meta-model
            task_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)

            for _ in range(5):  # Inner loop iterations
                task_predictions = task_model(support_inputs)
                task_loss = loss_function(task_predictions, support_targets)
                task_optimizer.zero_grad()
                task_loss.backward()
                task_optimizer.step()

            # Outer loop: Meta-optimization
            query_predictions = task_model(query_inputs)
            query_loss = loss_function(query_predictions, query_targets)
            meta_loss += query_loss

        # Update meta-model parameters
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Meta-Loss: {meta_loss.item():.4f}")

    return meta_model