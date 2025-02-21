import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import streamlit as st
from app.utils import calculate_utility, calculate_novelty, calculate_uncertainty
# Import visualization function
from app.visualization import visualize_exploration_exploitation


class MAMLModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(MAMLModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, output_size)
        )


    
    def forward(self, x):
        return self.network(x)





def meta_train(meta_model, data, input_columns, target_columns, epochs, inner_lr, outer_lr, num_tasks, inner_lr_decay, curiosity=0):
    optimizer = optim.Adam(meta_model.parameters(), lr=outer_lr, weight_decay=1e-3, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * 3, eta_min=1e-5)

    loss_function = nn.SmoothL1Loss()

    labeled_data = data.dropna(subset=target_columns).sample(frac=1).reset_index(drop=True)

    # Validate labeled data
    if labeled_data.empty:
        print("⚠️ Labeled data is empty after preprocessing. Check if the dataset has enough labeled samples.")
        return meta_model, scaler_inputs, scaler_targets


    scaler_inputs = StandardScaler().fit(labeled_data[input_columns])
    scaler_targets = StandardScaler().fit(labeled_data[target_columns])

    inputs = torch.tensor(scaler_inputs.transform(labeled_data[input_columns]), dtype=torch.float32)
    targets = torch.tensor(scaler_targets.transform(labeled_data[target_columns]), dtype=torch.float32)

    print(f"Input Data Shape: {data.shape}")
    print(f"Labeled Data Shape: {labeled_data.shape}")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")

    if len(inputs) == 0 or len(targets) == 0:
        print("❌ No valid data available for training! Please check the dataset.")
        return meta_model, scaler_inputs, scaler_targets

    batch_size = min(8, len(inputs) // 2)  # Adapt to half of the dataset size
    num_tasks = max(2, min(num_tasks, len(inputs) // batch_size))

    


    num_support = int(len(inputs) * 0.7)
    support_indices = np.random.choice(len(inputs), size=num_support, replace=False)
    query_indices = np.setdiff1d(np.arange(len(inputs)), support_indices)
    support_inputs, query_inputs = inputs[support_indices], inputs[query_indices]
    support_targets, query_targets = targets[support_indices], targets[query_indices]

    best_loss = float('inf')
    best_model_state = None
    patience = 15
    best_loss_counter = 0

    min_inner_lr = inner_lr * 0.7  
    decayed_inner_lr = inner_lr

    for epoch in range(epochs):
        meta_model.train()
        meta_loss = torch.zeros(1, device=next(meta_model.parameters()).device, requires_grad=True)

        decayed_inner_lr = max(inner_lr * (inner_lr_decay ** epoch), min_inner_lr)
        inner_loop_steps = min(32, len(inputs) * 4)  

        for task in range(num_tasks):
            task_model = copy.deepcopy(meta_model)
            task_optimizer = optim.Adam(task_model.parameters(), lr=decayed_inner_lr, weight_decay=1e-4)

            for _ in range(inner_loop_steps):
                task_predictions = task_model(support_inputs)
                task_loss = loss_function(task_predictions, support_targets)
                
                # ✅ Refined curiosity scaling
                scaling_factor = (1 + 0.1 * torch.tanh(torch.tensor(0.2 * curiosity, dtype=torch.float32)))
                task_loss = task_loss * scaling_factor

                task_optimizer.zero_grad()
                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=0.5)  
                task_optimizer.step()


            print(f"Task Loss Requires Grad: {task_loss.requires_grad}")

            query_predictions = task_model(query_inputs)
            query_loss = loss_function(query_predictions, query_targets)

            query_loss = query_loss * (1 + 0.2 * torch.tanh(torch.tensor(0.3 * curiosity, dtype=torch.float32)))
            print(f"Query Loss Requires Grad: {query_loss.requires_grad}")
            print(f"Task {task+1}/{num_tasks} - Task Loss: {task_loss.item():.4f}, Query Loss: {query_loss.item():.4f}")

            meta_loss = meta_loss + query_loss

            # Early Stopping Logic
            if query_loss.item() < best_loss:
                best_loss = query_loss.item()
                best_model_state = copy.deepcopy(meta_model.state_dict())
                best_loss_counter = 0
            else:
                best_loss_counter += 1

        optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step(best_loss)  # ✅ Correctly pass the monitored metric
        print(f"Meta Loss Requires Grad: {meta_loss.requires_grad}")


        print(f"Epoch {epoch+1}/{epochs} - Meta Loss: {meta_loss.item():.4f}, Best Loss: {best_loss:.4f}")
        print(f"Inner LR: {decayed_inner_lr:.6f}, Outer LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Tasks per Epoch: {num_tasks}, Batch Size: {batch_size}")


        # Check if early stopping should be triggered
        dynamic_patience = max(15, patience - (epoch // 20))  # Dynamically reduce patience but not below 10
        dynamic_patience = max(5, dynamic_patience)  # Further ensure at least 5 epochs patience

        if best_loss_counter >= dynamic_patience:
            print(f"🛑 Early stopping at epoch {epoch+1} due to {dynamic_patience} non-improving epochs.")
            break

    # Restore the best model weights if early stopping was triggered
    if best_model_state:
        meta_model.load_state_dict(best_model_state)

    return meta_model, scaler_inputs, scaler_targets















def evaluate_maml(meta_model, data, input_columns, target_columns, curiosity, weights, max_or_min, acquisition="UCB"):
    """
    Evaluates all unlabeled samples, computes utility, and selects the best candidate for lab testing.
    """
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    
    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for evaluation.")
        return None

    scaler_inputs = StandardScaler().fit(data[input_columns])
    scaler_targets = StandardScaler().fit(data.dropna(subset=target_columns)[target_columns])

    inputs_infer = scaler_inputs.transform(unlabeled_data[input_columns])
    inputs_infer_tensor = torch.tensor(inputs_infer, dtype=torch.float32)

    with torch.no_grad():
        predictions_scaled = meta_model(inputs_infer_tensor).numpy()

    predictions = scaler_targets.inverse_transform(predictions_scaled)
    novelty_scores = calculate_novelty(inputs_infer, scaler_inputs.transform(data.dropna(subset=target_columns)[input_columns]))
    meta_model.train()

    uncertainty_scores = calculate_uncertainty(meta_model, inputs_infer_tensor, num_perturbations=500, dropout_rate=0.5)

    # ✅ Add Debug Statements to Monitor Distribution
    print("Predictions - Min:", np.min(predictions), "Max:", np.max(predictions), "Mean:", np.mean(predictions))
    print("Uncertainty - Min:", np.min(uncertainty_scores), "Max:", np.max(uncertainty_scores), "Mean:", np.mean(uncertainty_scores))

    # ✅ Ensure `max_or_min` is correctly passed and formatted
    if max_or_min is None or not isinstance(max_or_min, list):
        max_or_min = ['max'] * len(target_columns)  # Default to maximizing

    print("Debug - Predictions shape:", predictions.shape)
    print("Debug - Uncertainty shape:", uncertainty_scores.shape)
    print("Debug - Weights shape:", weights.shape if isinstance(weights, np.ndarray) else "Not an array")
    print("Debug - max_or_min:", max_or_min)

    weights = np.array(weights) + np.random.uniform(0.1, 0.5, size=weights.shape)


    

    # Update the utility calculation call:
    utility_scores = calculate_utility(
        predictions, 
        uncertainty_scores, 
        None, 
        curiosity, 
        weights, 
        max_or_min, 
        acquisition=acquisition  # Dynamically set acquisition function
    )

    utility_scores = np.log1p(utility_scores)

    # ✅ Test curiosity influence (for debugging with acquisition function)
    test_utility_exploit = calculate_utility(
        predictions, 
        uncertainty_scores, 
        None, 
        -2, 
        weights, 
        max_or_min, 
        acquisition=acquisition
    )
    
    test_utility_explore = calculate_utility(
        predictions, 
        uncertainty_scores, 
        None, 
        2, 
        weights, 
        max_or_min, 
        acquisition=acquisition
    )

    print(f"Acquisition Function: {acquisition}")
    print("Low curiosity (Exploitation) - Utility:", test_utility_exploit[:5])
    print("High curiosity (Exploration) - Utility:", test_utility_explore[:5])

        # Store the results in a DataFrame
    result_df = unlabeled_data.copy()

    # Apply log scaling and set precision to avoid displaying zeros
    result_df["Utility"] = np.log1p(utility_scores).flatten()
    result_df["Uncertainty"] = np.clip(uncertainty_scores, 1e-6, None).flatten()
    result_df["Novelty"] = novelty_scores.flatten()

    for i, col in enumerate(target_columns):
        result_df[col] = predictions[:, i]

    result_df = result_df.sort_values(by="Utility", ascending=False)
    result_df["Selected for Testing"] = False
    result_df.iloc[0, result_df.columns.get_loc("Selected for Testing")] = True  

    # ✅ Reset index to ensure sequential row numbering (use default Streamlit numbering)
    result_df.reset_index(drop=True, inplace=True)


    # 🆕 Reorder columns without losing existing dataset columns
    columns_to_front = ["Idx_Sample"]
    target_columns = [col for col in result_df.columns if col in target_columns]
    metrics_columns = ["Utility", "Novelty", "Uncertainty"]
    
    # Preserve all other columns
    remaining_columns = [col for col in result_df.columns if col not in columns_to_front + target_columns + metrics_columns]

    # Set the new column order
    new_column_order = columns_to_front + metrics_columns + target_columns + remaining_columns 
    result_df = result_df[new_column_order]
    # 🆕 Display the DataFrame in Streamlit with only the built-in index
    #st.dataframe(result_df, use_container_width=True)


    return result_df




