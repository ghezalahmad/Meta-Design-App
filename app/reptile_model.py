import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app.utils import initialize_scheduler, calculate_utility, calculate_novelty, calculate_uncertainty
import math
import streamlit as st
# Import visualization function
from app.visualization import visualize_exploration_exploitation


class ReptileModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(ReptileModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softplus()
        )


    def forward(self, x):
        return self.network(x)

def reptile_train(model, data, input_columns, target_columns, epochs, learning_rate, num_tasks, scaler_x=None, scaler_y=None, early_stopping=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = torch.nn.SmoothL1Loss()
    
    if scaler_x is None:
        scaler_x = StandardScaler()
    if scaler_y is None:
        scaler_y = StandardScaler()
    
    smoothed_loss = float('inf')
    patience = 10  
    no_improve_counter = 0
    loss_threshold = 0.001
    
    # ✅ Get Labeled & Unlabeled Data
    labeled_data = data.dropna(subset=target_columns).sort_index()
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]

    if len(labeled_data) < 4:  # Ensure a minimum batch size of 4
        st.warning("❌ Not enough labeled samples to continue training.")
        return model, scaler_x, scaler_y

    # ✅ Standardize Inputs and Targets
    inputs = scaler_x.fit_transform(labeled_data[input_columns].values)
    targets = scaler_y.fit_transform(labeled_data[target_columns].values)
    
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    # ✅ Now compute the batch size using inputs length
    batch_size = min(8, max(2, len(inputs) // (2 * num_tasks)))
    inner_loop_steps = min(32, len(inputs) * 4)  # Inner loop steps similar to MAML
    
    scheduler = initialize_scheduler(optimizer, scheduler_type="CosineAnnealing", T_max=epochs // 2, eta_min=1e-5)
    
    # ✅ Train the Model with `num_tasks` and `inner_loop_steps`
    for epoch in range(epochs):
        epoch_loss = 0.0
        for task in range(num_tasks):
            indices = torch.randperm(len(inputs))[:batch_size]
            batch_inputs = inputs[indices]
            batch_targets = targets[indices]
            
            for _ in range(inner_loop_steps):  # Inner loop steps
                optimizer.zero_grad()
                predictions = model(batch_inputs)
                loss = loss_function(predictions, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
        
        smoothed_loss = 0.95 * smoothed_loss + 0.05 * (epoch_loss / num_tasks) if smoothed_loss != float('inf') else (epoch_loss / num_tasks)
        print(f"✅ Epoch {epoch}/{epochs}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}, Batch Size: {batch_size}, Num Tasks: {num_tasks}, Smoothed Loss: {smoothed_loss:.4f}")
        
        scheduler.step()
        
        if early_stopping:
            if abs(smoothed_loss - loss.item()) < loss_threshold:
                no_improve_counter += 1
            else:
                no_improve_counter = 0
            
            if no_improve_counter >= patience:
                print("🛑 Early stopping triggered: No significant loss improvement.")
                break

    return model, scaler_x, scaler_y










def evaluate_reptile(model, data, input_columns, target_columns, curiosity, weights, max_or_min, acquisition="UCB"):
    """
    Evaluates all unlabeled samples, computes utility, and selects the best candidate for lab testing.
    Includes debugging outputs for utility, uncertainty, novelty, and curiosity influence.
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
        predictions_scaled = model(inputs_infer_tensor).numpy()

    predictions = scaler_targets.inverse_transform(predictions_scaled)
    novelty_scores = calculate_novelty(inputs_infer, scaler_inputs.transform(data.dropna(subset=target_columns)[input_columns]))
    model.train()

    uncertainty_scores = calculate_uncertainty(model, inputs_infer_tensor, num_perturbations=500, dropout_rate=0.5)

    # ✅ Debugging Outputs
    print(f"\n🧠 Acquisition Function: {acquisition}")
    print("🔮 Predictions - Min:", np.min(predictions), "Max:", np.max(predictions), "Mean:", np.mean(predictions))
    print("❓ Uncertainty - Min:", np.min(uncertainty_scores), "Max:", np.max(uncertainty_scores), "Mean:", np.mean(uncertainty_scores))
    print("🌟 Novelty - Min:", np.min(novelty_scores), "Max:", np.max(novelty_scores), "Mean:", np.mean(novelty_scores))
    print("🧠 Curiosity:", curiosity)

    # Ensure max_or_min is formatted correctly
    if max_or_min is None or not isinstance(max_or_min, list):
        max_or_min = ['max'] * len(target_columns)

    # Compute Utility Scores
    utility_scores = calculate_utility(
        predictions,
        uncertainty_scores,
        None,
        curiosity,
        weights,
        max_or_min,
        acquisition=acquisition
    )

    utility_scores = np.log1p(utility_scores)

    # ✅ Test curiosity influence (for debugging with acquisition function)
    test_utility_exploit = calculate_utility(
        predictions, 
        uncertainty_scores, 
        None, 
        -2,  # Max exploitation
        weights, 
        max_or_min, 
        acquisition=acquisition
    )
    
    test_utility_explore = calculate_utility(
        predictions, 
        uncertainty_scores, 
        None, 
        2,  # Max exploration
        weights, 
        max_or_min, 
        acquisition=acquisition
    )

    print("\n🔍 **Utility Scores with Curiosity Variants:**")
    print("➡️ Default Curiosity (", curiosity, ") - Utility:", utility_scores[:5])
    print("📉 Max Exploitation (Curiosity = -2) - Utility:", test_utility_exploit[:5])
    print("📈 Max Exploration (Curiosity = +2) - Utility:", test_utility_explore[:5])


    result_df = unlabeled_data.copy()

    # Add calculated columns for utility, uncertainty, and novelty
    result_df["Utility"] = utility_scores.flatten()
    result_df["Uncertainty"] = np.clip(uncertainty_scores, 1e-6, None).flatten()
    result_df["Novelty"] = novelty_scores.flatten()

    # Add predicted values for the target columns
    for i, col in enumerate(target_columns):
        result_df[col] = predictions[:, i]

    # Sort by utility and mark the top candidate for testing
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


