import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from scipy.stats import norm
import streamlit as st
import pandas as pd
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from app.utils import calculate_utility, calculate_novelty, compute_acquisition_utility, select_acquisition_function

class MAMLModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3, dropout_rate=0.3):
        super(MAMLModel, self).__init__()
        
        # Improved architecture with residual connections and batch normalization
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Middle layers with residual connections
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights with He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        if x.shape[0] > 1:  # Only apply batch norm for batch size > 1
            x = self.input_bn(x)
        x = torch.relu(x)
        x = self.input_dropout(x)
        
        # Middle layers with residual connections
        for i in range(len(self.layers)):
            identity = x
            x = self.layers[i](x)
            if x.shape[0] > 1:
                x = self.bn_layers[i](x)
            x = torch.relu(x)
            x = self.dropout_layers[i](x)
            
            # Add residual connection if shapes match
            if identity.shape == x.shape:
                x = x + identity
        
        # Output layer
        output = self.output_layer(x)
        return torch.nn.functional.softplus(output)


def meta_train(meta_model, data, input_columns, target_columns, epochs=100, inner_lr=0.01, 
               outer_lr=0.001, num_tasks=4, inner_lr_decay=0.95, curiosity=0, 
               min_samples_per_task=3, early_stopping_patience=10):

    # Configure optimizers with weight decay to prevent overfitting on small datasets
    optimizer = optim.AdamW(meta_model.parameters(), lr=outer_lr, 
                           weight_decay=1e-3, betas=(0.9, 0.999))
    
    # Cosine annealing scheduler with warm restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=outer_lr * 0.1
    )
    
    # Adaptive loss function based on dataset size
    # Extract labeled data
    labeled_data = data.dropna(subset=target_columns).sample(frac=1).reset_index(drop=True)

    # ✅ Ensure target columns are numeric
    for col in target_columns:
        labeled_data[col] = pd.to_numeric(labeled_data[col], errors='coerce')

    # ✅ Drop rows with NaN values (if conversion failed for some rows)
    labeled_data = labeled_data.dropna(subset=target_columns)
    
    # Check if we have enough labeled data
    if labeled_data.empty:
        st.error("⚠️ No labeled data available for training.")
        return meta_model, None, None
    
    # ✅✅ NEW - Enhanced handling for extremely small datasets
    MINIMUM_DATA_POINTS = 8
    if len(labeled_data) > 0 and len(labeled_data) < MINIMUM_DATA_POINTS and not use_fallback:
        st.warning(f"🔄 Few labeled samples ({len(labeled_data)}). Applying data tiling for more robust evaluation.")
        tile_factor = int(np.ceil(MINIMUM_DATA_POINTS / len(labeled_data)))
        labeled_data_eval = pd.concat([labeled_data] * tile_factor, ignore_index=True)
        
        for col in input_columns:
            std = labeled_data[col].std()
            if std > 0:
                noise = np.random.normal(0, std * 0.05, size=len(labeled_data_eval))
                labeled_data_eval[col] += noise
        for col in target_columns:
            std = labeled_data[col].std()
            if std > 0:
                noise = np.random.normal(0, max(std * 0.02, 1e-6), size=len(labeled_data_eval))
                labeled_data_eval[col] += noise

            
        st.info(f"✅ Data tiled from {len(labeled_data)} to {len(labeled_data_eval)} samples.")
        labeled_data = labeled_data_eval  # 🔄 Apply tiling + noise

      
    # ✅ Now, safely compute min and max
    print(f"Target range: {labeled_data[target_columns].min().values} to {labeled_data[target_columns].max().values}")
    if len(labeled_data) < 10:
        # For very small datasets, use MSE which is more stable
        loss_function = nn.MSELoss()
        st.info("Using MSE loss for small dataset stability")
    else:
        # For larger datasets, use Huber loss for robustness to outliers
        loss_function = nn.SmoothL1Loss()
        st.info("Using Huber loss for robustness to outliers")

    # Validate if we have enough samples for few-shot learning
    if len(labeled_data) < min_samples_per_task * 2:
        st.warning(f"⚠️ Limited labeled samples ({len(labeled_data)}) - using few-shot adaptations.")
        # Reduce task complexity for very small datasets
        num_tasks = max(2, min(num_tasks, len(labeled_data) // 2))
    
    # Use RobustScaler for better handling of outliers in small datasets
    scaler_inputs = RobustScaler().fit(labeled_data[input_columns])
    scaler_targets = RobustScaler().fit(labeled_data[target_columns])
    
    # Transform the data
    inputs = torch.tensor(scaler_inputs.transform(labeled_data[input_columns]), dtype=torch.float32)
    targets = torch.tensor(scaler_targets.transform(labeled_data[target_columns]), dtype=torch.float32)
    
    # Ensure we have data to work with
    if len(inputs) == 0 or len(targets) == 0:
        st.error("❌ No valid data available for training after preprocessing.")
        return meta_model, scaler_inputs, scaler_targets
    
    # Dynamic batch size based on available data
    batch_size = max(2, min(8, len(inputs) // 4))
    
    # Adjust number of tasks based on available data
    num_tasks = max(2, min(num_tasks, len(inputs) // batch_size))
    
    # Cross-validation strategy for small datasets
    kf = KFold(n_splits=min(5, len(labeled_data)), shuffle=True)
    
    # Initialize tracking variables for early stopping
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Set minimum learning rate for inner loop
    min_inner_lr = inner_lr * 0.1
    current_inner_lr = inner_lr
    
    # Create a dictionary to store validation info for each fold
    fold_metrics = {}
    
    # Create episodic memory for knowledge retention
    episodic_memory = []
    episodic_memory_size = min(10, len(inputs))
    
    st.info(f"Starting meta-training with {len(labeled_data)} labeled samples")
    st.info(f"Batch size: {batch_size}, Number of tasks: {num_tasks}")
    
    # Progress bar for training
    progress_bar = st.progress(0)
    
    # Main training loop
    for epoch in range(epochs):
        meta_model.train()
        meta_losses = []
        
        # Update progress bar
        progress_bar.progress((epoch + 1) / epochs)
        
        # Decay inner learning rate with a smoother curve for material properties
        current_inner_lr = max(inner_lr * (inner_lr_decay ** (epoch / 10)), min_inner_lr)
        
        # Dynamically adjust inner loop steps based on epoch and dataset size
        inner_loop_steps = max(2, min(5, int(len(inputs) / batch_size / 2)))
        if epoch > epochs // 2:
            inner_loop_steps = max(inner_loop_steps, 3)  # Increase steps in later epochs
        
        # Use K-fold for task creation in small datasets
        task_indices = []
        for train_idx, val_idx in kf.split(inputs):
            if len(train_idx) >= min_samples_per_task and len(val_idx) >= 2:
                task_indices.append((train_idx, val_idx))
        
        # If K-fold doesn't provide enough tasks, create random ones
        while len(task_indices) < num_tasks:
            # For very small datasets, use leave-one-out approach
            if len(inputs) <= 10:
                train_size = max(2, len(inputs) - 2)
                train_idx = np.random.choice(len(inputs), size=train_size, replace=False)
                val_idx = np.setdiff1d(np.arange(len(inputs)), train_idx)
            else:
                # Random split with ~70% train, ~30% validation
                split_idx = int(0.7 * len(inputs))
                indices = torch.randperm(len(inputs))
                train_idx = indices[:split_idx]
                val_idx = indices[split_idx:]
            
            task_indices.append((train_idx, val_idx))
        
        # Process each task
        for task_idx, (support_indices, query_indices) in enumerate(task_indices[:num_tasks]):
            # Convert to tensors
            support_indices = torch.tensor(support_indices)
            query_indices = torch.tensor(query_indices)
            
            # Get support and query data
            support_inputs = inputs[support_indices]
            support_targets = targets[support_indices]
            query_inputs = inputs[query_indices]
            query_targets = targets[query_indices]
            
            # Add samples from episodic memory for knowledge retention
            if epoch > 0 and len(episodic_memory) > 0:
                memory_samples = min(len(episodic_memory), batch_size // 2)
                memory_indices = np.random.choice(len(episodic_memory), size=memory_samples, replace=False)
                
                for idx in memory_indices:
                    mem_input, mem_target = episodic_memory[idx]
                    support_inputs = torch.cat([support_inputs, mem_input.unsqueeze(0)], dim=0)
                    support_targets = torch.cat([support_targets, mem_target.unsqueeze(0)], dim=0)
            
            # Create a copy of the meta-model for this task
            task_model = copy.deepcopy(meta_model)
            
            # Adaptive optimizer selection based on sample size
            if len(support_inputs) < 10:
                task_optimizer = optim.SGD(task_model.parameters(), 
                                          lr=current_inner_lr, 
                                          momentum=0.9, 
                                          weight_decay=1e-4)
            else:
                task_optimizer = optim.Adam(task_model.parameters(), 
                                           lr=current_inner_lr, 
                                           weight_decay=1e-4)
            
            # Inner loop adaptation
            task_losses = []
            for step in range(inner_loop_steps):
                # For very small datasets, use all support data
                if len(support_inputs) <= batch_size:
                    batch_inputs = support_inputs
                    batch_targets = support_targets
                else:
                    # Sample a batch with importance weighting for difficult samples
                    if step > 0 and len(task_losses) > 0:
                        # Focus on difficult samples in later steps
                        with torch.no_grad():
                            difficulties = torch.abs(task_model(support_inputs) - support_targets).sum(dim=1)
                            probs = difficulties / difficulties.sum()
                            batch_indices = torch.multinomial(probs, min(batch_size, len(probs)), replacement=False)
                    else:
                        # Random sampling in first step
                        batch_indices = torch.randperm(len(support_inputs))[:batch_size]
                    
                    batch_inputs = support_inputs[batch_indices]
                    batch_targets = support_targets[batch_indices]
                
                # Forward pass
                task_predictions = task_model(batch_inputs)
                task_loss = loss_function(task_predictions, batch_targets)
                
                # Apply curiosity factor
                if curiosity != 0:
                    # Refined curiosity scaling for material discovery domain
                    scaling_factor = 1.0 + (0.1 * curiosity * (1.0 - (step / inner_loop_steps)))
                    task_loss = task_loss * scaling_factor
                
                # Backward pass
                task_optimizer.zero_grad()
                task_loss.backward()
                
                # Gradient clipping to stabilize training with small batches
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=0.5)
                task_optimizer.step()
                
                task_losses.append(task_loss.item())
            
            # Evaluate on query set
            task_model.eval()
            query_predictions = task_model(query_inputs)
            query_loss = loss_function(query_predictions, query_targets)
            
            # Apply curiosity to query loss as well for consistent optimization
            if curiosity != 0:
                query_loss = query_loss * (1 + 0.1 * curiosity)
            
            # Store good examples in episodic memory
            with torch.no_grad():
                for i in range(min(2, len(query_inputs))):
                    pred = task_model(query_inputs[i:i+1])
                    error = torch.abs(pred - query_targets[i:i+1]).mean().item()
                    
                    # Store examples with low error
                    if error < 0.2:
                        if len(episodic_memory) < episodic_memory_size:
                            episodic_memory.append((query_inputs[i], query_targets[i]))
                        else:
                            # Replace a random item
                            replace_idx = np.random.randint(0, len(episodic_memory))
                            episodic_memory[replace_idx] = (query_inputs[i], query_targets[i])
            
            # Meta-update (first-order approximation for efficiency)
            if task_idx == 0:
                meta_loss = query_loss
            else:
                meta_loss = meta_loss + query_loss
            
            # Store metrics
            fold_metrics[f"task_{task_idx}"] = {
                "support_size": len(support_inputs),
                "query_size": len(query_inputs),
                "final_task_loss": task_losses[-1] if task_losses else float('inf'),
                "query_loss": query_loss.item()
            }
            
            meta_losses.append(query_loss.item())
        
        # Average meta loss
        meta_loss = meta_loss / len(task_indices[:num_tasks])
        
        # Update meta-model
        optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping for meta-update
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Calculate average loss for this epoch
        avg_meta_loss = np.mean(meta_losses) if meta_losses else float('inf')
        
        # Early stopping check
        if avg_meta_loss < best_loss:
            best_loss = avg_meta_loss
            best_model_state = copy.deepcopy(meta_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            st.info(f"Epoch {epoch+1}/{epochs} - Meta Loss: {avg_meta_loss:.4f}, Best Loss: {best_loss:.4f}")
            st.info(f"Inner LR: {current_inner_lr:.6f}, Outer LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Dynamic patience based on dataset size and epoch
        if len(labeled_data) < 20:
            dynamic_patience = max(5, early_stopping_patience - (epoch // 10))
        else:
            dynamic_patience = early_stopping_patience
        
        # Check for early stopping
        if patience_counter >= dynamic_patience:
            st.warning(f"🛑 Early stopping at epoch {epoch+1} due to {dynamic_patience} non-improving epochs.")
            break
    
    # Restore best model weights
    if best_model_state:
        meta_model.load_state_dict(best_model_state)
        st.success(f"✅ Restored best model with loss: {best_loss:.4f}")
    
    # Final evaluation on all labeled data
    meta_model.eval()
    with torch.no_grad():
        all_predictions = meta_model(inputs)
        final_loss = loss_function(all_predictions, targets).item()
    
    st.success(f"Final evaluation loss on all labeled data: {final_loss:.4f}")
    
    return meta_model, scaler_inputs, scaler_targets


# Corrected data tiling and scaling logic in evaluate_maml

def evaluate_maml(meta_model, data, input_columns, target_columns, curiosity, weights, 
                  max_or_min, acquisition=None, min_labeled_samples=5, dynamic_acquisition=True):

    # Get labeled and unlabeled data
    unlabeled_data = data[data[target_columns].isna().any(axis=1)]
    labeled_data = data.dropna(subset=target_columns)

    if unlabeled_data.empty:
        st.warning("No unlabeled samples available for evaluation.")
        return None

    st.info(f"Evaluating with {len(labeled_data)} labeled samples and {len(unlabeled_data)} unlabeled samples")

    # Set fallback strategy based on labeled data count
    use_fallback = len(labeled_data) < min_labeled_samples

    # Enhanced handling for very small datasets (tiling + noise)
    MINIMUM_DATA_POINTS = 8
    if len(labeled_data) > 0 and len(labeled_data) < MINIMUM_DATA_POINTS and not use_fallback:
        st.warning(f"🔄 Few labeled samples ({len(labeled_data)}). Applying data tiling for robust evaluation.")
        tile_factor = int(np.ceil(MINIMUM_DATA_POINTS / len(labeled_data)))
        labeled_data_eval = pd.concat([labeled_data] * tile_factor, ignore_index=True)

        # Add noise only if std > 0
        for col in input_columns:
            std = labeled_data[col].std()
            if std > 0:
                noise = np.random.normal(0, std * 0.05, size=len(labeled_data_eval))
                labeled_data_eval[col] += noise

        for col in target_columns:
            std = labeled_data[col].std()
            if std > 0:
                noise = np.random.normal(0, max(std * 0.02, 1e-6), size=len(labeled_data_eval))
                labeled_data_eval[col] += noise

        st.info(f"✅ Data tiled from {len(labeled_data)} to {len(labeled_data_eval)} samples.")
    else:
        labeled_data_eval = labeled_data

    # Prediction using fallback GP or meta-model
    if use_fallback:
        st.warning(f"Using Gaussian Process fallback strategy due to limited data ({len(labeled_data)} samples)")
        result_df = unlabeled_data.copy()
        all_predictions = np.zeros((len(unlabeled_data), len(target_columns)))
        all_uncertainties = np.zeros((len(unlabeled_data), len(target_columns)))

        for i, target_col in enumerate(target_columns):
            kernel = Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                          n_restarts_optimizer=5, random_state=42)
            X_train = labeled_data[input_columns].values
            y_train = labeled_data[target_col].values.reshape(-1, 1)
            gp.fit(X_train, y_train)
            X_test = unlabeled_data[input_columns].values
            y_pred, y_std = gp.predict(X_test, return_std=True)

            # Ensure GP predictions are non-negative, consistent with NN model outputs
            y_pred = np.maximum(y_pred, 0)

            all_predictions[:, i] = y_pred
            all_uncertainties[:, i] = y_std
            result_df[target_col] = y_pred

        uncertainty_scores = np.mean(all_uncertainties, axis=1).reshape(-1, 1)
    else:
        st.info("Using meta-trained model for predictions")
        scaler_inputs = RobustScaler().fit(labeled_data[input_columns])
        scaler_targets = RobustScaler().fit(labeled_data[target_columns])
        inputs_infer = scaler_inputs.transform(unlabeled_data[input_columns])
        inputs_infer_tensor = torch.tensor(inputs_infer, dtype=torch.float32)

        meta_model.eval()
        with torch.no_grad():
            predictions_scaled = meta_model(inputs_infer_tensor).numpy()

        uncertainty_scores = calculate_uncertainty_ensemble(meta_model, inputs_infer_tensor,
                                                            num_samples=30, dropout_rate=0.3)

        predictions = scaler_targets.inverse_transform(predictions_scaled)
        # Ensure predictions are non-negative, aligning with expected output characteristics.
        predictions = np.maximum(predictions, 0)

        result_df = unlabeled_data.copy()
        for i, col in enumerate(target_columns):
            result_df[col] = predictions[:, i]

    # Novelty calculation
    X_labeled = labeled_data[input_columns].values
    novelty_scores = calculate_novelty(inputs_infer, RobustScaler().fit_transform(X_labeled))

    if acquisition is None:
        acquisition = select_acquisition_function(curiosity, len(labeled_data))
    st.info(f"Using SLAMD-style acquisition function: {acquisition}")

    weights = np.clip(np.array(weights) + np.random.uniform(0.01, 0.1, size=len(weights)), 0.1, 1.0)
    if max_or_min is None or not isinstance(max_or_min, list):
        max_or_min = ['max'] * len(target_columns)

    utility_scores = compute_acquisition_utility(predictions if not use_fallback else all_predictions,
                                                 uncertainty_scores, novelty_scores, curiosity,
                                                 weights, max_or_min, acquisition)
    utility_scores = np.log1p(utility_scores)

    result_df["Utility"] = utility_scores.flatten()
    result_df["Uncertainty"] = np.clip(uncertainty_scores, 1e-6, None).flatten()
    result_df["Novelty"] = novelty_scores.flatten()
    result_df["Exploration"] = result_df["Uncertainty"] * result_df["Novelty"]
    result_df["Exploitation"] = 1.0 - result_df["Uncertainty"]

    result_df = result_df.sort_values(by="Utility", ascending=False)
    result_df["Selected for Testing"] = False
    result_df.iloc[0, result_df.columns.get_loc("Selected for Testing")] = True
    result_df.reset_index(drop=True, inplace=True)

    columns_to_front = ["Idx_Sample"] if "Idx_Sample" in result_df.columns else []
    metrics_columns = ["Utility", "Exploration", "Exploitation", "Novelty", "Uncertainty"]
    remaining_columns = [col for col in result_df.columns if col not in columns_to_front + metrics_columns + target_columns + ["Selected for Testing"]]
    new_column_order = columns_to_front + metrics_columns + target_columns + remaining_columns + ["Selected for Testing"]
    new_column_order = [col for col in new_column_order if col in result_df.columns]

    result_df = result_df[new_column_order]
    return result_df


def calculate_uncertainty_ensemble(model, X, num_samples=30, dropout_rate=0.3):
    """
    Enhanced uncertainty estimation combining Monte Carlo dropout with model ensembling.
    
    Parameters:
    -----------
    model : nn.Module
        The model with dropout layers
    X : torch.Tensor
        Input data
    num_samples : int
        Number of Monte Carlo samples
    dropout_rate : float
        Dropout rate to use (if model has configurable dropout)
        
    Returns:
    --------
    uncertainty : numpy.ndarray
        Estimated uncertainty for each prediction
    """
    # Set model to training mode to enable dropout
    model.train()
    
    # For models with configurable dropout, try to update the rate
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
    
    # Collect predictions from multiple forward passes
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            pred = model(X)
            predictions.append(pred)
    
    # Stack predictions and compute statistics
    predictions_stack = torch.stack(predictions)
    mean_pred = torch.mean(predictions_stack, dim=0)
    var_pred = torch.var(predictions_stack, dim=0)
    
    # Combine epistemic (model) and aleatoric (data) uncertainty
    total_uncertainty = var_pred + torch.abs(mean_pred) * 0.05
    
    # Convert to numpy array and ensure proper shape for single uncertainty per sample
    uncertainty_np = total_uncertainty.numpy()
    
    # If we have multi-target predictions, average the uncertainties across targets
    if uncertainty_np.ndim > 1 and uncertainty_np.shape[1] > 1:
        uncertainty_np = np.mean(uncertainty_np, axis=1)
    
    return uncertainty_np


def calculate_utility_with_acquisition(predictions, uncertainties, novelty, curiosity, weights, max_or_min, acquisition="UCB"):
    """
    Utility calculation for materials discovery with acquisition functions.
    """
    # Ensure input types and shapes
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    novelty = np.array(novelty).flatten()
    weights = np.array(weights).reshape(1, -1)
    
    # Handle dimension mismatch - make sure uncertainties is the right shape
    if uncertainties.ndim > 1 and uncertainties.shape[0] == predictions.shape[0]:
        # If we have per-target uncertainties, average them
        uncertainties = np.mean(uncertainties, axis=1)
    
    # Flatten uncertainties to ensure consistent shape
    uncertainties = uncertainties.flatten()
    
    # Make sure the length matches predictions
    if len(uncertainties) != len(predictions):
        # If shapes don't match, broadcast or truncate
        if len(uncertainties) > len(predictions):
            # Too many uncertainties, truncate
            uncertainties = uncertainties[:len(predictions)]
        else:
            # Too few uncertainties, broadcast by repeating
            repeats = int(np.ceil(len(predictions) / len(uncertainties)))
            uncertainties = np.tile(uncertainties, repeats)[:len(predictions)]
    
    # Define minimum acceptable threshold.
    # Predictions below this threshold will have their utility scores heavily penalized.
    min_strength_threshold = 10  
    
    # Create a mask for rows that have any value below threshold, based on original predictions.
    invalid_rows_for_penalty = np.any(predictions < min_strength_threshold, axis=1)
    
    # Predictions used for the positive part of utility calculation remain as passed
    # (they are already >= 0 from the model evaluation stage).
    # Utility is calculated based on the model's actual (non-negative) predictions.
    # The penalty for being < min_strength_threshold is applied later to the final utility score.
    
    # Normalize predictions to [0, 1] range
    min_vals = np.min(predictions, axis=0, keepdims=True)
    max_vals = np.max(predictions, axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals < 1e-10, 1.0, range_vals)  # Avoid division by zero
    
    norm_predictions = (predictions - min_vals) / range_vals
    
    # Apply max/min direction to normalize predictions
    for i, direction in enumerate(max_or_min):
        if direction == "min":
            norm_predictions[:, i] = 1.0 - norm_predictions[:, i]
    
    # Apply weights to the normalized predictions
    weighted_predictions = norm_predictions * weights
    
    # Normalize uncertainties to [0, 1] range
    norm_uncertainties = np.zeros_like(uncertainties) if uncertainties.size == 0 else uncertainties / (np.max(uncertainties) + 1e-10)
    
    # Normalize novelty to [0, 1] range
    norm_novelty = np.zeros_like(novelty) if novelty.size == 0 else novelty / (np.max(novelty) + 1e-10)
    
    # Exploration-exploitation trade-off scaling
    curiosity_factor = np.clip(1.0 + 0.5 * curiosity, 0.1, 2.0)
    
    # Compute utility based on acquisition function
    if acquisition == "UCB":
        utility = weighted_predictions.sum(axis=1, keepdims=True) + curiosity_factor * norm_uncertainties.reshape(-1, 1)
        # Apply penalty to invalid rows (if any)
    if np.any(invalid_rows_for_penalty):
        utility[invalid_rows_for_penalty] = -np.inf

    elif acquisition == "EI":
        best_pred = np.max(weighted_predictions.sum(axis=1))
        improvement = np.maximum(0, weighted_predictions.sum(axis=1, keepdims=True) - best_pred)
        z = improvement / (norm_uncertainties.reshape(-1, 1) + 1e-10)
        cdf = 0.5 * (1 + np.tanh(z / np.sqrt(2)))
        pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        utility = improvement * cdf + norm_uncertainties.reshape(-1, 1) * pdf * curiosity_factor
        if np.any(invalid_rows_for_penalty):
            utility[invalid_rows_for_penalty] = -np.inf

    elif acquisition == "PI":
        best_pred = np.max(weighted_predictions.sum(axis=1))
        z = (weighted_predictions.sum(axis=1, keepdims=True) - best_pred) / (norm_uncertainties.reshape(-1, 1) + 1e-10)
        utility = 0.5 * (1 + np.tanh(z / np.sqrt(2)))
        if np.any(invalid_rows_for_penalty):
            utility[invalid_rows_for_penalty] = -np.inf

    elif acquisition == "MaxEntropy":
        utility = norm_uncertainties.reshape(-1, 1) + 0.5 * norm_novelty.reshape(-1, 1) * curiosity_factor
        if np.any(invalid_rows_for_penalty):
            utility[invalid_rows_for_penalty] = -np.inf

    else: # Default or UCB
        utility = weighted_predictions.sum(axis=1, keepdims=True) + curiosity_factor * norm_uncertainties.reshape(-1, 1)
        if np.any(invalid_rows_for_penalty):
            utility[invalid_rows_for_penalty] = -np.inf
    
    # Sort based on utility (descending)
    sorted_indices = np.argsort(-utility.flatten())  # Sort in descending order
    utility = utility[sorted_indices]
    
    # Apply logarithmic scaling to smooth the utility landscape
    return np.clip(utility, 1e-10, None)