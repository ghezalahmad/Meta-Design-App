# MetaDesign Dashboard: Material Mix Optimization with MAML and Reptile Models

## 🚀 Project Description
The **MetaDesign Dashboard** is an AI-driven application designed for material mix optimization, leveraging meta-learning techniques such as **Model-Agnostic Meta-Learning (MAML)** and **Reptile** models. It facilitates sequential learning for material discovery, drawing inspiration from the **SLAMD (Sequential Learning App for Materials Discovery)** methodology. The primary goal is to accelerate the discovery of sustainable and high-performance materials, such as cementitious composites, while minimizing the need for extensive laboratory experiments.

## 🎯 Key Features
- **Streamlit Interface:** User-friendly interface for dataset uploading, model configuration, and visualization.
- **MAML & Reptile Integration:** Allows dynamic switching between MAML and Reptile models for meta-learning.
- **PINN Integration:** Includes a Physics-Informed Neural Network (PINN) model option to incorporate physical constraints into the learning process.
- **Digital Lab for Design Space Creation:** A powerful feature to generate a design space from scratch. Instead of uploading a dataset, you can define material components, their properties (e.g., cost, density), and constraints. The application then generates a comprehensive dataset of material formulations for the AI models to analyze.
- **Full Sequential Learning Workflow:** The application now fully supports a closed-loop, iterative experimental workflow. It suggests a candidate, you test it in the lab, log the results directly in the UI, and the model retrains with the new data to provide an even more informed suggestion for the next experiment.
- **Automated Hyperparameter Tuning:** Supports adaptive learning rates, batch sizes, and epoch settings.
- **Acquisition Function Selection:** Choose between Expected Improvement (EI), Upper Confidence Bound (UCB), and Probability of Improvement (PI).
- **Interactive Visualizations:** Plotly-based visualizations for exploration vs. exploitation and curiosity impact.
- **Advanced Data Handling:** Supports missing values, a priori information, and batch dataset updates.

## 📂 Project Structure
```
MAML-App/
├─ app/                   # Application-specific modules
│   ├─ models/            # MAML and Reptile models
│   ├─ utils/             # Utility functions (e.g., calculate_utility, calculate_novelty)
│   ├─ digital_lab.py     # UI and logic for the Digital Lab feature
│   ├─ visualization.py    # Visualization functions with Plotly
├─ data/                  # Example datasets
├─ main.py                # Main Streamlit application
├─ requirements.txt       # Python dependencies
└─ README.md              # Project documentation
```

## 🛠️ Setup Instructions
1. **Clone the Repository:**
```bash
git clone https://github.com/ghezalahmad/MAML
cd MAML
```

2. **Create and Activate a Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Application:**
```bash
streamlit run main.py
```

5. **Access the Dashboard:**
- Open your browser at: `http://localhost:8501`

## 🧠 Usage: Running an Experimental Campaign
The dashboard is designed to guide a full experimental campaign. Follow these steps for an iterative workflow:

1. **Start Your Campaign:**
   - **Option A: Upload Existing Data:** Start with a CSV file containing your initial experiments. Some rows should have measured target values, while others can be candidates you want to evaluate (with empty target values).
   - **Option B: Create a Design Space:** Use the **Digital Lab** to define your material components and generate a large set of candidate formulations.

2. **Configure and Run:**
   - Select your input features and the target properties you want to optimize.
   - Choose a model (**MAML**, **Reptile**, or **PINN**) and configure its parameters.
   - Click **"Run Experiment"**. The model will train on your existing data and rank the unlabeled candidates.

3. **Synthesize and Test:**
   - The model will suggest a top-ranked candidate, but you are free to test **any** sample from the "Editable Dataset" table. Note the "Sample Index" of the formulation you choose to test in your lab.

4. **Log Your Results:**
   - Once you have the results from your physical experiment, return to the dashboard.
   - In the **"📝 Log Lab Experiment Results"** section, enter the **Sample Index** of the sample you tested.
   - Fill in the measured values for each of the target properties.
   - Click **"Add Result to Dataset"**. The dataset will be instantly updated with this new, valuable information, and the table will refresh.

5. **Iterate:**
   - The main button will now read **"Suggest Next Experiment"**.
   - Click it to re-run the analysis. The model will now learn from the result you just added and propose a new best candidate.
   - Repeat steps 3-5 to continuously refine your material formulations.

## 📊 Visualization
The application offers dynamic visualizations to illustrate:
- **Exploration vs. Exploitation:** Utility vs. Novelty with Uncertainty as color coding.
- **Curiosity Impact:** Shows how different curiosity settings affect candidate selection.
- **Distributions:** Histograms of utility, novelty, and uncertainty scores.

## 🧪 Testing
This project uses `pytest` for unit and integration testing.

1.  **Install Testing Dependencies:**
    Make sure you have installed the full set of dependencies, including the testing tools:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run All Tests:**
    To run the entire test suite, navigate to the project's root directory and run:
    ```bash
    python -m pytest
    ```
    This will automatically discover and run all tests in the `tests/` directory.

## 👥 Contributors
- **Ghezal Ahmad** - Project Lead
- **Collaborators:** Contributions are welcome! Please submit a pull request or open an issue for feedback.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



