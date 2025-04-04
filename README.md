# MetaDesign Dashboard: Material Mix Optimization with MAML and Reptile Models

## 🚀 Project Description
The **MetaDesign Dashboard** is an AI-driven application designed for material mix optimization, leveraging meta-learning techniques such as **Model-Agnostic Meta-Learning (MAML)** and **Reptile** models. It facilitates sequential learning for material discovery, drawing inspiration from the **SLAMD (Sequential Learning App for Materials Discovery)** methodology. The primary goal is to accelerate the discovery of sustainable and high-performance materials, such as cementitious composites, while minimizing the need for extensive laboratory experiments.

## 🎯 Key Features
- **Streamlit Interface:** User-friendly interface for dataset uploading, model configuration, and visualization.
- **MAML & Reptile Integration:** Allows dynamic switching between MAML and Reptile models for meta-learning.
- **Automated Hyperparameter Tuning:** Supports adaptive learning rates, batch sizes, and epoch settings.
- **Acquisition Function Selection:** Choose between Expected Improvement (EI), Upper Confidence Bound (UCB), and Probability of Improvement (PI).
- **Sequential Learning Strategy:** Iteratively suggest the best candidates for lab testing based on utility, novelty, and uncertainty.
- **Interactive Visualizations:** Plotly-based visualizations for exploration vs. exploitation and curiosity impact.
- **Advanced Data Handling:** Supports missing values, a priori information, and batch dataset updates.

## 📂 Project Structure
```
MAML-App/
├─ app/                   # Application-specific modules
│   ├─ models/            # MAML and Reptile models
│   ├─ utils/             # Utility functions (e.g., calculate_utility, calculate_novelty)
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

## 🧠 Usage
1. **Upload Dataset:** Upload your material dataset in CSV format.
2. **Model Selection:** Choose between **MAML** and **Reptile** models.
3. **Configure Parameters:** Adjust hyperparameters, batch size, epochs, and acquisition function.
4. **Run Meta-Learning:** Start the training process and evaluate the suggested material candidates.
5. **Visualize Results:** View interactive plots showing the exploration vs. exploitation balance and utility scores.
6. **Select Candidates for Testing:** Use the utility ranking to guide lab experiments.

## 📊 Visualization
The application offers dynamic visualizations to illustrate:
- **Exploration vs. Exploitation:** Utility vs. Novelty with Uncertainty as color coding.
- **Curiosity Impact:** Shows how different curiosity settings affect candidate selection.
- **Distributions:** Histograms of utility, novelty, and uncertainty scores.

## 👥 Contributors
- **Ghezal Ahmad** - Project Lead
- **Collaborators:** Contributions are welcome! Please submit a pull request or open an issue for feedback.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



