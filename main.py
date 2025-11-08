
import streamlit as st

st.set_page_config(
    page_title="Meta-Design Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Hero Section ---
st.title("Welcome to the Meta-Design Dashboard 🔬")
st.markdown("""
    **Accelerate your materials discovery with the power of AI.**

    This dashboard provides a comprehensive suite of tools to move from initial data to optimized material formulations.
    Leverage state-of-the-art meta-learning models, physics-informed neural networks, and advanced Bayesian optimization
    to discover novel materials with unprecedented efficiency.
""")

st.markdown("---")

# --- Key Features Section ---
st.header("Key Features")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 Advanced AI Models")
    st.markdown("""
    - **Meta-Learning:** Utilize MAML and Reptile to adapt quickly from limited experimental data.
    - **Physics-Informed AI:** Incorporate physical laws into your models with PINNs for more realistic predictions.
    - **Bayesian Optimization:** Intelligently search the design space to find optimal solutions faster.
    """)

with col2:
    st.subheader("📈 Streamlined Workflow")
    st.markdown("""
    - **Digital Lab:** Automatically generate vast design spaces and material formulations.
    - **Interactive Analysis:** Visualize results, explore trade-offs, and gain insights from your data.
    - **Closed-Loop Learning:** Log new experimental findings to continuously improve the AI's suggestions.
    """)

st.markdown("---")

# --- Getting Started Section ---
st.header("Get Started")
st.markdown("Navigate to the **Data Setup** page from the sidebar to begin your discovery journey.")

if st.button("Go to Data Setup"):
    st.switch_page("pages/1_Data_Setup.py")

