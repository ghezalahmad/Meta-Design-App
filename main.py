import streamlit as st

st.set_page_config(
    page_title="MetaDesign Home",
    page_icon="🏠",
    layout="wide"
)

st.title("Welcome to the MetaDesign Dashboard! 🏠")

st.markdown("""
    This application is your advanced toolkit for accelerating materials discovery using meta-learning and AI-driven experimentation.

    Whether you're exploring new compositions or optimizing for specific properties, MetaDesign guides you to the most promising candidates with fewer experiments, saving time and resources.
""")

st.subheader("How to Get Started:")
st.markdown("""
    1.  **Data Setup:** Navigate to the `1_Data_Setup` page from the sidebar to either upload your existing dataset or create a new design space from scratch using the Digital Lab.
    2.  **Experimentation:** On the `2_Experimentation` page, configure your model, set optimization goals, and run the AI to get suggestions for the next best experiment.
    3.  **Results Analysis:** Once an experiment is complete, visit the `3_Results_Analysis` page to dive deep into the results, visualize the data, and understand the model's suggestions.
""")

st.info("Use the sidebar on the left to navigate between pages. Your data and experiment state will be preserved as you move through the workflow.")

# Add footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>MetaDesign Dashboard - Advanced Meta-Learning for Materials Discovery</p>
    </div>
""", unsafe_allow_html=True)
