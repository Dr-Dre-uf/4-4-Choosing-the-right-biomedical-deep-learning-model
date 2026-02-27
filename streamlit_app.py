import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION & ACCESSIBILITY ---
# Ensure the layout is wide for screen-magnifier users and titles are descriptive
st.set_page_config(page_title="Module 4: 1D CNN & Clinical Evaluation", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Module 4 Interactive Labs")
st.sidebar.markdown("Navigate through the activities below to explore the concepts from your notebook.")

mode = st.sidebar.radio(
    "Select an Activity:",
    [
        "Activity 1: 1D CNN on Tabular Data", 
        "Activity 2: 5-Fold Cross-Validation", 
        "Activity 3: Clinical Metrics Calculator"
    ],
    help="Use the up and down arrow keys to navigate the menu options."
)

# ==========================================
# ACTIVITY 1: 1D CNN ON TABULAR DATA
# ==========================================
if mode == "Activity 1: 1D CNN on Tabular Data":
    st.title("Activity 1: 1D CNN on Tabular Data")
    
    with st.expander("Activity Instructions (Click to expand)", expanded=True):
        st.write("""
        **Objective:** Understand how a Convolutional Neural Network, typically used for 2D images, can process 1D clinical tabular data.
        
        **Action Items:**
        1. Review the 8 clinical features from the diabetes dataset.
        2. Adjust the 'Kernel Size' slider to see how the 1D filter groups features together.
        3. Observe how the data must be reshaped before feeding it into the CNN.
        """)
        
    st.header("The Dataset: Diabetes Clinical Features")
    st.write("In this notebook, we are predicting a binary outcome (0 or 1) based on 8 patient characteristics. Unlike an image which has height and width, this data is 1-Dimensional.")
    
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age"]
    df_features = pd.DataFrame([features], columns=[f"Feature {i+1}" for i in range(8)], index=["Patient X"])
    st.dataframe(df_features, use_container_width=True)
    
    st.markdown("---")
    st.header("Interactive Mechanics: The 1D Kernel")
    st.write("A 1D CNN slides a 'kernel' (a small filter) across the sequence of features to find local patterns. Adjust the kernel size below.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        kernel_size = st.slider(
            "Kernel Size", 
            min_value=2, max_value=4, value=3, 
            help="Determines how many adjacent features the filter looks at simultaneously."
        )
        st.info(f"The filter is looking at **{kernel_size} features** at a time as it slides from left to right.")
        
    with col2:
        st.write("**Simulated Filter Sliding Process:**")
        # Visualizing the 1D sliding window
        for i in range(len(features) - kernel_size + 1):
            window = features[i:i+kernel_size]
            st.code(f"Step {i+1}: [ " + " | ".join(window) + " ]")
            
    st.caption("Text Description: The code blocks above demonstrate a sliding window moving across the patient features one step at a time. This allows the Conv1D layer to learn relationships between adjacent clinical variables.")

# ==========================================
# ACTIVITY 2: 5-FOLD CROSS-VALIDATION
# ==========================================
elif mode == "Activity 2: 5-Fold Cross-Validation":
    st.title("Activity 2: 5-Fold Cross-Validation")
    
    with st.expander("Activity Instructions (Click to expand)", expanded=True):
        st.write("""
        **Objective:** Explore how K-Fold Cross-Validation prevents biased model evaluation.
        
        **Action Items:**
        1. Use the radio buttons to cycle through the 5 different folds.
        2. Observe how the training and validation datasets shift.
        3. Read the analysis of why this method is crucial for small clinical datasets.
        """)
        
    st.header("Interactive Mechanics: Data Splitting")
    st.write("Instead of training the model on one dataset once, K-Fold splits the data into 5 un-overlapped chunks. The model trains and tests 5 separate times, ensuring every data point is used for validation exactly once.")
    
    fold_choice = st.radio(
        "Select the active fold to visualize the data split:",
        ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        horizontal=True,
        help="Select a fold to see which 20% of the data is withheld for validation."
    )
    
    # Accessible High-Contrast Visualization of Folds
    st.write(f"**Current Split: {fold_choice}**")
    
    cols = st.columns(5)
    for i in range(5):
        fold_name = f"Fold {i+1}"
        with cols[i]:
            if fold_choice == fold_name:
                # High contrast highlight for validation set
                st.markdown(
                    "<div style='background-color:#1E88E5; color:white; padding:20px; text-align:center; border-radius:5px; font-weight:bold;'>Validation Set<br>(20%)</div>", 
                    unsafe_allow_html=True
                )
            else:
                # Subdued color for training set
                st.markdown(
                    "<div style='background-color:#E0E0E0; color:black; padding:20px; text-align:center; border-radius:5px;'>Training Set<br>(80%)</div>", 
                    unsafe_allow_html=True
                )
                
    st.caption("Text Description: The visual above shows 5 blocks representing the dataset. Four blocks are gray (Training Set), and one block is solid blue (Validation Set). As you select different folds, the blue validation block shifts, ensuring the model is tested on all parts of the data.")
    
    st.markdown("---")
    st.subheader("Why use K-Fold?")
    st.write("If we only randomly split the data once, we might accidentally put all the hardest-to-diagnose patients in the test set, making our model look artificially terrible. 5-Fold averages the performance across all splits, giving us a true, robust metric of how the model will perform in the real world.")

# ==========================================
# ACTIVITY 3: CLINICAL METRICS CALCULATOR
# ==========================================
elif mode == "Activity 3: Clinical Metrics Calculator":
    st.title("Activity 3: Clinical Metrics Breakdown")
    
    with st.expander("Activity Instructions (Click to expand)", expanded=True):
        st.write("""
        **Objective:** Understand how the Confusion Matrix drives the clinical evaluation metrics (Accuracy, Sensitivity, Specificity, and Precision).
        
        **Action Items:**
        1. Adjust the sliders to simulate True Positives, False Positives, True Negatives, and False Negatives.
        2. Watch how the metrics update in real-time.
        3. Match the sliders to the actual results from the notebook to see the model's true clinical profile.
        """)
        
    st.header("Interactive Mechanics: The Confusion Matrix")
    st.write("Adjust the raw predictions to see how they impact the clinical evaluation metrics.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Adjust Prediction Outcomes")
        # Defaults roughly align with the Notebook's average performance
        tp = st.slider("True Positives (Correctly predicted Diabetes)", 0, 100, 60, help="Patients who have the condition and were correctly identified.")
        tn = st.slider("True Negatives (Correctly predicted Healthy)", 0, 100, 83, help="Patients who are healthy and were correctly identified as healthy.")
        fp = st.slider("False Positives (Incorrectly predicted Diabetes)", 0, 100, 17, help="Patients who are healthy but were incorrectly flagged as having the condition.")
        fn = st.slider("False Negatives (Incorrectly predicted Healthy)", 0, 100, 40, help="Patients who have the condition but were missed by the model.")

    with col2:
        st.subheader("Calculated Clinical Metrics")
        
        # Prevent division by zero
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        st.metric("Accuracy (Overall Correctness)", f"{accuracy:.3f}")
        st.metric("Sensitivity / Recall (Ability to catch the disease)", f"{sensitivity:.3f}")
        st.metric("Specificity (Ability to correctly identify healthy patients)", f"{specificity:.3f}")
        st.metric("Precision (Reliability of a positive diagnosis)", f"{precision:.3f}")
        
    st.markdown("---")
    st.subheader("Notebook Analysis Reflection")
    st.info("In Notebook 4, the 1D CNN achieved an average **Accuracy of 0.74**, but a **Sensitivity of only 0.59**. In a clinical setting, low sensitivity means the model is missing a significant number of positive cases (False Negatives). This is why Accuracy alone is a dangerous metric in healthcare!")
