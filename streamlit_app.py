import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION & ACCESSIBILITY ---
st.set_page_config(page_title="Module 4: 1D CNN & Evaluation", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Module Settings")
scientific_context = st.sidebar.radio(
    "Select Learning Context:",
    ["Clinical (Patient Care)", "Foundational (Algorithmic & Basic Science)"],
    help="Toggle the interface to display examples relevant to your specific domain."
)
st.sidebar.markdown("---")

st.sidebar.title("Interactive Labs")
mode = st.sidebar.radio(
    "Select an Activity:",
    [
        "Activity 1: 1D CNN on Tabular Data", 
        "Activity 2: 5-Fold Cross-Validation", 
        "Activity 3: Evaluation Metrics Calculator"
    ],
    help="Navigate through the activities to explore the concepts from your notebook."
)

# ==========================================
# ACTIVITY 1: 1D CNN ON TABULAR DATA
# ==========================================
if mode == "Activity 1: 1D CNN on Tabular Data":
    st.title("Activity 1: 1D CNN on Tabular Data")
    
    with st.expander("Activity Instructions (Click to expand)", expanded=True):
        st.write("""
        **Objective:** Understand how a Convolutional Neural Network, typically used for 2D images, can process 1D tabular data.
        
        **Action Items:**
        1. Review the 8 tabular features below.
        2. Adjust the 'Kernel Size' slider to see how the 1D filter groups features together.
        3. Observe how the sliding window captures relationships between adjacent variables.
        """)
        
    if scientific_context == "Clinical (Patient Care)":
        st.header("The Dataset: Clinical Patient Features")
        st.write("In this notebook, we are predicting a binary outcome (Survival/Mortality or Diabetes status) based on 8 patient characteristics. Unlike an image which has height and width, this data is 1-Dimensional.")
        features = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Pedigree", "Age"]
    else:
        st.header("The Dataset: Biological Assay Features")
        st.write("In this notebook, we are predicting a binary biological outcome (e.g., active vs. inactive compound) based on 8 measured biomarkers. Unlike a microscopy image, this assay data is 1-Dimensional.")
        features = ["Assay Alpha", "Biomarker 1", "Biomarker 2", "Protein Level", "Transcript Count", "Cell Mass", "Receptor Affinity", "Specimen Age"]
        
    df_features = pd.DataFrame([features], columns=[f"Feature {i+1}" for i in range(8)], index=["Sample 1"])
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
        st.info(f"The Conv1D filter is looking at **{kernel_size} features** at a time as it slides from left to right.")
        
    with col2:
        st.write("**Simulated Filter Sliding Process:**")
        for i in range(len(features) - kernel_size + 1):
            window = features[i:i+kernel_size]
            st.code(f"Step {i+1}: [ " + " | ".join(window) + " ]")
            
    st.caption("Text Description: The code blocks above demonstrate a sliding window moving across the 1D features one step at a time. This allows the Conv1D layer to learn mathematical relationships between adjacent variables.")

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
        3. Read the analysis of why this method is crucial for small datasets.
        """)
        
    st.header("Interactive Mechanics: Data Splitting")
    st.write("Instead of training the model on one dataset once, K-Fold splits the data into 5 un-overlapped chunks. The model trains and tests 5 separate times, ensuring every data point is used for validation exactly once.")
    
    fold_choice = st.radio(
        "Select the active fold to visualize the data split:",
        ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        horizontal=True,
        help="Select a fold to see which 20% of the data is withheld for validation."
    )
    
    st.write(f"**Current Split: {fold_choice}**")
    
    cols = st.columns(5)
    for i in range(5):
        fold_name = f"Fold {i+1}"
        with cols[i]:
            if fold_choice == fold_name:
                st.markdown(
                    "<div style='background-color:#1E88E5; color:white; padding:20px; text-align:center; border-radius:5px; font-weight:bold;'>Validation Set<br>(20%)</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='background-color:#E0E0E0; color:black; padding:20px; text-align:center; border-radius:5px;'>Training Set<br>(80%)</div>", 
                    unsafe_allow_html=True
                )
                
    st.caption("Text Description: The visual above shows 5 blocks representing the dataset. Four blocks are gray (Training Set), and one block is solid blue (Validation Set). As you select different folds, the blue validation block shifts.")
    
    st.markdown("---")
    st.subheader("Why use K-Fold?")
    if scientific_context == "Clinical (Patient Care)":
        st.write("If we only randomly split the data once, we might accidentally put all the hardest-to-diagnose patients in the test set, making our model look artificially terrible. 5-Fold averages the performance across all splits, giving us a true, robust metric of how the model will perform in the hospital.")
    else:
        st.write("If we only randomly split the data once, we might accidentally put all the most volatile biological samples in the test set, making our model look artificially terrible. 5-Fold averages the performance across all splits, giving us a true, robust metric of how the model will perform in future experiments.")

# ==========================================
# ACTIVITY 3: EVALUATION METRICS CALCULATOR
# ==========================================
elif mode == "Activity 3: Evaluation Metrics Calculator":
    st.title("Activity 3: Evaluation Metrics Calculator")
    
    with st.expander("Activity Instructions (Click to expand)", expanded=True):
        st.write("""
        **Objective:** Understand how the Confusion Matrix drives evaluation metrics (Accuracy, Sensitivity, Specificity, and Precision).
        
        **Action Items:**
        1. Adjust the sliders to simulate True Positives, False Positives, True Negatives, and False Negatives.
        2. Watch how the metrics update in real-time.
        3. Match the sliders to the actual results from the notebook to see the model's true profile.
        """)
        
    st.header("Interactive Mechanics: The Confusion Matrix")
    
    if scientific_context == "Clinical (Patient Care)":
        pos_label = "Positive Condition (e.g., Diabetes/Mortality)"
        neg_label = "Healthy / Survived"
    else:
        pos_label = "Positive Target (e.g., Active Compound / Mutation)"
        neg_label = "Negative Target (e.g., Inactive / Wild-type)"
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Adjust Prediction Outcomes")
        tp = st.slider(f"True Positives (Correctly predicted {pos_label})", 0, 100, 60)
        tn = st.slider(f"True Negatives (Correctly predicted {neg_label})", 0, 100, 83)
        fp = st.slider(f"False Positives (Incorrectly predicted {pos_label})", 0, 100, 17)
        fn = st.slider(f"False Negatives (Incorrectly predicted {neg_label})", 0, 100, 40)

    with col2:
        st.subheader("Calculated Metrics")
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        st.metric("Accuracy (Overall Correctness)", f"{accuracy:.3f}")
        st.metric("Sensitivity / Recall (Ability to catch True Positives)", f"{sensitivity:.3f}")
        st.metric("Specificity (Ability to correctly identify True Negatives)", f"{specificity:.3f}")
        st.metric("Precision (Reliability of a positive prediction)", f"{precision:.3f}")
        
    st.markdown("---")
    st.subheader("Notebook Analysis Reflection")
    if scientific_context == "Clinical (Patient Care)":
        st.info("In Notebook 4, the 1D CNN achieved an average **Accuracy of 0.74**, but a **Sensitivity of only ~0.60**. In a clinical setting, low sensitivity means the model is missing a significant number of positive cases (False Negatives). This is why Accuracy alone is a dangerous metric in healthcare!")
    else:
        st.info("In Notebook 4, the 1D CNN achieved an average **Accuracy of 0.74**, but a **Sensitivity of only ~0.60**. In basic science, low sensitivity means the model is missing a significant number of actual targets (False Negatives). This is why Accuracy alone is insufficient when validating scientific models!")
