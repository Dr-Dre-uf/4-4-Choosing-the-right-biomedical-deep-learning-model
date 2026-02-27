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
        1. Set your Kernel Size.
        2. Use the 'Slide the Filter' control to manually step the 1D CNN across the dataset.
        3. Watch how the filter groups adjacent variables to create a new compressed feature map.
        """)
        
    
    st.markdown("---")
    
    if scientific_context == "Clinical (Patient Care)":
        features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Pedigree", "Age"]
    else:
        features = ["Assay Alpha", "Biomarker 1", "Biomarker 2", "Protein Level", "Transcript", "Cell Mass", "Affinity", "Specimen Age"]
        
    st.header("Interactive Mechanics: The Sliding 1D Kernel")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        kernel_size = st.slider("Kernel Size (Filter Width)", min_value=2, max_value=4, value=3)
        max_steps = len(features) - kernel_size + 1
        current_step = st.slider("Slide the Filter (Time Step)", min_value=1, max_value=max_steps, value=1)
        
    with col2:
        st.write("**Raw 1D Input Data:**")
        
        # Create a visual representation of the array
        display_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
        for i, feat in enumerate(features):
            # Highlight the active window
            if current_step - 1 <= i < current_step - 1 + kernel_size:
                display_html += f"<div style='background-color:#1E88E5; color:white; padding:10px; border-radius:5px; font-weight:bold;'>{feat}</div>"
            else:
                display_html += f"<div style='background-color:#E0E0E0; color:black; padding:10px; border-radius:5px;'>{feat}</div>"
        display_html += "</div>"
        
        st.markdown(display_html, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**New Feature Map Extracted by CNN:**")
        
        # Show what the CNN is currently extracting
        active_features = features[current_step - 1 : current_step - 1 + kernel_size]
        st.success(f"Output Node {current_step} ← Mathematical combination of: [ " + " + ".join(active_features) + " ]")
            
    st.caption("Text Description: As you slide the filter, the blue highlighted boxes show exactly which adjacent variables the CNN is currently analyzing to find local patterns. This is how a Conv1D layer processes tabular data.")

# ==========================================
# ACTIVITY 2: 5-FOLD CROSS-VALIDATION
# ==========================================
elif mode == "Activity 2: 5-Fold Cross-Validation":
    st.title("Activity 2: 5-Fold Cross-Validation")
    
    with st.expander("Activity Instructions (Click to expand)", expanded=True):
        st.write("""
        **Objective:** Explore how K-Fold Cross-Validation prevents biased model evaluation.
        
        **Action Items:**
        1. Cycle through the 5 different folds.
        2. Watch the simulated Roster below to see exactly which samples are withheld for validation.
        3. Observe how every single sample gets to be in the 'Validation' set exactly once.
        """)
        
    
    st.markdown("---")
    
    st.header("Interactive Mechanics: Data Splitting Simulator")
    
    fold_choice = st.radio(
        "Select the active training run:",
        [1, 2, 3, 4, 5],
        horizontal=True,
        format_func=lambda x: f"Fold {x}"
    )
    
    st.write(f"**Data Allocation for Fold {fold_choice}:**")
    
    # Simulate a dataset of 25 patients/samples
    total_samples = 25
    samples_per_fold = total_samples // 5
    
    start_val = (fold_choice - 1) * samples_per_fold
    end_val = start_val + samples_per_fold
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='text-align:center;'>Training Pool (80%)</h4>", unsafe_allow_html=True)
        train_list = [f"ID-{i:02d}" for i in range(1, total_samples + 1) if not (start_val < i <= end_val)]
        
        # Display as a clean grid
        df_train = pd.DataFrame(np.array(train_list).reshape(-1, 4))
        st.dataframe(df_train, use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown("<h4 style='text-align:center; color:#1E88E5;'>Validation Pool (20%)</h4>", unsafe_allow_html=True)
        val_list = [f"ID-{i:02d}" for i in range(start_val + 1, end_val + 1)]
        
        df_val = pd.DataFrame(np.array(val_list).reshape(-1, 1), columns=["Withheld for Testing"])
        st.dataframe(df_val, use_container_width=True, hide_index=True)

    st.caption("Text Description: The simulation divides 25 sample IDs. In each fold, exactly 5 unique IDs are moved to the Validation Pool. By the time you reach Fold 5, every single ID has been tested.")

# ==========================================
# ACTIVITY 3: EVALUATION METRICS CALCULATOR
# ==========================================
elif mode == "Activity 3: Evaluation Metrics Calculator":
    st.title("Activity 3: Evaluation Metrics Calculator")
    
    with st.expander("Activity Instructions (Click to expand)", expanded=True):
        st.write("""
        **Objective:** Understand how the Confusion Matrix drives evaluation metrics.
        
        **Action Items:**
        1. Adjust the sliders on the left to simulate prediction outcomes.
        2. Watch the Visual Confusion Matrix update.
        3. Observe the live bar chart to see how False Negatives instantly crash your Sensitivity score.
        """)
        
    
    st.markdown("---")
    
    if scientific_context == "Clinical (Patient Care)":
        pos_label = "Diabetes"
        neg_label = "Healthy"
    else:
        pos_label = "Target Match"
        neg_label = "No Match"
        
    col_sliders, col_matrix, col_chart = st.columns([1.2, 1.5, 1.5])
    
    with col_sliders:
        st.subheader("1. Adjust Predictions")
        tp = st.slider(f"True Positives", 0, 100, 60, help=f"Correctly predicted {pos_label}")
        tn = st.slider(f"True Negatives", 0, 100, 83, help=f"Correctly predicted {neg_label}")
        fp = st.slider(f"False Positives", 0, 100, 17, help=f"Incorrectly predicted {pos_label}")
        fn = st.slider(f"False Negatives", 0, 100, 40, help=f"Incorrectly predicted {neg_label}")

    with col_matrix:
        st.subheader("2. Confusion Matrix")
        # Visual Confusion Matrix
        st.success(f"**True Positives:**\n\n# {tp}")
        st.error(f"**False Positives:**\n\n# {fp}")
        st.warning(f"**False Negatives:**\n\n# {fn}")
        st.info(f"**True Negatives:**\n\n# {tn}")

    with col_chart:
        st.subheader("3. Live Clinical Metrics")
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Real-time Bar Chart of Metrics
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Sensitivity", "Specificity", "Precision"],
            "Score": [accuracy, sensitivity, specificity, precision]
        })
        st.bar_chart(metrics_df.set_index("Metric"), y="Score")
        
        # Highlight Notebook connection
        st.markdown(f"**Current Sensitivity: {sensitivity:.2f}**")
        if sensitivity < 0.70:
            st.error("⚠️ Danger: Sensitivity is too low. The model is missing too many positive cases!")
        else:
            st.success("✅ Sensitivity is stable.")
