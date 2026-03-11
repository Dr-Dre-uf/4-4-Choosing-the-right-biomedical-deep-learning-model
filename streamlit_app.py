import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION & ACCESSIBILITY ---
st.set_page_config(page_title="Notebook 4: 1D CNN Evaluation", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Notebook 4: Activity 1")
scientific_context = st.sidebar.radio(
    "Select Learning Context:",
    ["Clinical (Patient Care)", "Foundational (Algorithmic & Basic Science)"],
    help="Toggle the interface terminology to match your specific track. This changes the labels from patient features to biological assays."
)
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select a Step in Activity 1:",
    [
        "Step 1: Data Preprocessing", 
        "Step 2: Model Architecture", 
        "Step 3: Cross-Validation & Metrics"
    ],
    help="Navigate through the three sequential steps of Activity 1, mirroring the code cells in your Jupyter Notebook."
)

# ==========================================
# STEP 1: PREPROCESSING & 1D CNN
# ==========================================
if mode == "Step 1: Data Preprocessing":
    st.title("Activity 1, Step 1: Preprocessing & 1D CNN")
    
    with st.expander("Step Instructions", expanded=True):
        st.write("""
        **Objective:** Understand why data must be scaled, and how the 1D Kernel reads it.
        **Notebook Connection:** This simulates the `StandardScaler()` and `Conv1D()` functions from the first code blocks of Activity 1.
        """)
        
    
    st.markdown("---")
    
    st.header("Part A: StandardScaler Transformation")
    st.write("In Notebook 4, the data is passed through `StandardScaler` to center the mean at 0 and scale the standard deviation to 1. Toggle the scaler below to see why this is necessary.")
    
    # Raw data from the notebook's df.head()
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Pedigree", "Age"]
    raw_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    scaled_data = [0.6, 0.8, 0.1, 0.9, -0.6, 0.2, -0.4, 1.4] 
    
    apply_scaler = st.toggle(
        "Apply StandardScaler()", 
        value=False,
        help="Simulates the fit_transform() function from sklearn. This forces all features to share the same numerical scale."
    )
    
    display_data = scaled_data if apply_scaler else raw_data
    
    df_visual = pd.DataFrame([display_data], columns=features, index=["Patient 0"])
    st.dataframe(df_visual.style.format("{:.3f}"), use_container_width=True)
    
    if apply_scaler:
        st.success("Data Normalized: Notice how the massive numerical difference between 'Glucose' (148) and 'Pedigree' (0.627) has been eliminated. This prevents large numbers from artificially dominating the neural network's weights.")
    else:
        st.warning("Raw Data: Feeding this directly into a CNN causes the model to over-value 'Glucose' simply because the raw integer is larger than the others.")
        
    st.markdown("---")
    st.header("Part B: The 1D Sliding Kernel")
    st.write("Now that the data is scaled, the Conv1D layer slides a kernel (size=3 in our notebook) across the features.")
    
    current_step = st.slider(
        "Slide the Conv1D Filter (Time Step)", 
        min_value=1, 
        max_value=6, 
        value=1,
        help="Move the slider to manually shift the 1D convolution filter across the array. This mimics how the Conv1D layer processes sequences."
    )
    
    display_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
    for i, feat in enumerate(features):
        val = display_data[i]
        if current_step - 1 <= i < current_step - 1 + 3:
            display_html += f"<div style='background-color:#1E88E5; color:white; padding:10px; border-radius:5px; font-weight:bold; text-align:center;'>{feat}<br>{val:.2f}</div>"
        else:
            display_html += f"<div style='background-color:#E0E0E0; color:black; padding:10px; border-radius:5px; text-align:center;'>{feat}<br>{val:.2f}</div>"
    display_html += "</div>"
    
    st.markdown(display_html, unsafe_allow_html=True)
    st.caption("Text Description: The dark boxes represent the 3 adjacent features currently being multiplied by the kernel's weights to extract a latent pattern.")

# ==========================================
# STEP 2: THE ARCHITECTURE & DROPOUT
# ==========================================
elif mode == "Step 2: Model Architecture":
    st.title("Activity 1, Step 2: The Architecture & Dropout")
    
    with st.expander("Step Instructions", expanded=True):
        st.write("""
        **Objective:** Visualize the 7-layer Sequential model built in the notebook and interact with the Dropout layer.
        **Notebook Connection:** This explores the `model = Sequential([...])` block and the `Dropout(0.3)` function.
        """)
        
    st.header("Notebook 4 Model Architecture")
    
    st.code("""
    model = Sequential([
        Input(shape=(8, 1)),                  # Layer 1: Input
        Conv1D(32, kernel_size=3),            # Layer 2: Feature Extraction
        Conv1D(64, kernel_size=3),            # Layer 3: Deep Feature Extraction
        Flatten(),                            # Layer 4: 1D Vector Conversion
        Dense(32, activation='relu'),         # Layer 5: Fully Connected
        Dropout(0.3),                         # Layer 6: Regularization
        Dense(1, activation='sigmoid')        # Layer 7: Binary Output (0 or 1)
    ])
    # Total Params: 14,593
    """, language="python")
    
    st.markdown("---")
    st.header("Interactive Mechanics: The Dropout Regularizer")
    st.write("In Layer 6, the notebook applies `Dropout(0.3)`. This randomly turns off 30% of the neurons during training so the model does not become overly reliant on any single pathway (preventing overfitting).")
    
    dropout_rate = st.slider(
        "Adjust Dropout Rate", 
        min_value=0.0, 
        max_value=0.9, 
        value=0.3, 
        step=0.1,
        help="0.0 means all neurons are active. 0.9 means 90% of the network is deactivated during a training step. In Notebook 4, this is set to 0.3."
    )
    
    st.write(f"**Simulating 32 Neurons in the Dense Layer (Dropout = {dropout_rate*100:.0f}%)**")
    
    np.random.seed(42) 
    neurons = np.ones(32)
    drop_indices = np.random.choice(32, int(32 * dropout_rate), replace=False)
    neurons[drop_indices] = 0
    
    neuron_html = "<div style='display:flex; gap:10px; flex-wrap:wrap; width: 80%;'>"
    active_count = 0
    for n in neurons:
        if n == 1:
            neuron_html += "<div style='background-color:#4CAF50; width:30px; height:30px; border-radius:15px;' title='Active Neuron'></div>"
            active_count += 1
        else:
            neuron_html += "<div style='background-color:#9E9E9E; width:30px; height:30px; border-radius:15px; opacity:0.3;' title='Dropped Neuron'></div>"
    neuron_html += "</div>"
    
    st.markdown(neuron_html, unsafe_allow_html=True)
    st.caption(f"Text Description: Out of 32 total neurons, {active_count} are active (solid) and {32-active_count} are temporarily deactivated (faded) for this specific training epoch.")

# ==========================================
# STEP 3: 5-FOLD METRICS DASHBOARD
# ==========================================
elif mode == "Step 3: Cross-Validation & Metrics":
    st.title("Activity 1, Step 3: Cross-Validation & Metrics")
    
    with st.expander("Step Instructions", expanded=True):
        st.write("""
        **Objective:** Analyze the actual cross-validation results generated by your notebook.
        **Notebook Connection:** This dashboard visualizes the `kf.split(X)` loop and the final printed output metrics.
        """)
        
    
    st.markdown("---")
    
    fold_data = {
        "Fold 1": {"Acc": 0.695, "Sens": 0.636, "Spec": 0.727, "Prec": 0.565},
        "Fold 2": {"Acc": 0.779, "Sens": 0.681, "Spec": 0.822, "Prec": 0.627},
        "Fold 3": {"Acc": 0.708, "Sens": 0.443, "Spec": 0.882, "Prec": 0.711},
        "Fold 4": {"Acc": 0.765, "Sens": 0.660, "Spec": 0.811, "Prec": 0.608},
        "Fold 5": {"Acc": 0.765, "Sens": 0.569, "Spec": 0.884, "Prec": 0.750}
    }
    
    st.header("Part A: Notebook Output Analysis")
    selected_fold = st.selectbox(
        "Select a Training Fold to View Results:", 
        ["Average (Final Output)"] + list(fold_data.keys()),
        help="Toggle between the evaluation scores of specific folds, or view the overall averaged performance."
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    if selected_fold == "Average (Final Output)":
        st.info("These are the final averaged metrics across all 5 folds, exactly as printed at the end of Notebook 4.")
        acc = 0.742
        sens = 0.598
        spec = 0.825
        prec = 0.652
    else:
        st.info(f"Displaying evaluation metrics isolated to the 20% validation data used in {selected_fold}.")
        acc = fold_data[selected_fold]["Acc"]
        sens = fold_data[selected_fold]["Sens"]
        spec = fold_data[selected_fold]["Spec"]
        prec = fold_data[selected_fold]["Prec"]
        
    col1.metric("Accuracy", f"{acc:.3f}", help="The overall percentage of correct predictions (both positive and negative).")
    col2.metric("Sensitivity", f"{sens:.3f}", help="Also known as Recall. The ability of the model to correctly identify positive disease cases.")
    col3.metric("Specificity", f"{spec:.3f}", help="The ability of the model to correctly identify negative (healthy) cases.")
    col4.metric("Precision", f"{prec:.3f}", help="When the model predicts a positive case, this is how often it is actually correct.")
    
    st.markdown("---")
    st.header("Part B: Interactive Diagnosis")
    st.write("The Notebook asks: *'How is the performance? Is it better than the DNN?'* You will notice the **Average Sensitivity is 0.598**. This means the model is missing approximately 40% of the positive cases.")
    
    st.write("In the notebook, the prediction threshold is hardcoded to `0.5` (`y_pred_prob > 0.5`). Adjust the threshold slider below to see how data scientists fix low sensitivity.")
    
    threshold = st.slider(
        "Prediction Probability Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.1,
        help="The cutoff point. If the model is more than X% confident, it predicts a positive case. Lowering this makes the model more sensitive but less specific."
    )
    
    sim_sens = max(0.1, 0.598 + ((0.5 - threshold) * 0.8))
    sim_spec = max(0.1, 0.825 - ((0.5 - threshold) * 0.5))
    
    st.write(f"**Simulated Performance at Threshold > {threshold}:**")
    
    bar_df = pd.DataFrame({
        "Metric": ["Sensitivity (Catching Disease)", "Specificity (Correctly Healthy)"],
        "Score": [sim_sens, sim_spec]
    })
    st.bar_chart(bar_df.set_index("Metric"), y="Score")
    
    if threshold < 0.5:
        st.success(f"By lowering the threshold to {threshold}, Sensitivity improves to ~{sim_sens:.2f}. The model catches more cases, but at the cost of more False Positives (lower Specificity).")
    elif threshold > 0.5:
        st.error(f"By raising the threshold to {threshold}, Sensitivity drops to ~{sim_sens:.2f}. The model becomes overly conservative and misses even more cases.")
