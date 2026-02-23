import streamlit as st
import pandas as pd
import numpy as np
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biomedical DL Models", layout="wide")

# --- SIDEBAR NAVIGATION & SETTINGS ---
st.sidebar.title("Module Settings")
scientific_context = st.sidebar.radio(
    "Select Learning Context:",
    ["Clinical (Patient Care)", "Foundational (Algorithmic & Basic Data)"],
    help="Toggle the interface to display examples relevant to your specific domain."
)
st.sidebar.markdown("---")

st.sidebar.title("Interactive Concept Labs")
mode = st.sidebar.radio(
    "Select a Model Architecture:",
    [
        "1. Deep Neural Network (DNN)", 
        "2. Convolutional Neural Network (CNN)", 
        "3. Recurrent Neural Network (LSTM)", 
        "4. Transformer"
    ]
)

# ==========================================
# 1. DEEP NEURAL NETWORK (DNN)
# ==========================================
if mode == "1. Deep Neural Network (DNN)":
    st.title("Deep Neural Network (DNN)")
    
    st.write("""
    **From the Video Script:** DNNs are the foundation for deep learning models. They process covariates (features) in the input layer, extract abstract features through hidden layers, and produce continuous or binary outcomes in the output layer.
    """)
    
    
    st.markdown("---")
    st.header("Interactive Concept: The Overfitting Threshold")
    st.write("The script states: *'DNN usually needs a large sample size to train the model, which prevents overfitting. The number of parameters grows very large when the number of features is high-dimensional.'*")
    
    if scientific_context == "Clinical (Patient Care)":
        feature_desc = "Patient Covariates (e.g., vitals, demographics, lab panels)"
        sample_desc = "Number of Patients in Dataset"
    else:
        feature_desc = "Input Features (e.g., Gene expression levels from human samples)"
        sample_desc = "Number of Biological Samples"
        
    col1, col2 = st.columns([1, 1])
    with col1:
        features = st.slider(f"Number of {feature_desc}", min_value=10, max_value=20000, value=5000, step=100)
        hidden_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=5, value=2)
        samples = st.slider(sample_desc, min_value=100, max_value=50000, value=10000, step=500)
        
    with col2:
        # Simplified parameter math for demonstration
        neurons_per_layer = 64
        first_layer_params = features * neurons_per_layer
        hidden_params = (hidden_layers - 1) * (neurons_per_layer * neurons_per_layer)
        total_params = first_layer_params + hidden_params
        
        st.metric("Total Model Parameters", f"{total_params:,}")
        st.metric("Available Sample Size", f"{samples:,}")
        
        ratio = samples / (total_params + 1)
        
        if ratio > 0.5:
            st.success("✅ Stable Training: Your sample size is large enough to support the parameter count. Overfitting risk is low.")
        else:
            st.error("⚠️ Overfitting Warning: You have significantly more parameters than samples. The model will memorize the training data rather than learning generalized patterns.")
            st.progress(min(ratio * 2, 1.0))

# ==========================================
# 2. CONVOLUTIONAL NEURAL NETWORK (CNN)
# ==========================================
elif mode == "2. Convolutional Neural Network (CNN)":
    st.title("Convolutional Neural Network (CNN)")
    
    st.write("""
    **From the Video Script:** A CNN is a more advanced deep learning model for structured grid-like data. It uses Convolutional Layers to capture local features using kernels, and Pooling Layers to reduce spatial dimensions while preserving important features. Dropout randomly deactivates neurons to prevent overfitting.
    """)
    
    
    st.markdown("---")
    st.header("Interactive Concept: Spatial Compression & Dropout")
    
    if scientific_context == "Clinical (Patient Care)":
        st.write("Imagine this 8x8 matrix represents a localized section of a **digital pathology / histology image**.")
    else:
        st.write("Imagine this 8x8 matrix represents a localized section of **structured grid-like image data**.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Controls**")
        apply_pooling = st.button("Apply 2x2 Max Pooling")
        apply_dropout = st.button("Apply 20% Dropout")
        reset = st.button("Reset Matrix")
        
        # Initialize session state for the grid
        if 'cnn_grid' not in st.session_state or reset:
            np.random.seed(42)
            st.session_state.cnn_grid = np.random.randint(10, 99, size=(8, 8))
            
        if apply_pooling:
            # Simulate 2x2 Max Pooling
            old_grid = st.session_state.cnn_grid
            if old_grid.shape[0] > 2:
                new_shape = old_grid.shape[0] // 2
                new_grid = np.zeros((new_shape, new_shape), dtype=int)
                for i in range(new_shape):
                    for j in range(new_shape):
                        new_grid[i, j] = np.max(old_grid[i*2:i*2+2, j*2:j*2+2])
                st.session_state.cnn_grid = new_grid
            else:
                st.warning("Maximum pooling reached for this simulation.")
                
        if apply_dropout:
            # Simulate Dropout by setting random 20% to 0
            grid = st.session_state.cnn_grid
            mask = np.random.choice([0, 1], size=grid.shape, p=[0.2, 0.8])
            st.session_state.cnn_grid = grid * mask

    with col2:
        current_grid = st.session_state.cnn_grid
        st.write(f"**Current Dimension:** {current_grid.shape[0]} x {current_grid.shape[1]}")
        
        # Display the matrix using a stylized dataframe heatmap
        df = pd.DataFrame(current_grid)
        st.dataframe(df.style.background_gradient(cmap="Blues", vmin=0, vmax=99), use_container_width=True)
        st.caption("Notice how Max Pooling halves the dimensions while keeping the highest values, and Dropout randomly blanks out cells (zeros) to force the network to be robust.")

# ==========================================
# 3. RECURRENT NEURAL NETWORK (LSTM)
# ==========================================
elif mode == "3. Recurrent Neural Network (LSTM)":
    st.title("Recurrent Neural Network (LSTM)")
    
    st.write("""
    **From the Video Script:** An RNN handles sequential data by maintaining a memory of previous inputs. The LSTM variant introduces gates (input, forget, output) to control how much information from the past can be retained.
    """)
    
    
    st.markdown("---")
    st.header("Interactive Concept: Time-Steps & The Forget Gate")
    st.write("Step through time sequentially. Adjust the 'Forget Gate' to see how much of the original context survives to the final prediction step.")
    
    if scientific_context == "Clinical (Patient Care)":
        sequence = ["Visit 1: Elevated BP", "Visit 2: Weight Gain", "Visit 3: Mild Chest Pain", "Visit 4: Predict Status..."]
    else:
        sequence = ["Word 1: The", "Word 2: patient", "Word 3: exhibits", "Word 4: Predict next..."]
        
    forget_gate = st.slider("LSTM Forget Gate (0.0 = Forget Quickly, 1.0 = Retain Perfectly)", 0.0, 1.0, 0.7, 0.1)
    time_step = st.radio("Step Through Sequence Time ($T$):", [0, 1, 2, 3], horizontal=True, format_func=lambda x: f"T={x+1}")
    
    st.subheader(f"Current Input: **{sequence[time_step]}**")
    
    st.write("**Hidden State Memory (Context Retained):**")
    
    # Calculate decaying memory for all previous steps
    for t in range(time_step + 1):
        if t == time_step:
            st.info(f"🟢 **{sequence[t]}** (Current Input - 100% active)")
        else:
            retention = forget_gate ** (time_step - t)
            if retention > 0.1:
                st.success(f"🧠 **{sequence[t]}** (Retained at {retention*100:.0f}%)")
            else:
                st.error(f"🌫️ **{sequence[t]}** (Forgotten - dropped below 10%)")
                
    if time_step == 3 and forget_gate < 0.5:
        st.warning("Notice how a low forget gate causes the model to lose the crucial context from T=1 by the time it reaches the final prediction.")

# ==========================================
# 4. TRANSFORMER
# ==========================================
elif mode == "4. Transformer":
    st.title("Transformer")
    
    st.write("""
    **From the Video Script:** A Transformer translates an input sequence into an output sequence using an Encoder and Decoder. It excels at Natural Language Processing (NLP) because self-attention enables long-range interactions across the entire sequence simultaneously.
    """)
    
    
    st.markdown("---")
    st.header("Interactive Concept: The Self-Attention Matrix")
    
    if scientific_context == "Clinical (Patient Care)":
        st.write("Unlike an LSTM that reads one step at a time, a Transformer looks at the entire sequence of clinical notes at once. It mathematically calculates how strongly every word relates to every other word.")
        words = ["Patient", "denies", "pain", "but", "exhibits", "fever"]
    else:
        st.write("Unlike an LSTM that reads one step at a time, a Transformer looks at the entire input sequence at once. It mathematically calculates how strongly every sequence element relates to the others.")
        words = ["Model", "predicts", "protein", "folding", "structure", "accurately"]

    st.write("**Simulated Attention Heatmap**")
    
    # Generate a mock attention matrix based on a slider
    focus_level = st.select_slider("Attention Focus Level", options=["Local Context Only", "Long-Range Interactions (Standard)", "Over-Attending (Noisy)"], value="Long-Range Interactions (Standard)")
    
    np.random.seed(10)
    matrix = np.zeros((6, 6))
    
    if focus_level == "Local Context Only":
        # Diagonal focus (words only attend to neighbors)
        for i in range(6):
            matrix[i, i] = 1.0
            if i > 0: matrix[i, i-1] = 0.5
            if i < 5: matrix[i, i+1] = 0.5
    elif focus_level == "Long-Range Interactions (Standard)":
        # Words attend to meaningful connections far away
        matrix = np.random.uniform(0.1, 0.4, (6, 6))
        for i in range(6): matrix[i, i] = 1.0 # Self
        matrix[0, 5] = 0.9 # First word attends heavily to last word
        matrix[5, 0] = 0.9
    else:
        # Everything attends to everything randomly
        matrix = np.random.uniform(0.6, 1.0, (6, 6))
        
    df_attention = pd.DataFrame(matrix, index=words, columns=words)
    st.dataframe(df_attention.style.background_gradient(cmap="Purples", vmin=0, vmax=1.0), use_container_width=True)
    
    st.caption("The dark purple squares indicate where the model is 'paying attention' to establish context. Notice how long-range interactions allow the first word to directly connect with the last word without relying on a step-by-step memory bottleneck.")
