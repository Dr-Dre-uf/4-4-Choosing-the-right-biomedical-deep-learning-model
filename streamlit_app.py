import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biomedical DL Models", layout="wide")

st.sidebar.title("DL Learning Module")
mode = st.sidebar.radio(
    "Select an Activity:",
    [
        "Activity 1: Architecture Mechanics", 
        "Activity 2: Model Selection Simulator", 
        "Activity 3: Knowledge Check"
    ]
)

# ==========================================
# ACTIVITY 1: ARCHITECTURE MECHANICS (NEWLY INTERACTIVE)
# ==========================================
if mode == "Activity 1: Architecture Mechanics":
    st.title("Activity 1: Under the Hood of DL Models")
    
    with st.expander("📝 Activity Instructions", expanded=True):
        st.write("""
        1. Select an architecture.
        2. Interact with the model's core mechanics to see how it processes data mathematically.
        """)
        
    model_choice = st.selectbox(
        "Select an Architecture to Inspect:",
        ["Deep Neural Network (DNN)", "Convolutional Neural Network (CNN)", "Recurrent Neural Network (LSTM)", "Transformer"]
    )
    
    st.markdown("---")
    
    if model_choice == "Deep Neural Network (DNN)":
        st.subheader("Deep Neural Network (DNN)")
        st.write("DNNs are the foundation, but they are sensitive to high-dimensional inputs.")
        
        
        st.markdown("### 🧮 Interactive: The Parameter Explosion")
        st.write("Adjust the inputs below to see why the script warns that DNN parameters grow very large!")
        
        col1, col2 = st.columns(2)
        with col1:
            features = st.slider("Number of Input Features (e.g., Genes)", min_value=10, max_value=20000, value=100, step=100)
            hidden_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=10, value=2)
            neurons_per_layer = st.slider("Neurons per Hidden Layer", min_value=16, max_value=512, value=64, step=16)
        
        with col2:
            # Simple parameter calculation: (input * neurons) + (hidden layers-1)*(neurons*neurons) + (neurons * output) + biases
            first_layer_params = (features * neurons_per_layer) + neurons_per_layer
            hidden_params = (hidden_layers - 1) * ((neurons_per_layer * neurons_per_layer) + neurons_per_layer)
            output_params = (neurons_per_layer * 1) + 1 # Assuming binary output
            total_params = first_layer_params + hidden_params + output_params
            
            st.metric("Total Model Parameters", f"{total_params:,}")
            if total_params > 1000000:
                st.error("⚠️ Overfitting Risk High! You need a massive sample size to train this many parameters.")
            else:
                st.success("✅ Manageable parameter count.")

    elif model_choice == "Convolutional Neural Network (CNN)":
        st.subheader("Convolutional Neural Network (CNN)")
        st.write("CNNs handle grid-like data (like histology images) efficiently using kernels and pooling.")
        
        
        st.markdown("### 📉 Interactive: Spatial Dimension Reduction")
        st.write("The script states: *'Pooling Layers reduce spatial dimensions while preserving important features.'* See it in action.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            img_size = st.select_slider("Input Image Size (Pixels)", options=[64, 128, 256, 512, 1024], value=256)
            st.info(f"Input Tensor: {img_size} x {img_size}")
            
        with col2:
            apply_pool1 = st.checkbox("Apply Max Pooling Layer 1 (2x2)", value=True)
            apply_pool2 = st.checkbox("Apply Max Pooling Layer 2 (2x2)", value=False)
            
        with col3:
            current_size = img_size
            if apply_pool1: current_size = current_size // 2
            if apply_pool2: current_size = current_size // 2
            
            reduction = 100 - ((current_size**2) / (img_size**2) * 100)
            
            st.metric("Output Dimension", f"{current_size} x {current_size}")
            st.metric("Data Size Reduced By", f"{reduction:.1f}%")

    elif model_choice == "Recurrent Neural Network (LSTM)":
        st.subheader("Recurrent Neural Network (LSTM)")
        st.write("LSTMs introduce gates to control how much past information is retained for future prediction.")
        
        
        st.markdown("### 🧠 Interactive: The Forget Gate Simulator")
        st.write("Simulate processing a patient's sequential medical history.")
        
        history = ["2015: Mild Asthma", "2018: High Blood Pressure", "2022: Type 2 Diabetes", "2025: Current Visit"]
        st.write(f"**Patient Sequence:** `{' ➔ '.join(history)}`")
        
        forget_gate = st.slider("Forget Gate Value (0 = Forget All Past, 1 = Remember All Past)", 0.0, 1.0, 0.5, 0.1)
        
        st.write("**Model's Context Memory at 'Current Visit':**")
        retained_memory = []
        for i, event in enumerate(history[:-1]):
            strength = forget_gate ** (len(history) - 2 - i) # decays based on gate and distance
            if strength > 0.1:
                retained_memory.append(f"{event} (Strength: {strength:.2f})")
                
        if not retained_memory:
            st.warning("The model has forgotten the past medical history! It is only looking at the current visit.")
        else:
            for mem in retained_memory:
                st.success(mem)

    elif model_choice == "Transformer":
        st.subheader("Transformer")
        st.write("Transformers use an Encoder/Decoder structure and self-attention to translate input sequences to output sequences.")
        
        
        st.markdown("### 👁️ Interactive: Long-Range Self-Attention")
        st.write("Unlike RNNs that process step-by-step, self-attention looks at *all* data points simultaneously.")
        
        seq_length = st.slider("Sequence Length (Words or Clinical Notes)", 10, 500, 100)
        
        st.write("Because every word looks at every other word, attention calculations grow quadratically ($N^2$):")
        interactions = seq_length * seq_length
        st.metric("Total Attention Interactions Computed at Once", f"{interactions:,}")
        st.info("This is why Transformers are so powerful for finding connections in complex patient histories, but require significant computational power!")

# ==========================================
# ACTIVITY 2: MODEL SELECTION SIMULATOR
# ==========================================
elif mode == "Activity 2: Model Selection Simulator":
    st.title("Activity 2: The Biomedical Matchmaker")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Select Your Biomedical Data:")
        data_type = st.radio("What are you analyzing?", [
            "Gene expression matrix (20,000 genes) to predict cell viability",
            "Histology images to predict cancer survival time",
            "Sequential patient medical history to predict future status",
            "Complex natural language clinical notes (long-range dependencies)"
        ])
        
    with col2:
        st.subheader("Algorithm Analysis")
        if "Gene expression" in data_type:
            st.success("### Optimal Fit: Deep Neural Network (DNN)")
            st.write("**Why:** DNNs handle distinct covariates in the same domain (like 20,000 individual genes). Just watch out for the high-dimensional parameter explosion!")
        elif "Histology" in data_type:
            st.success("### Optimal Fit: Convolutional Neural Network (CNN)")
            st.write("**Why:** CNNs use kernels and pooling to capture local spatial features in structured, grid-like image data.")
        elif "Sequential patient" in data_type:
            st.success("### Optimal Fit: Recurrent Neural Network (LSTM)")
            st.write("**Why:** LSTMs are explicitly designed to maintain a hidden state that carries past information step-by-step.")
        elif "Complex natural language" in data_type:
            st.success("### Optimal Fit: Transformer")
            st.write("**Why:** Transformers use self-attention to process entire sequences at once, outperforming RNNs on long-range linguistic dependencies.")

# ==========================================
# ACTIVITY 3: KNOWLEDGE CHECK (Same as previous version)
# ==========================================
elif mode == "Activity 3: Knowledge Check":
    # [Insert the exact same quiz code from the previous response here to save space]
    st.write("*(Quiz code remains identical to the previous version)*")
