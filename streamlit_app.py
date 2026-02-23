import streamlit as st
import pandas as pd
import numpy as np

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

st.sidebar.title("Interactive Labs")
mode = st.sidebar.radio(
    "Select a Sandbox Environment:",
    [
        "Lab 1: Architecture Mechanics", 
        "Lab 2: System Deployment Simulator"
    ],
    help="Navigate between the mechanical breakdown of individual models and the deployment simulator."
)

# ==========================================
# LAB 1: ARCHITECTURE MECHANICS
# ==========================================
if mode == "Lab 1: Architecture Mechanics":
    st.title("Lab 1: Architecture Mechanics")
    
    with st.expander("Lab Instructions", expanded=True):
        st.write("""
        **Objective:** Visually explore how different neural network architectures process data, manage memory, and scale computationally.
        
        **Action Items:**
        1. Select an architecture from the dropdown menu.
        2. Read the structural breakdown to understand the model's core components.
        3. Manipulate the environment variables in the interactive control panel.
        4. Observe the dynamic charts to see exactly how your adjustments impact the model's structural footprint.
        """)
        
    model_choice = st.selectbox(
        "Select an Architecture to Inspect:",
        ["Deep Neural Network (DNN)", "Convolutional Neural Network (CNN)", "Recurrent Neural Network (LSTM)", "Transformer"],
        help="Loads the mechanical simulator and architectural blueprints for the selected model."
    )
    
    st.markdown("---")
    
    if model_choice == "Deep Neural Network (DNN)":
        st.subheader("Deep Neural Network (DNN)")
        st.write("DNNs extract increasingly abstract features from covariates, usually in the same domain. However, they require careful monitoring to prevent overfitting when dealing with high-dimensional data.")
        
        
        st.markdown("### Interactive Control Panel: Parameter Scaling")
        if scientific_context == "Clinical (Patient Care)":
            feature_label = "Input Patient Covariates"
            feature_val = 500
            max_val = 5000
        else:
            feature_label = "Input Gene Expressions"
            feature_val = 10000
            max_val = 20000
            
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Adjust the network depth and width to observe the parameter distribution.**")
            features = st.slider(feature_label, min_value=10, max_value=max_val, value=feature_val, help="The volume of raw input data feeding into the first layer of the network.")
            hidden_layers = st.slider("Hidden Layers", min_value=1, max_value=10, value=3, help="The depth of the network. More layers extract more abstract features but increase complexity.")
            neurons_per_layer = st.slider("Neurons per Layer", min_value=16, max_value=512, value=64, step=16, help="The width of each hidden layer. More neurons increase the model's capacity to learn patterns.")
        
        with col2:
            first_layer_params = (features * neurons_per_layer) + neurons_per_layer
            hidden_params = (hidden_layers - 1) * ((neurons_per_layer * neurons_per_layer) + neurons_per_layer)
            total_params = first_layer_params + hidden_params + (neurons_per_layer * 1) + 1 
            
            st.metric("Total Model Parameters", f"{total_params:,}", help="The total sum of weights and biases the model must optimize during training.")
            
            # Dynamic visualization of parameter distribution
            param_data = pd.DataFrame({
                "Layer Type": ["Input to First Hidden", "Between Hidden Layers"],
                "Parameter Count": [first_layer_params, hidden_params]
            })
            st.bar_chart(param_data.set_index("Layer Type"))
            st.caption("Notice how high-dimensional inputs cause the first layer parameters to completely dominate the network's computational capacity.")

    elif model_choice == "Convolutional Neural Network (CNN)":
        st.subheader("Convolutional Neural Network (CNN)")
        st.write("CNNs use convolutional layers and pooling to extract hierarchical features from structured, grid-like data while preserving spatial relationships.")
        
        
        st.markdown("### Interactive Control Panel: Spatial Compression")
        if scientific_context == "Clinical (Patient Care)":
            img_label = "fMRI Scan Resolution"
        else:
            img_label = "Raw Image Resolution"
            
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Apply pooling layers to observe the mathematical reduction in spatial dimensions.**")
            img_size = st.select_slider(img_label, options=[256, 512, 1024, 2048, 4096], value=1024, help="The initial pixel width and height of the input grid.")
            pools = st.slider("Number of 2x2 Max Pooling Layers", min_value=0, max_value=5, value=1, help="Each pooling layer halves the spatial dimensions, compressing the data while retaining the highest activation values.")
            
        with col2:
            sizes = [img_size]
            for i in range(pools):
                sizes.append(sizes[-1] // 2)
                
            reduction = 100 - ((sizes[-1]**2) / (img_size**2) * 100)
            st.metric("Final Output Dimension", f"{sizes[-1]} x {sizes[-1]}", help="The final grid size passed to the fully connected dense layers.")
            st.metric("Spatial Data Compressed By", f"{reduction:.2f}%", help="The percentage of raw spatial data discarded to improve computational efficiency.")
            
            # Dynamic visual of dimension reduction
            chart_data = pd.DataFrame({"Layer Step": [f"Pool {i}" for i in range(pools + 1)], "Grid Size (1D)": sizes})
            st.line_chart(chart_data.set_index("Layer Step"))
            st.caption("Observe the exponential decay of spatial dimensions as the data passes deeper into the CNN architecture.")

    elif model_choice == "Recurrent Neural Network (LSTM)":
        st.subheader("Recurrent Neural Network (LSTM)")
        st.write("LSTMs handle sequential data by maintaining a hidden state. They utilize specialized gates to control how much past information is retained for future prediction.")
        
        
        st.markdown("### Interactive Control Panel: Memory Retention")
        if scientific_context == "Clinical (Patient Care)":
            st.write("**Scenario:** Processing a patient's historical visit records to predict current status.")
        else:
            st.write("**Scenario:** Processing a linguistic sequence step-by-step to predict the next word.")
            
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Adjust the retention factor to see how memory degrades over sequential time steps.**")
            forget_gate = st.slider("Information Retention Factor", min_value=0.1, max_value=1.0, value=0.5, step=0.1, help="Represents the percentage of information the LSTM's forget gate allows to pass to the next time step. 1.0 represents perfect memory.")
            steps = st.slider("Sequence Length (Time Steps)", min_value=2, max_value=20, value=10, help="The number of sequential inputs the model must process.")
        
        with col2:
            # Calculate decay over time
            memory_strength = [forget_gate ** i for i in range(steps)]
            decay_data = pd.DataFrame({
                "Time Steps Ago": list(range(steps)),
                "Memory Signal Strength": memory_strength
            })
            
            st.write("**Signal Strength of the First Event Over Time**")
            st.area_chart(decay_data.set_index("Time Steps Ago"))
            st.caption("Observe the vanishing gradient. If the retention factor is too low, the model completely forgets the earliest data points by the end of the sequence.")

    elif model_choice == "Transformer":
        st.subheader("Transformer")
        st.write("Transformers translate input sequences into output sequences using an Encoder and Decoder structure. They utilize self-attention to process all data points simultaneously.")
        
        
        st.markdown("### Interactive Control Panel: Self-Attention Scaling")
        if scientific_context == "Clinical (Patient Care)":
            st.write("**Scenario:** Mining long-range dependencies in comprehensive clinical notes.")
        else:
            st.write("**Scenario:** Processing massive linguistic inputs for an NLP backbone model.")
            
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Adjust the sequence length to observe the computational cost of self-attention.**")
            seq_length = st.slider("Input Sequence Length (Tokens)", min_value=10, max_value=2000, value=100, step=10, help="The total number of words, sub-words, or data points in the input sequence.")
        
        with col2:
            interactions = seq_length * seq_length
            st.metric("Simultaneous Attention Interactions", f"{interactions:,}", help="Because every token mathematically attends to every other token, the compute cost scales quadratically.")
            
            # Show quadratic scaling curve
            scale_range = np.arange(10, seq_length + 10, 50)
            compute_cost = scale_range ** 2
            scale_data = pd.DataFrame({
                "Sequence Length": scale_range,
                "Compute Cost (Interactions)": compute_cost
            })
            st.line_chart(scale_data.set_index("Sequence Length"))
            st.caption("The quadratic curve demonstrates why Transformers excel at long-range context but require massive parallel processing power for long sequences.")

# ==========================================
# LAB 2: SYSTEM DEPLOYMENT SIMULATOR
# ==========================================
elif mode == "Lab 2: System Deployment Simulator":
    st.title("Lab 2: System Deployment Simulator")
    
    with st.expander("Lab Instructions", expanded=True):
        st.write("""
        **Objective:** Practice configuring the correct deep learning pipeline based on the underlying structure of incoming data streams.
        
        **Action Items:**
        1. Review the active data stream scenario on the left.
        2. Use the configuration panel to deploy the optimal neural network architecture.
        3. Observe the system readout. If the architecture is incompatible with the data structure, the pipeline will fail, and the system will provide an optimization warning.
        """)
        
    if scientific_context == "Clinical (Patient Care)":
        cases = {
            "Data Stream Alpha: Personal characteristics and covariates for survival prediction.": "Deep Neural Network (DNN)",
            "Data Stream Beta: fMRI grid-like data to predict brain-related disorders.": "Convolutional Neural Network (CNN)",
            "Data Stream Gamma: Step-by-step sequential medical history updates.": "Recurrent Neural Network (LSTM)",
            "Data Stream Delta: Complex, long-range dependencies across years of unstructured medical history.": "Transformer"
        }
    else:
        cases = {
            "Data Stream Alpha: Independent covariates in the same domain (20,000 gene expressions).": "Deep Neural Network (DNN)",
            "Data Stream Beta: Raw grid-like data (pixels from microscopy images).": "Convolutional Neural Network (CNN)",
            "Data Stream Gamma: Linguistic sequences processed step-by-step using strict sentence context.": "Recurrent Neural Network (LSTM)",
            "Data Stream Delta: NLP sequence-to-sequence translation (ChatGPT backbone) requiring broad context.": "Transformer"
        }

    st.markdown("---")
    
    for case_text, correct_model in cases.items():
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.info(f"**{case_text}**")
            
        with col2:
            user_choice = st.selectbox(
                "Deploy Architecture:", 
                ["Select Model...", "Deep Neural Network (DNN)", "Convolutional Neural Network (CNN)", "Recurrent Neural Network (LSTM)", "Transformer"],
                key=case_text,
                help="Select the model engineered to handle this specific mathematical structure."
            )
            
            if user_choice != "Select Model...":
                if user_choice == correct_model:
                    st.success("Pipeline Active: Architecture perfectly matches data topology.")
                else:
                    st.error("Pipeline Failed: Sub-optimal architecture selected for this data structure. Review the module guidelines.")
        st.markdown("---")
