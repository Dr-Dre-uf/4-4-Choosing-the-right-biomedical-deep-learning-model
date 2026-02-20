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
    ],
    help="Navigate through the module. Start with Architecture Mechanics to learn the concepts, use the Simulator to apply them, and finish with the Knowledge Check."
)

# ==========================================
# ACTIVITY 1: ARCHITECTURE MECHANICS
# ==========================================
if mode == "Activity 1: Architecture Mechanics":
    st.title("Activity 1: Under the Hood of DL Models")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        **Objective:** Understand the mathematical and structural differences between deep learning architectures.
        
        **Steps:**
        1. Select a deep learning architecture from the dropdown menu below.
        2. Read the structural breakdown to understand how the model processes data.
        3. Use the interactive sliders and toggles in the 'Interactive Mechanics' section to simulate how the model handles parameters, dimensions, and memory.
        4. Observe the changing metrics to see the algorithm's strengths and limitations in real-time.
        """)
        
    model_choice = st.selectbox(
        "Select an Architecture to Inspect:",
        ["Deep Neural Network (DNN)", "Convolutional Neural Network (CNN)", "Recurrent Neural Network (LSTM)", "Transformer"],
        help="Choose a model to load its specific architecture diagram and interactive simulator."
    )
    
    st.markdown("---")
    
    if model_choice == "Deep Neural Network (DNN)":
        st.subheader("Deep Neural Network (DNN)")
        st.write("DNNs are the foundation, but they are sensitive to high-dimensional inputs.")
        
        
        st.markdown("### Interactive Mechanics: The Parameter Explosion")
        st.write("Adjust the inputs below to see why the module warns that DNN parameters grow very large with high-dimensional data.")
        
        col1, col2 = st.columns(2)
        with col1:
            features = st.slider(
                "Number of Input Features", 
                min_value=10, max_value=20000, value=100, step=100,
                help="Represents your raw data, such as the number of genes sequenced or patient covariates."
            )
            hidden_layers = st.slider(
                "Number of Hidden Layers", 
                min_value=1, max_value=10, value=2,
                help="Adding layers makes the network 'deeper', allowing it to learn more complex patterns but increasing computational cost."
            )
            neurons_per_layer = st.slider(
                "Neurons per Hidden Layer", 
                min_value=16, max_value=512, value=64, step=16,
                help="The number of computational units in each hidden layer."
            )
        
        with col2:
            first_layer_params = (features * neurons_per_layer) + neurons_per_layer
            hidden_params = (hidden_layers - 1) * ((neurons_per_layer * neurons_per_layer) + neurons_per_layer)
            output_params = (neurons_per_layer * 1) + 1 
            total_params = first_layer_params + hidden_params + output_params
            
            st.metric("Total Model Parameters", f"{total_params:,}", help="The total number of weights and biases the model must learn.")
            if total_params > 1000000:
                st.error("Overfitting Risk High! You need a massive sample size to train this many parameters effectively.")
            else:
                st.success("Manageable parameter count. This model is less prone to overfitting on standard datasets.")

    elif model_choice == "Convolutional Neural Network (CNN)":
        st.subheader("Convolutional Neural Network (CNN)")
        st.write("CNNs handle grid-like data efficiently using kernels and pooling.")
        
        
        st.markdown("### Interactive Mechanics: Spatial Dimension Reduction")
        st.write("The script states: 'Pooling Layers reduce spatial dimensions while preserving important features.' See how pooling shrinks data mathematically.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            img_size = st.select_slider(
                "Input Image Size (Pixels)", 
                options=[64, 128, 256, 512, 1024], value=256,
                help="The resolution of the input image, such as a histology slide. Higher resolution means exponentially more data."
            )
            st.info(f"Input Tensor: {img_size} x {img_size}")
            
        with col2:
            apply_pool1 = st.checkbox(
                "Apply Max Pooling Layer 1 (2x2)", value=True,
                help="A 2x2 pooling layer cuts the height and width of the image in half, keeping only the most activated features."
            )
            apply_pool2 = st.checkbox(
                "Apply Max Pooling Layer 2 (2x2)", value=False,
                help="Applying a second pooling layer further compresses the data before passing it to the dense layers."
            )
            
        with col3:
            current_size = img_size
            if apply_pool1: current_size = current_size // 2
            if apply_pool2: current_size = current_size // 2
            
            reduction = 100 - ((current_size**2) / (img_size**2) * 100)
            
            st.metric("Output Dimension", f"{current_size} x {current_size}")
            st.metric("Data Size Reduced By", f"{reduction:.1f}%", help="The percentage of spatial data discarded, drastically reducing required parameters.")

    elif model_choice == "Recurrent Neural Network (LSTM)":
        st.subheader("Recurrent Neural Network (LSTM)")
        st.write("LSTMs introduce gates to control how much past information is retained for future prediction.")
        
        
        st.markdown("### Interactive Mechanics: The Forget Gate Simulator")
        st.write("Simulate processing a patient's sequential medical history over time.")
        
        history = ["2015: Mild Asthma", "2018: High Blood Pressure", "2022: Type 2 Diabetes", "2025: Current Visit"]
        st.write(f"**Patient Sequence:** `{' -> '.join(history)}`")
        
        forget_gate = st.slider(
            "Forget Gate Retention Value", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="0.0 means the model immediately forgets all past steps. 1.0 means it retains perfect memory of the entire sequence."
        )
        
        st.write("**Model's Context Memory at 'Current Visit':**")
        retained_memory = []
        for i, event in enumerate(history[:-1]):
            strength = forget_gate ** (len(history) - 2 - i) 
            if strength > 0.1:
                retained_memory.append(f"{event} (Strength: {strength:.2f})")
                
        if not retained_memory:
            st.warning("The model has forgotten the past medical history. It is only evaluating the current visit.")
        else:
            for mem in retained_memory:
                st.success(mem)

    elif model_choice == "Transformer":
        st.subheader("Transformer")
        st.write("Transformers use an Encoder/Decoder structure and self-attention to translate input sequences to output sequences.")
        
        
        st.markdown("### Interactive Mechanics: Long-Range Self-Attention")
        st.write("Unlike RNNs that process step-by-step, self-attention looks at all data points simultaneously.")
        
        seq_length = st.slider(
            "Sequence Length (Words or Clinical Notes)", 
            min_value=10, max_value=500, value=100,
            help="The number of tokens (words or data points) in the input sequence."
        )
        
        st.write("Because every word looks at every other word, attention calculations grow quadratically (N squared):")
        interactions = seq_length * seq_length
        st.metric(
            "Total Attention Interactions Computed", 
            f"{interactions:,}", 
            help="The total number of context connections the model evaluates in a single pass."
        )
        st.info("This quadratic scaling is why Transformers excel at finding complex connections in patient histories, but require significant computational power.")

# ==========================================
# ACTIVITY 2: MODEL SELECTION SIMULATOR
# ==========================================
elif mode == "Activity 2: Model Selection Simulator":
    st.title("Activity 2: The Biomedical Matchmaker")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        **Objective:** Practice matching biomedical data types to the correct deep learning architecture.
        
        **Steps:**
        1. Read the provided biomedical data scenarios on the left.
        2. Select the scenario that best represents your current data analysis goal.
        3. Review the algorithm analysis on the right to see which model is recommended and why.
        """)
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Select Your Biomedical Data:")
        data_type = st.radio(
            "What are you analyzing?", 
            [
                "Gene expression matrix (20,000 genes) to predict cell viability",
                "Histology images to predict cancer survival time",
                "Sequential patient medical history to predict future status",
                "Complex natural language clinical notes (long-range dependencies)"
            ],
            help="Choose the data format that aligns with your research or clinical goal."
        )
        
    with col2:
        st.subheader("Algorithm Analysis")
        if "Gene expression" in data_type:
            st.success("Optimal Fit: Deep Neural Network (DNN)")
            st.write("**Why:** DNNs handle distinct covariates in the same domain. Ensure you have a large sample size to prevent overfitting on 20,000 features.")
        elif "Histology" in data_type:
            st.success("Optimal Fit: Convolutional Neural Network (CNN)")
            st.write("**Why:** CNNs use kernels and pooling to capture local spatial features efficiently in structured, grid-like image data.")
        elif "Sequential patient" in data_type:
            st.success("Optimal Fit: Recurrent Neural Network (LSTM)")
            st.write("**Why:** LSTMs are explicitly designed to maintain a hidden state that carries past information step-by-step.")
        elif "Complex natural language" in data_type:
            st.success("Optimal Fit: Transformer")
            st.write("**Why:** Transformers use self-attention to process entire sequences at once, outperforming RNNs on long-range linguistic dependencies.")

# ==========================================
# ACTIVITY 3: KNOWLEDGE CHECK
# ==========================================
elif mode == "Activity 3: Knowledge Check":
    st.title("Activity 3: Knowledge Assessment")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        **Objective:** Test your retention of the core concepts covered in the video module.
        
        **Steps:**
        1. Read each question carefully.
        2. Select the best answer from the multiple-choice options.
        3. Review the immediate feedback provided below your selection to understand why the answer is correct or incorrect.
        """)
        
    st.subheader("Question 1")
    q1 = st.radio(
        "Which of the following about DNN is true?", 
        [
            "A) Deep neural network only works for regression task.", 
            "B) Deep neural network only works for classification task.", 
            "C) Deep neural network is sensitive to high-dimensional input and overfitting.", 
            "D) The number of parameters in Deep neural network is small."
        ], 
        index=None, key="q1",
        help="Recall the Parameter Explosion interactive simulation."
    )
    if q1:
        if q1.startswith("C"):
            st.success("Correct. High dimensionality increases parameters exponentially, raising overfitting risks.")
        else:
            st.error("Incorrect. The correct answer is C.")
            
    st.markdown("---")
            
    st.subheader("Question 2")
    q2 = st.radio(
        "What is the following statements about CNN is false?", 
        [
            "A) CNN adopts dropout to reduce the parameters.", 
            "B) CNN adopts maxpooling to reduce the parameters.", 
            "C) CNN capture local feature using kernels.", 
            "D) CNN can only do classification task."
        ], 
        index=None, key="q2",
        help="Think about the types of output a neural network can produce in the output layer."
    )
    if q2:
        if q2.startswith("D"):
            st.success("Correct. This statement is false. CNNs can handle both classification and regression tasks.")
        else:
            st.error("Incorrect. The correct answer is D.")
            
    st.markdown("---")
            
    st.subheader("Question 3")
    q3 = st.radio(
        "Which of the following statements about LSTM is false?", 
        [
            "A) LSTM is good at modeling sequential data.", 
            "B) LSTM can be used for time-series forecasting.", 
            "C) LSTM can only have one output in the last time step.", 
            "D) LSTM can do both classification and regression task."
        ], 
        index=None, key="q3",
        help="Consider the 'Many-to-Many' architectural capability."
    )
    if q3:
        if q3.startswith("C"):
            st.success("Correct. This statement is false. LSTMs can do Many-to-Many tasks, producing outputs at multiple time steps.")
        else:
            st.error("Incorrect. The correct answer is C.")

    st.markdown("---")

    st.subheader("Question 4")
    q4 = st.radio(
        "If you have medical history of patients and want to predict the further medical status, which deep learning model is best to use?", 
        [
            "A) Deep neural network.", 
            "B) Recurrent neural network.", 
            "C) Convolutional neural network.", 
            "D) Transformer."
        ], 
        index=None, key="q4",
        help="Which model handles long-range sequence dependencies most effectively?"
    )
    if q4:
        if q4.startswith("D"):
            st.success("Correct. While RNNs can process sequences, Transformers are considered the best modern approach for capturing complex, long-range dependencies in patient history.")
        else:
            st.error("Incorrect. The correct answer is D.")
