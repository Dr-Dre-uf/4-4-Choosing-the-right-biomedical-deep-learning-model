import streamlit as st
import pandas as pd
import numpy as np
import time

# --- PAGE CONFIG & SESSION STATE ---
st.set_page_config(page_title="Biomedical DL Models", layout="wide")

# Initialize session state for gamification tracking
if 'matchmaker_score' not in st.session_state:
    st.session_state.matchmaker_score = 0
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0

# --- SIDEBAR NAVIGATION & SETTINGS ---
st.sidebar.title("Module Settings")
scientific_context = st.sidebar.radio(
    "Select Learning Context:",
    ["Clinical (Patient Care)", "Foundational (Basic Science)"],
    help="Toggle the terminology and examples to match your specific field of study."
)
st.sidebar.markdown("---")

st.sidebar.title("DL Learning Module")
mode = st.sidebar.radio(
    "Select a Mission:",
    [
        "Mission 1: The Architecture Lab", 
        "Mission 2: The Diagnostic Challenge", 
        "Mission 3: Final Certification"
    ],
    help="Complete each mission sequentially to master deep learning model selection."
)

# Progress Tracker
total_score = st.session_state.matchmaker_score + st.session_state.quiz_score
st.sidebar.markdown("---")
st.sidebar.subheader("Your Progress")
st.sidebar.progress(min(total_score / 8.0, 1.0)) # 4 points matchmaker, 4 points quiz

# ==========================================
# MISSION 1: THE ARCHITECTURE LAB
# ==========================================
if mode == "Mission 1: The Architecture Lab":
    st.title("Mission 1: The Architecture Lab")
    
    with st.expander("Mission Briefing", expanded=True):
        st.write("""
        **Objective:** You are an engineer tasked with optimizing different deep learning models. 
        
        **Your Orders:**
        1. Select an architecture below.
        2. Read the structural breakdown.
        3. Complete the interactive 'Lab Task' to prove you understand the model's mechanics.
        """)
        
    model_choice = st.selectbox(
        "Select an Architecture to Engineer:",
        ["Deep Neural Network (DNN)", "Convolutional Neural Network (CNN)", "Recurrent Neural Network (LSTM)", "Transformer"],
        help="Load the model schematic into the lab."
    )
    
    st.markdown("---")
    
    if model_choice == "Deep Neural Network (DNN)":
        st.subheader("Deep Neural Network (DNN)")
        st.write("DNNs are the foundation, but they are sensitive to high-dimensional inputs.")
        
        
        st.markdown("### Lab Task: Prevent Overfitting")
        if scientific_context == "Clinical (Patient Care)":
            st.write("**Scenario:** You are using 10,000 patient covariates. **Task:** Adjust the network depth and width so the total parameter count stays below 1,000,000 to prevent overfitting on your limited patient cohort.")
            feature_label = "Patient Covariates"
        else:
            st.write("**Scenario:** You are processing 10,000 genes. **Task:** Adjust the network depth and width so the total parameter count stays below 1,000,000 to prevent overfitting on your limited cell samples.")
            feature_label = "Sequenced Genes"
            
        col1, col2 = st.columns(2)
        with col1:
            features = st.slider(feature_label, min_value=1000, max_value=20000, value=10000, step=1000, disabled=True)
            hidden_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=10, value=5)
            neurons_per_layer = st.slider("Neurons per Hidden Layer", min_value=16, max_value=512, value=256, step=16)
        
        with col2:
            first_layer_params = (features * neurons_per_layer) + neurons_per_layer
            hidden_params = (hidden_layers - 1) * ((neurons_per_layer * neurons_per_layer) + neurons_per_layer)
            total_params = first_layer_params + hidden_params + (neurons_per_layer * 1) + 1 
            
            st.metric("Total Model Parameters", f"{total_params:,}", delta=f"{1000000 - total_params:,} from limit", delta_color="inverse")
            if total_params <= 1000000:
                st.success("Lab Task Complete! You have successfully engineered a lightweight DNN.")
            else:
                st.error("Warning: Parameter count exceeds limits. The model is highly likely to overfit. Reduce layers or neurons.")

    elif model_choice == "Convolutional Neural Network (CNN)":
        st.subheader("Convolutional Neural Network (CNN)")
        st.write("CNNs handle grid-like data efficiently using kernels and pooling.")
        
        
        st.markdown("### Lab Task: Data Compression")
        if scientific_context == "Clinical (Patient Care)":
            st.write("**Scenario:** Your hospital servers are overloaded with massive digital pathology slides. **Task:** Apply pooling layers to compress the spatial dimensions by at least 90% before it hits the dense layer.")
        else:
            st.write("**Scenario:** Your lab's GPU cluster is struggling with high-res microscopy arrays. **Task:** Apply pooling layers to compress the spatial dimensions by at least 90% before it hits the dense layer.")
            
        col1, col2, col3 = st.columns(3)
        with col1:
            img_size = st.select_slider("Input Resolution (Pixels)", options=[256, 512, 1024, 2048], value=1024)
            st.info(f"Input Matrix: {img_size} x {img_size}")
            
        with col2:
            apply_pool1 = st.checkbox("Apply Max Pooling Layer 1 (2x2)", value=False)
            apply_pool2 = st.checkbox("Apply Max Pooling Layer 2 (2x2)", value=False)
            apply_pool3 = st.checkbox("Apply Max Pooling Layer 3 (2x2)", value=False)
            
        with col3:
            current_size = img_size
            if apply_pool1: current_size = current_size // 2
            if apply_pool2: current_size = current_size // 2
            if apply_pool3: current_size = current_size // 2
            reduction = 100 - ((current_size**2) / (img_size**2) * 100)
            
            st.metric("Output Dimension", f"{current_size} x {current_size}")
            st.metric("Data Compressed By", f"{reduction:.1f}%")
            
            if reduction >= 90.0:
                st.success("Lab Task Complete! You extracted the hierarchical features while saving massive computational resources.")
            else:
                st.warning("Keep compressing. Apply more pooling layers to reach 90%.")

    elif model_choice == "Recurrent Neural Network (LSTM)":
        st.subheader("Recurrent Neural Network (LSTM)")
        st.write("LSTMs introduce gates to control how much past information is retained for future prediction.")
        
        
        st.markdown("### Lab Task: Tune the Memory Gate")
        if scientific_context == "Clinical (Patient Care)":
            st.write("**Task:** Tune the Forget Gate so the model remembers the '2015: Mild Asthma' diagnosis with a strength greater than 0.20 when evaluating the Current Visit.")
            history = ["2015: Mild Asthma", "2018: High BP", "2022: Type 2 Diabetes", "2025: Current Visit"]
            target_event = "2015: Mild Asthma"
        else:
            st.write("**Task:** Tune the Forget Gate so the model remembers the 't=0: Compound Added' event with a strength greater than 0.20 when evaluating the final state.")
            history = ["t=0: Compound Added", "t=10m: Receptor Binding", "t=30m: Calcium Influx", "t=60m: Final State"]
            target_event = "t=0: Compound Added"
            
        st.write(f"**Sequence Timeline:** `{' -> '.join(history)}`")
        
        forget_gate = st.slider("Forget Gate Retention Factor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
        strength_of_target = forget_gate ** (len(history) - 2)
        st.metric(f"Memory Strength of '{target_event}'", f"{strength_of_target:.2f}")
        
        if strength_of_target >= 0.20:
            st.success("Lab Task Complete! The LSTM has successfully carried critical distant information through the sequence to inform the current prediction.")
        else:
            st.error("The model has forgotten the earliest event. Increase the retention factor.")

    elif model_choice == "Transformer":
        st.subheader("Transformer")
        st.write("Transformers use an Encoder/Decoder structure and self-attention to translate sequences.")
        
        
        st.markdown("### Lab Task: Manage Server Load")
        if scientific_context == "Clinical (Patient Care)":
            st.write("**Scenario:** You are applying NLP to 10 years of clinical notes. **Task:** Transformers compute quadratically. Find the maximum sequence length your server can handle before surpassing 150,000 attention interactions.")
        else:
            st.write("**Scenario:** You are modeling a large protein structure. **Task:** Transformers compute quadratically. Find the maximum sequence length your server can handle before surpassing 150,000 attention interactions.")
            
        seq_length = st.slider("Sequence Length (Tokens)", min_value=10, max_value=800, value=100)
        interactions = seq_length * seq_length
        
        st.metric("Simultaneous Attention Interactions", f"{interactions:,}")
        
        if interactions > 150000:
            st.error("SERVER OVERLOAD. The quadratic scaling of self-attention has exceeded processing capacity.")
        elif interactions >= 100000 and interactions <= 150000:
            st.success("Lab Task Complete! You optimized the sequence window right up to the computational threshold.")
        else:
            st.warning("Server underutilized. You can process longer sequences.")

# ==========================================
# MISSION 2: THE DIAGNOSTIC CHALLENGE
# ==========================================
elif mode == "Mission 2: The Diagnostic Challenge":
    st.title("Mission 2: The Diagnostic Challenge")
    
    with st.expander("Mission Briefing", expanded=True):
        st.write("""
        **Objective:** Prove your ability to assign the correct deep learning model to real-world data scenarios.
        
        **Your Orders:**
        1. Read the four data cases below.
        2. Assign the optimal model architecture to each case.
        3. Submit your configuration to see your score.
        """)
        
    if scientific_context == "Clinical (Patient Care)":
        cases = [
            "Case Alpha: Tabular dataset of 5,000 patients with 50 lab test covariates.",
            "Case Beta: Database of 100,000 high-resolution digital pathology images.",
            "Case Gamma: Step-by-step sequential records of patient vital signs over 48 hours.",
            "Case Delta: 50-page unstructured, free-text clinical discharge summaries."
        ]
    else:
        cases = [
            "Case Alpha: High-dimensional matrix of 20,000 gene expression levels.",
            "Case Beta: Database of 100,000 fluorescent microscopy cellular arrays.",
            "Case Gamma: Step-by-step time-series recordings of single-cell action potentials.",
            "Case Delta: Massive, complex genomic DNA sequences requiring long-range context."
        ]

    answers = ["Deep Neural Network (DNN)", "Convolutional Neural Network (CNN)", "Recurrent Neural Network (LSTM)", "Transformer"]
    options = ["Select a Model..."] + answers
    
    with st.form("matchmaker_form"):
        st.subheader("Assign Models to Cases")
        c1 = st.selectbox(cases[0], options)
        c2 = st.selectbox(cases[1], options)
        c3 = st.selectbox(cases[2], options)
        c4 = st.selectbox(cases[3], options)
        
        submit_match = st.form_submit_button("Run Diagnostics")
        
        if submit_match:
            score = 0
            with st.spinner("Analyzing your configurations..."):
                time.sleep(1) # Adds gamified suspense
                
            if c1 == answers[0]: score += 1
            if c2 == answers[1]: score += 1
            if c3 == answers[2]: score += 1
            if c4 == answers[3]: score += 1
            
            st.session_state.matchmaker_score = score
            
            st.metric("Diagnostic Score", f"{score} / 4")
            if score == 4:
                st.success("Perfect Configuration! You have cleared Mission 2.")
            else:
                st.error("Sub-optimal configurations detected. Review the data structures and try again.")

# ==========================================
# MISSION 3: FINAL CERTIFICATION
# ==========================================
elif mode == "Mission 3: Final Certification":
    st.title("Mission 3: Final Certification")
    
    with st.expander("Mission Briefing", expanded=True):
        st.write("""
        **Objective:** Pass the final knowledge assessment to complete the module.
        
        **Your Orders:**
        1. Answer all four theoretical questions.
        2. Submit your exam for final grading.
        """)
        
    with st.form("certification_form"):
        q1 = st.radio(
            "1. Which of the following about DNN is true?", 
            [
                "A) Deep neural network only works for regression task.", 
                "B) Deep neural network only works for classification task.", 
                "C) Deep neural network is sensitive to high-dimensional input and overfitting.", 
                "D) The number of parameters in Deep neural network is small."
            ], index=None
        )
        st.markdown("---")
        
        q2 = st.radio(
            "2. What is the following statements about CNN is false?", 
            [
                "A) CNN adopts dropout to reduce the parameters.", 
                "B) CNN adopts maxpooling to reduce the parameters.", 
                "C) CNN capture local feature using kernels.", 
                "D) CNN can only do classification task."
            ], index=None
        )
        st.markdown("---")
        
        q3 = st.radio(
            "3. Which of the following statements about LSTM is false?", 
            [
                "A) LSTM is good at modeling sequential data.", 
                "B) LSTM can be used for time-series forecasting.", 
                "C) LSTM can only have one output in the last time step.", 
                "D) LSTM can do both classification and regression task."
            ], index=None
        )
        st.markdown("---")
        
        q4_text = "4. If you have medical history of patients and want to predict the further medical status, which deep learning model is best to use?" if scientific_context == "Clinical (Patient Care)" else "4. If you have a massive, complex biological sequence and want to model its long-range dependencies, which deep learning model is best to use?"
        q4 = st.radio(
            q4_text, 
            [
                "A) Deep neural network.", 
                "B) Recurrent neural network.", 
                "C) Convolutional neural network.", 
                "D) Transformer."
            ], index=None
        )
        
        submit_quiz = st.form_submit_button("Submit Certification")
        
        if submit_quiz:
            if not all([q1, q2, q3, q4]):
                st.warning("Please answer all questions before submitting.")
            else:
                score = 0
                if q1.startswith("C"): score += 1
                if q2.startswith("D"): score += 1
                if q3.startswith("C"): score += 1
                if q4.startswith("D"): score += 1
                
                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True
                
                st.metric("Final Grade", f"{(score/4)*100:.0f}%")
                if score == 4:
                    st.success("Certification Granted. You are ready to deploy biomedical deep learning models.")
                else:
                    st.error("Certification Denied. Review the module and try again.")
