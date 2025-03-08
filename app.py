import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Stable Code 3B Model (Optimized for CPU)
@st.cache_resource
def load_model():
    model_name = "stabilityai/stable-code-3b"  # ✅ Known 3B model 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Load model & tokenizer
model, tokenizer = load_model()

# Streamlit UI
st.title("Stable Code 3B Coding Assistant")
st.subheader("Generate, Debug, or Complete Code")

# User input
prompt = st.text_area("Enter your code prompt:", height=150)
max_length = st.slider("Max Output Length", 50, 512, 256)

if st.button("Generate Code"):
    if prompt.strip():
        with st.spinner("Generating code..."):
            inputs = tokenizer(prompt, return_tensors="pt")  # ✅ CPU-friendly (No .to("cuda"))
            outputs = model.generate(**inputs, max_length=max_length)
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.subheader("Generated Code:")
        st.code(generated_code, language="python")
    else:
        st.warning("Please enter a valid prompt.")
