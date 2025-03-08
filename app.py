import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_API"

model_name = "codellama/CodeLlama-7b-Instruct-hf"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        device_map="auto",
        use_auth_token=True
    )
    return tokenizer, model

tokenizer, model = load_model()

def create_prompt(language, task_prompt):
    """
    Constructs a detailed prompt for CodeLlama to generate complex code.
    It includes a base instruction and few-shot examples for clarity.
    """
    base_prompt = (
        f"Generate a complete, well-structured, and clean {language} program for the following task:\n"
        f"{task_prompt}\n\n"
        "Requirements:\n"
        "- Follow best practices and use a modular approach.\n"
        "- Include detailed inline comments explaining each major section and function.\n"
        "- Provide a clear explanation at the end of the code about how the program works.\n\n"
    )
    
    few_shot_examples = ""
    if language.strip().lower() == "python":
         few_shot_examples = (
             "# Example (Python):\n"
             "# Task: Create a function that adds two numbers\n"
             "def add(a, b):\n"
             "    # Returns the sum of a and b\n"
             "    return a + b\n\n"
             "# Explanation:\n"
             "# This function accepts two parameters and returns their sum.\n\n"
         )
    elif language.strip().lower() == "java":
         few_shot_examples = (
             "// Example (Java):\n"
             "// Task: Create a method to check if a number is even\n"
             "public class NumberUtils {\n"
             "    // Returns true if the number is even\n"
             "    public static boolean isEven(int number) {\n"
             "        return number % 2 == 0;\n"
             "    }\n"
             "}\n\n"
             "// Explanation:\n"
             "// The isEven method checks if a number is divisible by 2.\n\n"
         )
    return few_shot_examples + base_prompt

def generate_code(prompt_text, max_length=512, temperature=0.7, top_p=0.95):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

st.title("Kaggle CodeGenie Prototype Front End")
st.write("This is a front end for the CodeLlama-powered Kaggle CodeGenie Prototype. Provide the programming language and a detailed task description, then click 'Generate Code'.")

language = st.text_input("Programming Language", value="Python")
task_prompt = st.text_area("Task Description", height=150, value="Create a function that calculates the factorial of a number.")

if st.button("Generate Code"):
    if language and task_prompt:
        with st.spinner("Generating code..."):
            prompt_text = create_prompt(language, task_prompt)
            generated_code = generate_code(prompt_text)
        st.code(generated_code, language=language.lower())
    else:
        st.error("Please provide both a programming language and a task description.")
