import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Educational Content Generator",
    layout="wide"
)

st.title("AI Educational Content Generator")
st.subheader("For College Professors | FDP | Curriculum Design")

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
)



    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,
        do_sample=False
    )
    return generator

generator = load_model()

# -----------------------------
# Sidebar Inputs (User Context)
# -----------------------------
st.sidebar.header("Educational Context")

subject_area = st.sidebar.text_input(
    "Discipline",
    placeholder="e.g., Data Science, AI, Management"
)

topic_name = st.sidebar.text_input(
    "Course / Topic",
    placeholder="e.g., Support Vector Machines"
)

learner_level = st.sidebar.selectbox(
    "Learner Level",
    ["UG", "PG", "Diploma"]
)

academic_goal = st.sidebar.selectbox(
    "Academic Goal",
    ["Concept Clarity", "Assessment", "Engagement", "Revision"]
)

structured_output = st.sidebar.text_area(
    "Desired Output Format",
    placeholder="e.g., Lecture Title, Learning Outcomes, Examples, Questions"
)

# -----------------------------
# Generate Button
# -----------------------------
if st.button("Generate Educational Content"):
    if not all([subject_area, topic_name, structured_output]):
        st.warning("Please fill in all required fields.")
    else:
        with st.spinner("Generating classroom-ready content..."):
            prompt = f"""
User:
You are an AI Educational Content Specialist assisting college professors.

Educational Context:
- Discipline: {subject_area}
- Course / Topic: {topic_name}
- Learner Level: {learner_level}
- Academic Goal: {academic_goal}

Task:
Generate educational content aligned with the above context.

Guidelines:
- Use clear, academically appropriate language
- Align with Bloom’s Taxonomy where applicable
- Maintain factual accuracy
- Avoid unnecessary jargon
- Ensure content is classroom-ready

Output Format:
{structured_output}

Assistant:
"""
            output = generator(prompt)
            response = output[0]["generated_text"]

        st.success("Content Generated Successfully")
        st.markdown("### Generated Educational Content")
        st.write(response)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Powered by Generative AI | Designed for Higher Education")
