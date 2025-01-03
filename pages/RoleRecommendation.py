import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import gdown
import os
from PyPDF2 import PdfReader

# Helper function to download files from Google Drive and save to a specified location
def download_file_from_drive(file_id, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {destination}..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)
            st.info(f"Downloaded {destination} successfully.")
    else:
        st.success(f"File {destination} already exists, skipping download.")

# Define the collaborative filtering model
class CollaborativeFiltering(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(CollaborativeFiltering, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, resume_embedding, job_embedding):
        combined = torch.cat([resume_embedding, job_embedding], dim=1)
        interaction = self.fc(combined)
        return interaction

# Load job embeddings
@st.cache_resource(show_spinner="Loading word embedding model...")
def load_job_embeddings():
    embed_file = "models/job_embeddings.pkl"
    job_embeddings_file_id = "1nVYT_eV2nuN__8nqPK1dFdsDqW6TfvQ8"
    download_file_from_drive(job_embeddings_file_id, embed_file)

    if not os.path.exists(embed_file):
        st.error("Job embeddings file is missing and failed to download.")
        return None, None

    df = pd.read_pickle(embed_file)
    job_tensors = torch.tensor(df['job_embedding'].tolist(), dtype=torch.float)
    return df, job_tensors

# Load the recommendation model
@st.cache_resource(show_spinner="Loading recommendation model...")
def load_model():
    model_file = "models/job_recommendation_model.pth"
    model_file_id = "your_model_file_id_here"  # Replace with your model file ID
    download_file_from_drive(model_file_id, model_file)

    df, job_tensors = load_job_embeddings()
    if df is None or job_tensors is None:
        st.error("Failed to load job embeddings.")
        return None

    embedding_dim = len(df['job_embedding'][0])
    model = CollaborativeFiltering(embedding_dim)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

# Parse the uploaded PDF resume
def parse_pdf(file):
    try:
        with st.spinner("Parsing PDF..."):
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            st.success("PDF parsed successfully.")
            return text.strip()
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return None

# Extract unique skills from the resume text
def extract_skills(resume_text, skill_keywords):
    found_skills = set()  # Use a set to handle duplicates
    for skill in skill_keywords:
        if skill.lower() in resume_text.lower():
            found_skills.add(skill)  # Add skills to the set
    return sorted(found_skills)  # Return a sorted list of unique skills

# Load skill keywords
@st.cache_resource(show_spinner="Loading skill keywords...")
def load_skill_keywords():
    skill_keywords_file = "models/skill_keywords.txt"
    # skill_keywords_file_id = "your_skill_keywords_file_id_here"  # Replace with your file ID
    # download_file_from_drive(skill_keywords_file_id, skill_keywords_file)

    with open(skill_keywords_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Load job embeddings and recommendation model
with st.spinner("Loading job embeddings..."):
    df, job_tensors = load_job_embeddings()
    if df is None or job_tensors is None:
        st.stop()

with st.spinner("Loading recommendation model..."):
    model = load_model()
    if model is None:
        st.stop()

# Load SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("Role Recommendation System")
st.write("Upload your resume (PDF) or paste your resume below to find matching Roles!")

# Load skill keywords
skill_keywords = load_skill_keywords()

# Resume input (Upload or text area)
pasted_text = st.text_area("Or paste your resume text below:", height=200)
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

resume_text = None

if uploaded_file is not None:
    resume_text = parse_pdf(uploaded_file)
    if resume_text:
        st.markdown("### Extracted Resume Text")
        st.markdown(f"```text\n{resume_text}\n```")
elif pasted_text.strip():
    resume_text = pasted_text
    st.markdown("### Entered Resume Text")
    st.markdown(f"```text\n{resume_text}\n```")

# Extract and display unique skills
if resume_text:
    skills = extract_skills(resume_text, skill_keywords)
    st.markdown("### Extracted Skills")
    if skills:
        st.markdown(", ".join(skills))
    else:
        st.markdown("No skills matched from the predefined list.")

# Progress bar example (optional for embedding generation)
progress = st.progress(0)

if st.button("Recommend Roles"):
    if resume_text is not None and resume_text.strip() != "":
        progress.progress(10)
        with st.spinner('Generating resume embedding...'):
            resume_embedding = torch.tensor(embed_model.encode(resume_text), dtype=torch.float).unsqueeze(0)
        progress.progress(50)

        with st.spinner('Recommending Roles...'):
            def recommend_jobs(resume_embedding, job_tensors, top_n=5):
                similarities = torch.nn.functional.cosine_similarity(resume_embedding, job_tensors)
                _, top_indices = torch.topk(similarities, top_n)
                return top_indices

            # Get top job recommendations
            top_indices = recommend_jobs(resume_embedding, job_tensors, top_n=5)
            recommended_jobs = df.iloc[top_indices.tolist()]['Title']
        progress.progress(100)

        # Display recommended jobs
        st.success("Recommended Roles:")
        for idx, job in enumerate(recommended_jobs, 1):
            st.markdown(f"**{idx}. {job}**")
    else:
        st.warning("Please upload a valid PDF or paste your resume to see recommendations!")
