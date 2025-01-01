import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import gdown
import os
import PyPDF2

# Helper function to download files from Google Drive and save to a specified location
def download_file_from_drive(file_id, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        st.write(f"Downloading {destination}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)
    else:
        st.write(f"File {destination} already exists, skipping download.")

# Load custom CSS for styling
with open("styles/style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

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

# Load job embeddings and model
@st.cache_resource(show_spinner="Loading word embedding model...")
def load_job_embeddings():
    embed_file = "models/job_embeddings.pkl"
    job_embeddings_file_id = "1nVYT_eV2nuN__8nqPK1dFdsDqW6TfvQ8"  # Replace with your file ID
    download_file_from_drive(job_embeddings_file_id, embed_file)

    if not os.path.exists(embed_file):
        st.error("Job embeddings file is missing and failed to download.")
        return None, None

    df = pd.read_pickle(embed_file)
    job_tensors = torch.tensor(df['job_embedding'].tolist(), dtype=torch.float)
    return df, job_tensors

@st.cache_resource(show_spinner="Loading recommendation model...")
def load_model():
    model_file = "models/job_recommendation_model.pth"
    model_file_id = "your_model_file_id_here"  # Replace with your file ID
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

# PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Load job embeddings and pre-trained model
df, job_tensors = load_job_embeddings()
if df is None or job_tensors is None:
    st.stop()

model = load_model()
if model is None:
    st.stop()

# Load SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("Role Recommendation System")
st.write("Provide your resume to find matching roles!")

# Input options
option = st.radio("How would you like to provide your resume?", options=["Paste Text", "Upload PDF"], horizontal=True)

resume_text = ""
if option == "Paste Text":
    resume_text = st.text_area("Paste your resume here", height=200)
elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload your resume (PDF format only)", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
                st.success("Text extracted successfully!")
            except Exception as e:
                st.error(f"Failed to extract text from PDF: {e}")

# Add a submit button
if st.button("Recommend Roles"):
    if resume_text.strip():
        with st.spinner('Generating resume embedding...'):
            resume_embedding = torch.tensor(embed_model.encode(resume_text), dtype=torch.float).unsqueeze(0)

        with st.spinner('Recommending roles...'):
            def recommend_jobs(resume_embedding, job_tensors, top_n=5):
                similarities = torch.nn.functional.cosine_similarity(resume_embedding, job_tensors)
                _, top_indices = torch.topk(similarities, top_n)
                return top_indices

            top_indices = recommend_jobs(resume_embedding, job_tensors, top_n=5)
            recommended_jobs = df.iloc[top_indices.tolist()]['Title']

            st.write("### Recommended Roles:")
            for idx, job in enumerate(recommended_jobs, 1):
                st.write(f"{idx}. {job}")
    else:
        st.warning("Please provide your resume to get recommendations!")
