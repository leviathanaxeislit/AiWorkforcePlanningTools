import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import gdown
import os

# Helper function to download files from Google Drive and save to a specified location
def download_file_from_drive(file_id, destination):
    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # If the file doesn't exist, download it
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
@st.cache_resource(show_spinner="loading word embedding model")
def load_job_embeddings():
    embed_file = "models/job_embeddings.pkl"
    
    # Download the job embeddings from Google Drive if not found
    job_embeddings_file_id = "1nVYT_eV2nuN__8nqPK1dFdsDqW6TfvQ8"  # Job embeddings file ID
    download_file_from_drive(job_embeddings_file_id, embed_file)

    # Check if file exists after download
    if not os.path.exists(embed_file):
        st.error("Job embeddings file is missing and failed to download.")
        return None, None

    df = pd.read_pickle(embed_file)
    job_tensors = torch.tensor(df['job_embedding'].tolist(), dtype=torch.float)
    return df, job_tensors

@st.cache_resource(show_spinner="loading Recommendation Model")
def load_model():
    model_file = "models/job_recommendation_model.pth"
    
    # Download the model file from Google Drive if not found
    model_file_id = "your_model_file_id_here"  # Replace with your model file ID
    download_file_from_drive(model_file_id, model_file)

    # Ensure the embeddings file exists before loading
    df, job_tensors = load_job_embeddings()
    if df is None or job_tensors is None:
        st.error("Failed to load job embeddings.")
        return None

    embedding_dim = len(df['job_embedding'][0])
    model = CollaborativeFiltering(embedding_dim)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

# Load job embeddings and pre-trained model
df, job_tensors = load_job_embeddings()
if df is None or job_tensors is None:
    st.stop()  # Stop the app if embeddings are not available

model = load_model()
if model is None:
    st.stop()  # Stop the app if the model is not loaded properly

# Load SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("Role Recommendation System")
st.write("Paste your resume below to find matching Roles!")

resume_text = st.text_area("Paste your resume here", height=200)

# Add a submit button
if st.button("Recommend Roles"):
    if resume_text.strip() != "":
        # Display spinner while generating resume embedding
        with st.spinner('Generating resume embedding...'):
            resume_embedding = torch.tensor(embed_model.encode(resume_text), dtype=torch.float).unsqueeze(0)

        # Display spinner while recommending jobs
        with st.spinner('Recommending Roles...'):
            def recommend_jobs(resume_embedding, job_tensors, top_n=5):
                similarities = torch.nn.functional.cosine_similarity(resume_embedding, job_tensors)
                _, top_indices = torch.topk(similarities, top_n)
                return top_indices

            # Get top job recommendations
            top_indices = recommend_jobs(resume_embedding, job_tensors, top_n=5)
            recommended_jobs = df.iloc[top_indices.tolist()]['Title']

            # Display recommended jobs
            st.write("Recommended Roles:")
            for idx, job in enumerate(recommended_jobs, 1):
                st.write(f"{idx}. {job}")
    else:
        st.warning("Please paste your resume to see recommendations!")
