import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import gdown
import os
from PyPDF2 import PdfReader
import time
import urllib.parse

st.set_page_config(
    page_title="Role Recommendation Model",
    page_icon="https://www.careerguide.com/career/wp-content/uploads/2021/01/a2413959910293.5a33a9bde96e8.gif",
    initial_sidebar_state="collapsed",
)

cols = st.columns(3)  # Create 3 columns

try:
    with cols[0]:
        st.page_link(page="app.py", icon="🏠", label="Home")
    with cols[1]:
        st.page_link(
            page="pages/Promotion Model.py", icon="💹", label="Promotion Model"
        )
    with cols[2]:
        st.page_link(
            page="pages/Role Recommendation Model.py",
            icon="🏢",
            label="Role Recommendation",
        )
except Exception as e:
    st.error(f"Error loading page links: {e}")

st.markdown(
    """
            <p style="font-size: 35px; font-family: 'Gugi', serif;font-weight: 400;">ROLE RECOMMENDATION SYSTEM</p>
            """,
    unsafe_allow_html=True,
)

st.image(
    image="https://www.careerguide.com/career/wp-content/uploads/2021/01/a2413959910293.5a33a9bde96e8.gif",
    use_container_width=True,
)

# Add Font Awesome CSS and Javascript (Important!)
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" integrity="sha512-9usAa10IRO0HhonpyAIVpjrylPvoDwiPUiKdWk5t3PyolY1cOd4DSE0Ga+ri4AuTroPR5aQvXU9xC6qOPnzFeg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://kit.fontawesome.com/a838ad3310.js" crossorigin="anonymous"></script>
    """,
    unsafe_allow_html=True,
)

with open("styles/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


def download_file_from_drive(file_id, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        message_placeholder = st.empty()
        with st.spinner(f"Downloading {destination}..."):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}", destination, quiet=False
            )
        message_placeholder.info(f"Downloaded {destination} successfully.")
        time.sleep(2)
        message_placeholder.empty()
    else:
        message_placeholder = st.empty()
        message_placeholder.success(
            f"File {destination} already exists, skipping download."
        )
        time.sleep(2)
        message_placeholder.empty()


class CollaborativeFiltering(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(CollaborativeFiltering, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, resume_embedding, job_embedding):
        combined = torch.cat([resume_embedding, job_embedding], dim=1)
        interaction = self.fc(combined)
        return interaction


@st.cache_resource(show_spinner="Loading word embedding model...")
def load_job_embeddings():
    embed_file = "models/job_embeddings.pkl"
    job_embeddings_file_id = "1nVYT_eV2nuN__8nqPK1dFdsDqW6TfvQ8"
    download_file_from_drive(job_embeddings_file_id, embed_file)

    if not os.path.exists(embed_file):
        st.error("Job embeddings file is missing and failed to download.")
        return None, None

    df = pd.read_pickle(embed_file)
    job_tensors = torch.tensor(df["job_embedding"].tolist(), dtype=torch.float)
    return df, job_tensors


@st.cache_resource(show_spinner="Loading recommendation model...")
def load_model():
    model_file = "models/job_recommendation_model.pth"
    model_file_id = "your_model_file_id_here"
    download_file_from_drive(model_file_id, model_file)

    df, job_tensors = load_job_embeddings()
    if df is None or job_tensors is None:
        st.error("Failed to load job embeddings.")
        return None

    embedding_dim = len(df["job_embedding"][0])
    model = CollaborativeFiltering(embedding_dim)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model


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


def extract_skills(resume_text, skill_keywords):
    found_skills = set()
    for skill in skill_keywords:
        if skill.lower() in resume_text.lower():
            found_skills.add(skill)
    return sorted(found_skills)


@st.cache_resource(show_spinner="Loading skill keywords...")
def load_skill_keywords():
    skill_keywords_file = "models/skill_keywords.txt"
    with open(skill_keywords_file, "r") as file:
        return [line.strip() for line in file.readlines()]


def recommend_jobs(resume_embedding, job_tensors, df, top_n=5):
    similarities = torch.nn.functional.cosine_similarity(resume_embedding, job_tensors)
    _, top_indices = torch.topk(similarities, top_n)
    recommended_jobs = []
    for index in top_indices:
        job_index = index.item()
        job_title = df.iloc[job_index]["Title"]
        score = similarities[index].item() * 100
        recommended_jobs.append((job_title, score))
    return recommended_jobs


# --- New Function to Generate LinkedIn Job Search URL ---
def generate_linkedin_job_search_url(job_title):
    """Generates a LinkedIn job search URL for a specific job title."""
    keywords = job_title  # Only job title for LinkedIn
    encoded_keywords = urllib.parse.quote_plus(keywords)
    linkedin_url = f"https://www.linkedin.com/jobs/search/?keywords={encoded_keywords}&location=India"
    return linkedin_url


# --- New Function to Generate Naukri Job Search URL ---
def generate_naukri_job_search_url(job_title):
    """Generates a Naukri job search URL for a specific job title."""
    keywords = job_title  # Using only the job title for Naukri search
    encoded_keywords = urllib.parse.quote_plus(keywords)
    naukri_url = f"https://www.naukri.com/{encoded_keywords}-jobs?k={encoded_keywords}&nignbevent_src=jobsearchDeskGNB"
    return naukri_url


# Load necessary components BEFORE tabs
df, job_tensors = load_job_embeddings()
if df is None or job_tensors is None:
    st.stop()
model = load_model()
if model is None:
    st.stop()
skill_keywords = load_skill_keywords()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

tabs = st.tabs(["Resume Upload", "Job Description Scoring"])

resume_text = None

with tabs[0]:
    st.write("Upload your resume (PDF) to find matching roles!")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)", type=["pdf"], key="resume_upload_tab"
    )
    if uploaded_file is not None:
        resume_text = parse_pdf(uploaded_file)
        if resume_text:
            st.markdown("### Extracted Resume Text")
            st.markdown(f"```text\n{resume_text}\n```")
            skills = extract_skills(resume_text, skill_keywords)  # skills is used to extract keywords
            st.markdown("### Extracted Skills")
            if skills:
                st.markdown(", ".join(skills))
            else:
                st.markdown("No skills matched from the predefined list.")

            if st.button("Recommend Roles"):
                if resume_text is not None and resume_text.strip() != "":
                    with st.spinner("Generating resume embedding..."):
                        resume_embedding = torch.tensor(
                            embed_model.encode(resume_text), dtype=torch.float
                        ).unsqueeze(0)
                    with st.spinner("Recommending Roles..."):
                        recommended_jobs = recommend_jobs(
                            resume_embedding, job_tensors, df, top_n=5
                        )
                        st.success("Recommended Roles:")
                        for i, (job_title, score) in enumerate(recommended_jobs):
                            linkedin_url = generate_linkedin_job_search_url(job_title)
                            naukri_url = generate_naukri_job_search_url(job_title)

                            st.write(f"**{job_title}** - Suitability Score: {score:.1f}")

                            col1, col2 = st.columns(2)  # Create 2 columns
                            with col1:
                                linkedin_button = f"""
                                    <a href="{linkedin_url}" target="_blank">
                                        <button style="
                                            background-color: #0077B5; /* LinkedIn Blue */
                                            border: none;
                                            color: white;
                                            padding: 10px 20px;
                                            text-align: center;
                                            text-decoration: none;
                                            display: inline-block;
                                            font-size: 14px;
                                            cursor: pointer;
                                            border-radius: 5px;
                                        "><i class="fab fa-linkedin"></i> LinkedIn Jobs
                                        </button>
                                    </a>
                                    """
                                st.markdown(linkedin_button, unsafe_allow_html=True)

                            with col2:
                                naukri_button = f"""
                                    <a href="{naukri_url}" target="_blank">
                                        <button style="
                                            background-color: #E07A5F; /* Naukri Orange/Red */
                                            border: none;
                                            color: white;
                                            padding: 10px 20px;
                                            text-align: center;
                                            text-decoration: none;
                                            display: inline-block;
                                            font-size: 14px;
                                            cursor: pointer;
                                            border-radius: 5px;
                                        "><i class="fas fa-building"></i> Naukri Jobs
                                        </button>
                                    </a>
                                    """
                                st.markdown(naukri_button, unsafe_allow_html=True)

                else:
                    st.warning("Please upload a valid PDF resume.")
        else:
            st.warning("Could not parse the uploaded PDF.")
    else:
        st.warning("Please upload a valid PDF resume.")

with tabs[1]:  # Job Description Scoring Tab
    st.write("Enter a job description and use the resume uploaded above to score it.")
    if resume_text:  # Check if resume has been uploaded
        job_description = st.text_area("Enter Job Description:", height=200)
        if st.button("Score Resume"):
            if job_description.strip():
                with st.spinner("Scoring..."):
                    resume_embedding = torch.tensor(
                        embed_model.encode(resume_text), dtype=torch.float
                    ).unsqueeze(0)
                    job_description_embedding = torch.tensor(
                        embed_model.encode(job_description), dtype=torch.float
                    ).unsqueeze(0)
                    similarity_score = (
                        torch.nn.functional.cosine_similarity(
                            resume_embedding, job_description_embedding
                        ).item()
                        * 100
                    )
                    st.success(
                        f"Resume Suitability Score for this job description: {similarity_score:.1f}"
                    )
            else:
                st.warning("Please enter a job description.")
    else:
        st.warning("Please upload a resume on the 'Resume Upload' tab first.")