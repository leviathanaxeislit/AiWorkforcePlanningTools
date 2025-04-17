import streamlit as st
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import gdown
import os
from PyPDF2 import PdfReader
import urllib.parse
import google.generativeai as genai
from docx import Document
from io import BytesIO
import docx2txt
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import json

# --- App Configuration ---
st.set_page_config(
    page_title="Role Recommendation Model",
    page_icon="https://www.careerguide.com/career/wp-content/uploads/2021/01/a2413959910293.5a33a9bde96e8.gif",
    initial_sidebar_state="collapsed",
)

# --- CSS Styling ---
with open("styles/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# --- Font Awesome Icons ---
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" integrity="sha512-9usAa10IRO0HhonpyAIVpjrylPvoDwiPUiKdWk5t3PyolY1cOd4DSE0Ga+ri4AuTroPR5aQvXU9xC6qOPnzFeg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://kit.fontawesome.com/a838ad3310.js" crossorigin="anonymous"></script>
    """,
    unsafe_allow_html=True,
)


# --- Global Utils ---
def initialize_models():
    """Initialize all necessary models and resources."""
    try:
        # Initialize embedding model
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize job data
        df, job_tensors = load_job_embeddings()

        # Initialize neural model
        model = load_model()

        # Initialize skill keywords
        skill_keywords = load_skill_keywords()

        # Initialize LLM
        llm_model = initialize_llm()

        # Store the LLM model in the session state for access across functions
        st.session_state["llm_model"] = llm_model

        return embed_model, df, job_tensors, model, skill_keywords, llm_model
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        st.stop()


def load_job_embeddings():
    """Load job embeddings from disk or download from drive."""
    embed_file = "models/job_embeddings.pkl"
    job_embeddings_file_id = "1nVYT_eV2nuN__8nqPK1dFdsDqW6TfvQ8"
    download_file_from_drive(job_embeddings_file_id, embed_file)

    if not os.path.exists(embed_file):
        st.error("Job embeddings file is missing and failed to download.")
        return None, None

    df = pd.read_pickle(embed_file)
    job_tensors = torch.tensor(df["job_embedding"].tolist(), dtype=torch.float)
    return df, job_tensors


def load_model():
    """Load the recommendation model from disk or download from drive."""
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


class CollaborativeFiltering(torch.nn.Module):
    """Neural network for job recommendation."""

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


def load_skill_keywords():
    """Load skill keywords from file."""
    skill_keywords_file = "models/skill_keywords.txt"
    with open(skill_keywords_file, "r") as file:
        return [line.strip() for line in file.readlines()]


def download_file_from_drive(file_id, destination):
    """Download file from Google Drive."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {destination}..."):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}", destination, quiet=False
            )


def initialize_llm():
    """Initialize the LLM with API key."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None


# --- Resume Parsing Functions ---
def parse_resume(file):
    """Parse resume from various file formats."""
    try:
        file_extension = file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            return parse_pdf(file)
        elif file_extension == "docx":
            return docx2txt.process(file)
        elif file_extension == "txt":
            return file.getvalue().decode("utf-8")
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None


def parse_pdf(file):
    """Parse text from PDF file."""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return None


# --- Analysis Functions ---
def extract_skills(resume_text, skill_keywords):
    """Extract skills from resume text based on predefined keywords."""
    found_skills = set()
    for skill in skill_keywords:
        if skill.lower() in resume_text.lower():
            found_skills.add(skill)
    return sorted(found_skills)


def extract_keywords(text, min_length=4):
    """Extract potential keywords from text."""
    # Remove special characters and convert to lowercase
    text = re.sub(r"[^\w\s]", " ", text.lower())

    # Split text into words
    words = text.split()

    # Remove short words and common stopwords
    stopwords = {
        "the",
        "and",
        "to",
        "of",
        "a",
        "in",
        "for",
        "is",
        "on",
        "that",
        "with",
        "as",
        "an",
    }
    keywords = [
        word for word in words if len(word) >= min_length and word not in stopwords
    ]

    # Return unique keywords
    return list(set(keywords))


def calculate_ats_score(resume_text, job_description):
    """Calculate ATS compatibility score based on keyword matching."""
    # Extract keywords from job description
    job_keywords = extract_keywords(job_description)

    # Check how many keywords appear in the resume
    matched_keywords = []
    for keyword in job_keywords:
        if keyword.lower() in resume_text.lower():
            matched_keywords.append(keyword)

    # Calculate score (percentage of keywords matched)
    if job_keywords:
        ats_score = (len(matched_keywords) / len(job_keywords)) * 100
    else:
        ats_score = 0

    return (
        ats_score,
        matched_keywords,
        [k for k in job_keywords if k not in matched_keywords],
    )


def analyze_resume_job_fit(resume_text, job_description):
    """Calculate comprehensive job fit metrics."""
    prompt = f"""
    Resume: {resume_text[:2000]}...
    Job Description: {job_description[:2000]}...

    Analyze how well this resume matches the job description. Return ONLY a JSON object with these exact fields:
    1. technical_skills_match: percentage (0-100)
    2. soft_skills_match: percentage (0-100) 
    3. experience_match: percentage (0-100)
    4. education_match: percentage (0-100)
    5. overall_match: percentage (0-100)

    Example response format:
    {{
        "technical_skills_match": 75,
        "soft_skills_match": 60,
        "experience_match": 85,
        "education_match": 70,
        "overall_match": 72.5
    }}
    """

    try:
        response = generate_content_with_llm(
            prompt, model=st.session_state.get("llm_model")
        )

        # Show raw output for debugging
        # st.write("Raw LLM response:", response)

        # Extract first JSON object from response using regex
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response.")

    except Exception as e:
        st.error(f"Error analyzing job fit: {e}")
        return {
            "technical_skills_match": 50,
            "soft_skills_match": 50,
            "experience_match": 50,
            "education_match": 50,
            "overall_match": 50,
        }


def recommend_jobs(resume_embedding, job_tensors, df, top_n=5):
    """Recommend jobs based on resume embedding similarity."""
    similarities = torch.nn.functional.cosine_similarity(resume_embedding, job_tensors)
    _, top_indices = torch.topk(similarities, top_n)
    recommended_jobs = []
    for index in top_indices:
        job_index = index.item()
        job_title = df.iloc[job_index]["Title"]
        score = similarities[index].item() * 100
        recommended_jobs.append((job_title, score))
    return recommended_jobs


# --- Content Generation Functions ---
def generate_content_with_llm(prompt, max_retries=3, model=None):
    """Generate content using LLM with retry mechanism."""
    # Use model parameter if provided, otherwise try to get the global one
    if model is None:
        # Try to access the global model
        try:
            model = st.session_state.get("llm_model")
            if model is None:
                return "LLM not initialized. Please check your API key configuration."
        except:
            return "LLM not initialized. Please check your API key configuration."

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Failed to generate content: {str(e)}"
            # Wait briefly before retrying
            import time

            time.sleep(1)


def generate_interview_prep(job_title, resume_text):
    """Generate interview preparation content."""
    prompt = f"""
    Based on the job title '{job_title}' and this resume:
    {resume_text[:1500]}...
    
    Create a comprehensive interview preparation guide with:
    1. 5-7 technical questions specific to the role
    2. 3-5 behavioral questions
    3. 2-3 questions the candidate should ask the interviewer
    4. Brief suggested answers for each question based on the candidate's background
    
    Format this as a structured guide with clear sections and bullet points.
    """

    return generate_content_with_llm(prompt, model=st.session_state.get("llm_model"))


def generate_cover_letter(resume_text, job_description):
    """Generate cover letter tailored to the job."""
    prompt = f"""
    Based on this resume:
    {resume_text[:1500]}...
    
    And this job description:
    {job_description[:1500]}...
    
    Create a professional, tailored cover letter that:
    1. Addresses key requirements from the job description
    2. Highlights relevant experience and skills from the resume
    3. Shows enthusiasm for the role and company
    4. Maintains a professional tone
    5. Follows standard cover letter format with intro, body, and closing
    
    The cover letter should be approximately 300-400 words.
    """

    return generate_content_with_llm(prompt, model=st.session_state.get("llm_model"))


def generate_resume_improvements(resume_text, job_description):
    """Generate suggestions to improve resume for the job."""
    prompt = f"""
    Based on this resume:
    {resume_text[:1500]}...
    
    And this job description:
    {job_description[:1500]}...
    
    Provide 3-5 specific, actionable improvements to better align this resume with the job description.
    Focus on missing keywords, skills gaps, and formatting suggestions.
    Format as a bulleted list of improvements with brief explanations.
    """

    return generate_content_with_llm(prompt, model=st.session_state.get("llm_model"))


def generate_career_path(job_title):
    """Generate career progression path for the job."""
    prompt = f"""
    For a professional starting in a {job_title} position, outline a clear 5-year career progression path.
    Include:
    1. Potential next roles (2-3 options)
    2. Key skills to develop for advancement
    3. Typical timeframes for progression
    4. Alternative specialization paths
    
    Format as a concise, step-by-step progression plan with timeline markers.
    """

    return generate_content_with_llm(prompt, model=st.session_state.get("llm_model"))


def generate_salary_estimate(job_title, skills):
    """Generate salary estimate for the job."""
    skills_str = ", ".join(skills[:10]) if skills else "general skills"

    prompt = f"""
    Provide a salary estimate for a {job_title} role in India with these skills: {skills_str}.
    Include:
    1. Salary range for entry-level (0-2 years experience)
    2. Salary range for mid-level (3-5 years experience)
    3. Salary range for senior-level (6+ years experience)
    
    Format as ranges in both monthly (₹XX,XXX - ₹YY,YYY) and annual amounts (₹X,XX,XXX - ₹Y,YY,YYY).
    """

    return generate_content_with_llm(prompt, model=st.session_state.get("llm_model"))


def generate_job_market_insights(job_title):
    """Generate market insights for the job."""
    prompt = f"""
    Provide a brief analysis of the current job market in India for {job_title} positions.
    Include:
    1. Current demand level
    2. Key industries hiring for this role
    3. Top skills in demand
    4. Growth outlook for next 2-3 years
    5. Major companies hiring
    
    Format as a concise, informative overview in 2-3 paragraphs.
    """

    return generate_content_with_llm(prompt, model=st.session_state.get("llm_model"))


def generate_learning_resources(missing_skills):
    """Generate learning resource recommendations for missing skills."""
    skills_str = ", ".join(missing_skills[:5]) if missing_skills else "relevant skills"

    prompt = f"""
    Recommend specific learning resources for developing these skills: {skills_str}.
    For each skill, provide:
    1. One recommended online course with platform
    2. One free resource (article, tutorial, etc.)
    3. Estimated time to achieve basic competency
    
    Format as a concise, structured list with clear headings and brief descriptions.
    """

    return generate_content_with_llm(prompt, model=st.session_state.get("llm_model"))


# --- UI Component Functions ---
def create_job_fit_dashboard(job_fit_metrics):
    """Create a visual dashboard for job fit metrics."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Technical Skills Match")
        st.progress(job_fit_metrics["technical_skills_match"] / 100)
        st.write(f"{job_fit_metrics['technical_skills_match']:.1f}%")

        st.subheader("Soft Skills Match")
        st.progress(job_fit_metrics["soft_skills_match"] / 100)
        st.write(f"{job_fit_metrics['soft_skills_match']:.1f}%")

    with col2:
        st.subheader("Experience Match")
        st.progress(job_fit_metrics["experience_match"] / 100)
        st.write(f"{job_fit_metrics['experience_match']:.1f}%")

        st.subheader("Education Match")
        st.progress(job_fit_metrics["education_match"] / 100)
        st.write(f"{job_fit_metrics['education_match']:.1f}%")

    st.subheader("Overall Compatibility")
    st.progress(job_fit_metrics["overall_match"] / 100)
    st.write(f"{job_fit_metrics['overall_match']:.1f}%")


def create_wordcloud(text):
    """Create and display a word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)


def create_docx_download_button(content, filename):
    """Create a downloadable docx file with the given content."""
    doc = Document()
    doc.add_paragraph(content)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


def generate_job_links(job_title):
    """Generate job search links for the given job title."""
    linkedin_url = generate_linkedin_job_search_url(job_title)
    naukri_url = generate_naukri_job_search_url(job_title)

    col1, col2 = st.columns(2)
    with col1:
        linkedin_button = f"""
            <a href="{linkedin_url}" target="_blank">
                <button style="
                    background-color: #0077B5;
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
                    background-color: #E07A5F;
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


def generate_linkedin_job_search_url(job_title):
    """Generate LinkedIn job search URL for the given job title."""
    keywords = job_title
    encoded_keywords = urllib.parse.quote_plus(keywords)
    return f"https://www.linkedin.com/jobs/search/?keywords={encoded_keywords}&location=India"


def generate_naukri_job_search_url(job_title):
    """Generate Naukri job search URL for the given job title."""
    keywords = job_title
    encoded_keywords = urllib.parse.quote_plus(keywords)
    return f"https://www.naukri.com/{encoded_keywords}-jobs?k={encoded_keywords}&nignbevent_src=jobsearchDeskGNB"


# --- Main Application ---
def main():
    # Page Header
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

    # Navigation
    cols = st.columns(3)
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

    # Initialize models
    embed_model, df, job_tensors, model, skill_keywords, llm_model = initialize_models()

    # Tabs
    tabs = st.tabs(["Resume Upload", "Job Description Analysis"])

    with tabs[0]:  # Resume Upload Tab
        st.write("Upload your resume to find matching roles!")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="resume_upload_tab",
        )

        if uploaded_file:
            resume_text = parse_resume(uploaded_file)
            if resume_text:
                st.session_state.resume_text = resume_text

                with st.expander("View Extracted Resume Text"):
                    st.markdown(
                        f"```{resume_text[:1000]}{'...' if len(resume_text) > 1000 else ''}```"
                    )

                # Extract and display skills
                skills = extract_skills(resume_text, skill_keywords)
                st.markdown("### Extracted Skills")
                if skills:
                    st.markdown(", ".join(skills))
                else:
                    st.markdown("No skills matched from the predefined list.")

                # Generate resume embedding
                with st.spinner("Analyzing resume..."):
                    st.session_state.resume_embedding = torch.tensor(
                        embed_model.encode(resume_text), dtype=torch.float
                    ).unsqueeze(0)
                    st.session_state.extracted_skills = skills

                # Recommend roles
                if st.button("Recommend Roles"):
                    with st.spinner("Finding matching roles..."):
                        recommended_jobs = recommend_jobs(
                            st.session_state.resume_embedding, job_tensors, df, top_n=5
                        )
                        st.session_state.recommended_jobs = recommended_jobs

                        st.success("Recommended Roles:")
                        for i, (job_title, score) in enumerate(recommended_jobs):
                            st.write(
                                f"**{job_title}** - Suitability Score: {score:.1f}"
                            )
                            generate_job_links(job_title)

                            # Create expander for job details
                            with st.expander(f"Explore {job_title} Career Path"):
                                career_path = generate_career_path(job_title)
                                st.write(career_path)

                            with st.expander(f"Salary Estimate for {job_title}"):
                                salary_estimate = generate_salary_estimate(
                                    job_title, skills
                                )
                                st.write(salary_estimate)
            else:
                st.warning("Could not parse the uploaded resume.")

    with tabs[1]:  # Job Description Analysis Tab
        st.write("Enter a job description and analyze it against your resume.")

        # Check if resume is uploaded
        if "resume_text" not in st.session_state:
            st.warning("Please upload a resume on the 'Resume Upload' tab first.")
        else:
            # Initialize job description in session state
            if "job_description" not in st.session_state:
                st.session_state.job_description = ""

            # Job description input
            job_description = st.text_area(
                "Enter Job Description:",
                height=200,
                key="job_description",
                value=st.session_state.job_description,
            )

            # Process job description
            if job_description.strip():
                if st.button("Analyze Job Fit"):
                    with st.spinner("Analyzing job fit..."):
                        # Calculate ATS score
                        ats_score, matched_keywords, missing_keywords = (
                            calculate_ats_score(
                                st.session_state.resume_text, job_description
                            )
                        )

                        # Generate job description embedding
                        job_embedding = torch.tensor(
                            embed_model.encode(job_description), dtype=torch.float
                        ).unsqueeze(0)

                        # Calculate similarity score
                        similarity_score = (
                            torch.nn.functional.cosine_similarity(
                                st.session_state.resume_embedding, job_embedding
                            ).item()
                            * 100
                        )

                        # Comprehensive job fit analysis
                        job_fit_metrics = analyze_resume_job_fit(
                            st.session_state.resume_text, job_description
                        )

                        # Store analysis results in session state
                        st.session_state.ats_score = ats_score
                        st.session_state.matched_keywords = matched_keywords
                        st.session_state.missing_keywords = missing_keywords
                        st.session_state.similarity_score = similarity_score
                        st.session_state.job_fit_metrics = job_fit_metrics

                        # Display basic results
                        st.success(
                            f"Overall Job Compatibility Score: {similarity_score:.1f}%"
                        )
                        st.info(f"ATS Score (Keyword Match): {ats_score:.1f}%")

                        # Display detailed analysis
                        create_job_fit_dashboard(job_fit_metrics)

                # Only show these options if analysis has been run
                if "job_fit_metrics" in st.session_state:
                    # Create expandable sections for detailed analysis
                    with st.expander("Keyword Analysis"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Matched Keywords")
                            st.write(", ".join(st.session_state.matched_keywords[:20]))
                        with col2:
                            st.subheader("Missing Keywords")
                            st.write(", ".join(st.session_state.missing_keywords[:20]))

                    with st.expander("Resume Improvement Suggestions"):
                        improvements = generate_resume_improvements(
                            st.session_state.resume_text, job_description
                        )
                        st.write(improvements)

                    with st.expander("Learning Resources for Missing Skills"):
                        if st.session_state.missing_keywords:
                            resources = generate_learning_resources(
                                st.session_state.missing_keywords
                            )
                            st.write(resources)
                        else:
                            st.write("No significant skill gaps identified!")

                    with st.expander("Generate Cover Letter"):
                        if st.button("Create Cover Letter"):
                            cover_letter = generate_cover_letter(
                                st.session_state.resume_text, job_description
                            )
                            st.write(cover_letter)
                            create_docx_download_button(
                                cover_letter, "cover_letter.docx"
                            )

                    with st.expander("Interview Preparation"):
                        if st.button("Prepare for Interview"):
                            job_title = extract_job_title(job_description)
                            interview_prep = generate_interview_prep(
                                job_title, st.session_state.resume_text
                            )
                            st.write(interview_prep)
                            create_docx_download_button(
                                interview_prep, "interview_prep.docx"
                            )

                    with st.expander("Job Market Insights"):
                        job_title = extract_job_title(job_description)
                        market_insights = generate_job_market_insights(job_title)
                        st.write(market_insights)
            else:
                st.info("Please enter a job description to analyze.")


def extract_job_title(job_description):
    """Extract job title from job description."""
    # Simple regex to find job title patterns
    patterns = [
        r"(?i)job title:\s*([^\n]+)",
        r"(?i)position:\s*([^\n]+)",
        r"(?i)role:\s*([^\n]+)",
        r"(?i)hiring for\s*([^\n]+)",
        r"(?i)looking for\s*a?\s*([^,\.\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, job_description)
        if match:
            return match.group(1).strip()

    # Default to generic title if no match found
    return "Professional Role"


if __name__ == "__main__":
    main()
