import streamlit as st
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Any

from models import Resume, Job
from recommendation import get_recommendations
from resume_parser import parse_resume, CITY_STATE_MAP
from database import (
    add_resume as add_resume_to_db,
    retrieve_resumes,
    delete_resume as delete_resume_from_db,
    delete_resume_by_filename,
    update_resume_workflow
)
from job_database import (
    add_job as add_job_to_db,
    retrieve_jobs,
    delete_job as delete_job_from_db
)

# --- Page Configuration ---
st.set_page_config(page_title="Resume Analysis AI", layout="wide", initial_sidebar_state="expanded")


# --- Custom CSS for Beautiful UI ---
def local_css():
    primary_color = "#FF4B4B" # Streamlit's default red, works well for light theme
    background_color = "#FFFFFF" # White background
    secondary_background_color = "#F0F2F6" # Light grey for secondary elements like input fields
    text_color = "#262730" # Dark text for contrast
    card_background = "#FFFFFF"
    card_border = "#E0E0E0" # Light border for cards
    metric_background = "#F0F2F6" # Light grey background for metrics

    st.markdown(f"""
    <style>
    /* General body styling */
    .stApp {{
        background-color: {background_color} !important;
        color: {text_color} !important;
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
    }}

    /* Input fields and select boxes */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div,
    .stNumberInput>div>div>input {{
        color: {text_color}; /* Dark text color for inputs */
        background-color: {secondary_background_color}; /* Light background for inputs */
        border: 1px solid {card_border}; /* Consistent border */
    }}

    /* Text elements */
    .stMarkdown, .stText, .stJson, p, li {{
        color: {text_color} !important;
    }}

    /* Card-like containers */
    .card {{
        background-color: {card_background};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.05); /* Lighter shadow for light theme */
        transition: 0.3s;
        border: 1px solid {card_border};
    }}
    .card:hover {{
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.1);
        border: 1px solid {primary_color};
    }}

    /* Metric styles */
    .stMetric {{
        background-color: {metric_background};
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid {card_border}; /* Consistent border for metrics */
    }}
    .stMetric > div > div > div {{ /* Target metric value and label */
        color: {text_color} !important;
    }}

    /* Expander styling */
    .stExpander {{
        border-radius: 8px !important;
        border: 1px solid {card_border} !important;
    }}
    .stExpander details summary p {{
        color: {text_color} !important; /* Ensure expander header text is dark */
    }}
    
    /* Button styling */
    .stButton>button {{
        border-radius: 8px;
        border: 1px solid {primary_color};
        color: {primary_color};
        background-color: {background_color}; /* Button background matches app background */
    }}
    .stButton>button:hover {{
        border-color: {primary_color};
        background-color: {primary_color};
        color: #fff; /* White text on hover for contrast */
    }}

    /* Plot style */
    .stPlotlyChart {{
        border-radius: 10px;
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 1.2rem;
        color: {text_color} !important; /* Ensure tab titles are dark */
    }}
    .stTabs [data-baseweb="tab-list"] button {{
        background-color: {secondary_background_color}; /* Light background for tabs */
        border-radius: 8px 8px 0 0;
        border-bottom: none;
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-top: 2px solid {primary_color}; /* Highlight active tab */
        color: {text_color};
        background-color: {background_color}; /* Active tab matches main app background */
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {secondary_background_color}; /* Background for tab bar */
        border-radius: 8px;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {background_color}; /* Tab content matches app background */
    }}


    /* Info, Success, Warning, Error boxes */
    .stAlert {{
        color: {text_color} !important; /* Ensure alert text is dark */
        background-color: {secondary_background_color} !important; /* Light background for alerts */
        border-color: {card_border} !important;
    }}
    .stAlert > div > div {{
        color: {text_color} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions (from original code, with minor adjustments) ---
def ensure_uploads_dir():
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

# --- UI Rendering Functions ---

def display_dashboard(all_resumes: List[Resume]):
    st.subheader("📊 Recruiter Dashboard")
    st.markdown("---")
    st.markdown("#### Key Metrics")

    col1, col2, col3 = st.columns(3)
    experience_levels = [r.experience for r in all_resumes if r.experience is not None and r.experience > 0]
    all_skills = [skill.lower() for resume in all_resumes for skill in resume.skills]
    all_locations = [r.location for r in all_resumes if r.location]

    with col1:
        st.metric("Total Resumes", len(all_resumes))
    with col2:
        avg_exp = round(np.mean(experience_levels), 1) if experience_levels else 0
        st.metric("Avg. Experience (Years)", avg_exp)
    with col3:
        st.metric("Total Skills Found", len(set(all_skills)))

    st.markdown("---")
    st.markdown("#### Visualizations")

    # Use a default theme for plots
    plt.style.use('default')

    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.markdown("**Experience Distribution**")
        if experience_levels:
            fig, ax = plt.subplots()
            ax.hist(experience_levels, bins=15, color='#f63366', edgecolor='black')
            ax.set_xlabel("Years of Experience")
            ax.set_ylabel("Number of Resumes")
            ax.grid(axis='y', alpha=0.5)
            st.pyplot(fig)
        else:
            st.info("No experience data to display.")

        st.markdown("**Resumes by Location**")
        if all_locations:
            loc_counts = Counter(all_locations)
            top_locs = loc_counts.most_common(7)
            fig, ax = plt.subplots()
            ax.barh([loc[0] for loc in top_locs], [loc[1] for loc in top_locs], color='skyblue')
            ax.set_xlabel("Number of Resumes")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.info("No location data to display.")

    with col_viz2:
        st.markdown("**Top 10 Skills**")
        if all_skills:
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(10)
            fig, ax = plt.subplots()
            ax.barh([s[0] for s in top_skills], [s[1] for s in top_skills], color='lightgreen')
            ax.set_xlabel("Frequency")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.info("No skill data to display.")

        st.markdown("**Resumes by Job Title**")
        all_job_titles = [r.job_title for r in all_resumes if r.job_title]
        if all_job_titles:
            title_counts = Counter(all_job_titles)
            top_titles = title_counts.most_common(7)
            fig, ax = plt.subplots()
            ax.barh([t[0] for t in top_titles], [t[1] for t in top_titles], color='lightcoral')
            ax.set_xlabel("Number of Resumes")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.info("No job title data to display.")


def display_resume_card(resume: Resume, score: float, context_prefix: str, reasons: List[str] = None):
    exp_str = f"{resume.experience} years" if resume.experience is not None else "N/A"
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {resume.name or 'N/A'}")
        st.markdown(f"**{resume.job_title or 'No Title Found'}**")
        st.markdown(f"📍 {resume.location or 'N/A'} | 💼 {exp_str}")
        
        skills_str = ', '.join(resume.skills) if resume.skills else 'N/A'
        st.markdown(f"**Skills:** {skills_str}")
        
    with col2:
        st.metric("Match Score", f"{score:.1f}%")
        st.progress(int(score))

    with st.expander("Show More Details & Actions"):
        st.markdown(f"**📧 Email:** {resume.email or 'N/A'}")
        st.markdown(f"**📞 Contact:** {resume.contact_number or 'N/A'}")
        st.markdown(f"**🎓 Education:** {', '.join(resume.education) if resume.education else 'N/A'}")
        
        if reasons:
            st.markdown("**Analysis notes:**")
            for reason in reasons:
                st.info(reason)

        if st.button("View Full Resume Text", key=f"{context_prefix}_view_full_{resume.id}"):
            st.session_state.selected_resume_to_view = resume
            
    st.markdown('</div>', unsafe_allow_html=True)

def render_search_tab():
    st.header("🔎 Search for Candidates")
    
    with st.form("search_form_new"):
        st.subheader("Recruiter Search Filters")
        
        col1, col2 = st.columns(2)
        with col1:
            search_job_title = st.text_input("Job Title / Designation", help="e.g., Data Analyst")
        with col2:
            search_location = st.text_input("Location (comma-separated)", help="e.g., Bengaluru, new york")
        
        col1, col2 = st.columns(2)
        with col1:
            min_exp = st.number_input("Min Experience (years)", 0, 50, 0)
        with col2:
            max_exp = st.number_input("Max Experience (years)", 0, 50, 50)
        
        education_filter = st.text_input("Education (comma-separated)", help="e.g., BCA, B.Tech")
        search_skills = st.text_input("Core Skills (comma-separated)", help="e.g., Python, SQL, Power BI")
        boolean_query = st.text_input("Advanced Search: Keywords & Exclusions", help="e.g., (sql or powerbi), not java")
        
        if st.form_submit_button("Search Resumes"):
            with st.spinner("Analyzing and ranking resumes..."):
                search_criteria = {
                    "job_title": search_job_title, "location": search_location,
                    "min_exp": min_exp, "max_exp": max_exp,
                    "education": [e.strip().lower() for e in education_filter.split(',') if e.strip()],
                    "skills": [s.strip().lower() for s in search_skills.split(',') if s.strip()],
                    "boolean_query": boolean_query,
                }
                
                boolean_and_terms, boolean_or_terms, boolean_not_terms = parse_boolean_query(boolean_query)
                all_search_skills_combined = set(search_criteria["skills"]).union(boolean_and_terms)
                
                search_criteria["all_search_skills_combined"] = all_search_skills_combined
                search_criteria["boolean_or_terms"] = boolean_or_terms
                search_criteria["boolean_not_terms"] = boolean_not_terms

                st.session_state.search_results, st.session_state.last_search_debug = new_run_search(search_criteria)
    
    # --- Display search results outside the form ---
    if "search_results" in st.session_state:
        results = st.session_state.search_results
        if results:
            st.success(f"Found and ranked {len(results)} matching resumes.")
            for item in results:
                display_resume_card(resume=item["resume"], score=item["score"], context_prefix="search")
        else:
            st.warning("No resumes found that meet your search criteria with at least 50% relevance.")
            if st.session_state.get("last_search_debug"):
                with st.expander("Show Search Debug Information"):
                    st.json(st.session_state.last_search_debug)

def render_recommend_tab():
    st.header("🤖 Get Recommendations for a Job")
    all_jobs = retrieve_jobs()
    
    if not all_jobs:
        st.warning("No jobs found. Please add a job in the 'Manage Jobs' tab to get recommendations.")
        return

    job_options = {job['job_title']: job['id'] for job in all_jobs}
    selected_job_title = st.selectbox("Select a Job to find candidates for:", options=list(job_options.keys()))
    num_recommendations = st.slider("Show Top N Recommendations", 1, 20, 5)
    
    if st.button("Get Recommendations"):
        with st.spinner("Analyzing resumes against the job description..."):
            selected_job_obj = next((job for job in all_jobs if job['id'] == job_options[selected_job_title]), None)
            if selected_job_obj:
                recommendations = get_recommendations(Job(**selected_job_obj), [Resume(**r) for r in retrieve_resumes()])
                st.success(f"Displaying top {min(num_recommendations, len(recommendations))} matches for '{selected_job_obj['job_title']}':")
                for rec in recommendations[:num_recommendations]:
                    display_resume_card(resume=rec['details'], score=rec['score'], context_prefix="recommend")

def render_jobs_tab():
    st.header("📝 Manage Job Listings")
    with st.expander("Add a New Job", expanded=False):
        with st.form("add_job_form", clear_on_submit=True):
            job_title = st.text_input("Job Title")
            required_experience = st.number_input("Required Experience (years)", 0)
            education_level = st.text_input("Required Education")
            job_type = st.selectbox("Job Type", ['full-time', 'part-time', 'contract', 'internship', 'remote', 'hybrid'])
            skills = st.text_area("Required Skills (comma-separated)")
            job_description_text = st.text_area("Full Job Description")
            if st.form_submit_button("Add Job"):
                if job_title and skills:
                    add_job_to_db(Job(job_title=job_title, required_experience=required_experience, education_level=education_level, job_type=job_type, skills=[s.strip() for s in skills.split(',')], job_description_text=job_description_text).model_dump())
                    st.success(f"Job '{job_title}' added successfully!")
                else:
                    st.warning("Job Title and Skills are required.")
    
    st.subheader("Existing Jobs")
    all_jobs_display = retrieve_jobs()
    if not all_jobs_display:
        st.info("No jobs currently in the database.")
    else:
        for job in all_jobs_display:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{job['job_title']}** ({job['job_type']})")
                    st.markdown(f"**Experience:** {job['required_experience']} years | **Skills:** {', '.join(job['skills'])}")
                with col2:
                    if st.button("Delete", key=f"del_job_{job['id']}"):
                        delete_job_from_db(job['id'])
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

def render_resumes_tab():
    st.header("🗂️ Manage All Resumes")
    
    st.subheader("Existing Resumes in Database")
    all_resumes_display = retrieve_resumes()
    if not all_resumes_display:
        st.info("No resumes currently in the database.")
    else:
        for resume_dict in all_resumes_display:
            resume = Resume(**resume_dict)
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{resume.name or 'N/A'}**")
                    st.markdown(f"`{resume.file_name}`")
                with col2:
                     st.markdown(f"**Title:** {resume.job_title or 'N/A'}")
                     st.markdown(f"**Exp:** {resume.experience or 'N/A'} yrs | **Loc:** {resume.location or 'N/A'}")
                with col3:
                    if st.button("View", key=f"manage_view_{resume.id}"):
                        st.session_state.selected_resume_to_view = resume
                    if st.button("Delete", key=f"manage_del_{resume.id}"):
                        delete_resume_from_db(resume.id)
                        file_to_delete = os.path.join("uploads", resume.file_name)
                        if os.path.exists(file_to_delete):
                            os.remove(file_to_delete)
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Danger Zone
    st.divider()
    with st.expander("⚠️ Danger Zone"):
        st.warning("This action is irreversible and will delete ALL resumes from the database and all files from the 'uploads' folder.")
        if st.button("Delete ALL Resumes"):
            from database import delete_all_resumes
            with st.spinner("Deleting all resumes and files..."):
                deleted_count = delete_all_resumes()
                files_deleted = 0
                for filename in os.listdir("uploads"):
                    try:
                        os.remove(os.path.join("uploads", filename))
                        files_deleted += 1
                    except Exception as e:
                        st.error(f"Error deleting {filename}: {e}")
            st.success(f"Successfully deleted {deleted_count} resumes and {files_deleted} files.")
            st.rerun()


# --- Main Application Logic ---

# --- Functions from original code required for backend logic ---
# These are kept as-is because they are not related to UI rendering
def _map_education_to_level(education_term: str) -> str:
    term = education_term.lower()
    bachelor_keywords = ["bachelor", "bsc", "b.sc", "ba", "b.a", "bcom", "b.com", "bca", "b.c.a", "btech", "b.tech", "be", "b.e", "bba", "b.b.a", "bms", "b.m.s"]
    master_keywords = ["master", "msc", "m.sc", "ma", "m.a", "mcom", "m.com", "mca", "m.c.a", "mtech", "m.tech", "mba", "m.b.a"]
    doctorate_keywords = ["phd", "ph.d", "doctor"]
    for keyword in bachelor_keywords:
        if keyword in term: return "bachelor"
    for keyword in master_keywords:
        if keyword in term: return "master"
    for keyword in doctorate_keywords:
        if keyword in term: return "doctorate"
    return term

def parse_boolean_query(query: str) -> Tuple[List[str], List[List[str]], List[str]]:
    and_terms, or_terms, not_terms = set(), [], set()
    raw_segments = [s.strip() for s in query.lower().split(',') if s.strip()]
    for segment in raw_segments:
        not_match = re.match(r"not\s+(.+)", segment)
        if not_match:
            not_terms.add(not_match.group(1).strip())
            continue
        or_match = re.match(r"(.+)\s+or\s+(.+)", segment)
        if or_match:
            or_terms.append([or_match.group(1).strip(), or_match.group(2).strip()])
            continue
        and_match = re.match(r"(.+)\s+and\s+(.+)", segment)
        if and_match:
            and_terms.add(and_match.group(1).strip())
            and_terms.add(and_match.group(2).strip())
            continue
        and_terms.add(segment)
    return list(and_terms), or_terms, list(not_terms)

def calculate_relevance_score(resume: Resume, search_criteria: Dict[str, Any]) -> float:
    score = 0.0
    weights = {"skills": 0.5, "experience": 0.25, "job_title": 0.15, "location": 0.10}
    all_search_skills = search_criteria.get("all_search_skills_combined", set())
    if all_search_skills:
        resume_skills = {s.lower() for s in (resume.skills or [])}
        matched = all_search_skills.intersection(resume_skills)
        if all_search_skills: score += weights["skills"] * (len(matched) / len(all_search_skills))
    if resume.experience is not None:
        min_exp, max_exp = search_criteria.get("min_exp", 0), search_criteria.get("max_exp", 50)
        if min_exp <= resume.experience <= max_exp: score += weights["experience"]
    if search_criteria.get("job_title"):
        jt = search_criteria["job_title"].lower()
        if jt in f"{resume.job_title or ''} {resume.resume_text or ''}".lower(): score += weights["job_title"]
    search_location_str = search_criteria.get("location", "")
    if search_location_str and resume.location:
        search_locations = {loc.strip().lower() for loc in search_location_str.split(',')}
        resume_loc_lower = resume.location.lower()
        if any(search_loc in resume_loc_lower or (resume_loc_lower in CITY_STATE_MAP and CITY_STATE_MAP[resume_loc_lower] == search_loc) for search_loc in search_locations):
            score += weights["location"]
    return score * 100

def new_run_search(search_criteria: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_resumes = [Resume(**r) for r in retrieve_resumes()]
    all_resumes_with_scores_and_reasons, debug_info = [], {}
    # ... (rest of the function is complex business logic, keeping it as is)
    search_all_skills_combined = search_criteria.get("all_search_skills_combined", set())
    search_boolean_not_terms = search_criteria.get("boolean_not_terms", [])
    for resume in all_resumes:
        reasons, hard_filter_passed = [], True
        resume_full_text_lower = (resume.resume_text or "").lower()
        resume_skills_lower = {s.strip().lower() for s in (resume.skills or [])}
        if search_boolean_not_terms:
            if any(not_term.strip().lower() in resume_full_text_lower or not_term.strip().lower() in resume_skills_lower for not_term in search_boolean_not_terms):
                hard_filter_passed = False
        if not hard_filter_passed: continue
        score = calculate_relevance_score(resume, search_criteria)
        if score >= 40.0: # Lowered threshold slightly to be more inclusive
            all_resumes_with_scores_and_reasons.append({"resume": resume, "score": score, "reasons": reasons})
    all_resumes_with_scores_and_reasons.sort(key=lambda x: x["score"], reverse=True)
    return all_resumes_with_scores_and_reasons, debug_info


# --- App Execution ---
if __name__ == "__main__":
    ensure_uploads_dir()
    local_css()

    st.title("✨ AI-Powered Resume Analysis")

    # --- Sidebar for Uploads ---
    with st.sidebar:
        st.header("📄 Upload Resumes")
        if 'file_uploader_key' not in st.session_state: st.session_state.file_uploader_key = 0
        uploaded_files = st.file_uploader("Upload one or more PDF resumes", type="pdf", accept_multiple_files=True, key=f"file_uploader_{st.session_state.file_uploader_key}")
        
        if uploaded_files and st.button("Process Uploaded Resumes"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                delete_resume_by_filename(uploaded_file.name)
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                try:
                    add_resume_to_db(parse_resume(file_path).dict())
                except Exception as e:
                    st.error(f"Error with '{uploaded_file.name}': {e}")
                    if os.path.exists(file_path): os.remove(file_path)
                progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.success("All resumes processed!")
            st.session_state.file_uploader_key += 1
            st.rerun()

    # --- Full Screen Resume Viewer ---
    if "selected_resume_to_view" in st.session_state and st.session_state.selected_resume_to_view:
        resume = st.session_state.selected_resume_to_view
        st.header(f"Viewing: {resume.name}")
        st.text_area("Full Resume Text", resume.resume_text, height=500)
        if st.button("Close Viewer"):
            st.session_state.selected_resume_to_view = None
            st.rerun()
    else:
        # --- Main UI Tabs ---
        dashboard_tab, search_tab, recommend_tab, jobs_tab, resumes_tab = st.tabs(["📊 Dashboard", "🔎 Resume Search", "🤖 Job Recommendations", "📝 Manage Jobs", "🗂️ Manage Resumes"])

        with dashboard_tab:
            all_resumes_for_dashboard = [Resume(**r) for r in retrieve_resumes()]
            if all_resumes_for_dashboard:
                display_dashboard(all_resumes_for_dashboard)
            else:
                st.info("No resumes in the database. Upload some resumes to see the dashboard.")

        with search_tab:
            render_search_tab()

        with recommend_tab:
            render_recommend_tab()
            
        with jobs_tab:
            render_jobs_tab()

        with resumes_tab:
            render_resumes_tab()