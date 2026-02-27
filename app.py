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
st.set_page_config(page_title="Resume Analysis AI", layout="wide")

# --- Helper Functions ---
def ensure_uploads_dir():
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        print("✓ Created 'uploads' directory")

def display_dashboard(all_resumes: List[Resume]):
    st.subheader("Recruiter Dashboard")
    st.markdown("#### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Resumes", len(all_resumes))
    experience_levels = [r.experience for r in all_resumes if r.experience is not None and r.experience > 0]
    with col2:
        avg_exp = round(np.mean(experience_levels), 1) if experience_levels else 0
        st.metric("Avg. Experience (Years)", avg_exp)
    all_skills = [skill.lower() for resume in all_resumes for skill in resume.skills]
    with col3:
        top_skills = [item[0] for item in Counter(all_skills).most_common(5)]
        st.markdown("**Top 5 Skills**"); st.text(', '.join(top_skills))
    all_locations = [r.location for r in all_resumes if r.location]
    with col4:
        top_locations = [item[0] for item in Counter(all_locations).most_common(5)]
        st.markdown("**Top 5 Locations**"); st.text(', '.join(top_locations))
    st.markdown("---")
    st.markdown("#### Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Experience Distribution**")
        if experience_levels:
            fig, ax = plt.subplots(); ax.hist(experience_levels, bins=15, color='skyblue', edgecolor='black')
            ax.set_xlabel("Years of Experience"); ax.set_ylabel("Number of Resumes"); st.pyplot(fig)
        else: st.info("No experience data to display.")
    with col2:
        st.markdown("**Resumes by Location**")
        if all_locations:
            loc_counts = Counter(all_locations); top_locs = loc_counts.most_common(10)
            fig, ax = plt.subplots(); ax.barh([loc[0] for loc in top_locs], [loc[1] for loc in top_locs], color='lightgreen')
            ax.set_xlabel("Number of Resumes"); st.pyplot(fig)
        else: st.info("No location data to display.")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Resumes by Job Title**")
        all_job_titles = [r.job_title for r in all_resumes if r.job_title]
        if all_job_titles:
            title_counts = Counter(all_job_titles); top_titles = title_counts.most_common(10)
            fig, ax = plt.subplots(); ax.barh([t[0] for t in top_titles], [t[1] for t in top_titles], color='lightcoral')
            ax.set_xlabel("Number of Resumes"); st.pyplot(fig)
        else: st.info("No job title data to display.")

def _map_education_to_level(education_term: str) -> str:
    term = education_term.lower()

    bachelor_keywords = [
        "bachelor", "bsc", "b.sc", "ba", "b.a", "bcom", "b.com", "bca", "b.c.a",
        "btech", "b.tech", "be", "b.e", "bba", "b.b.a", "bms", "b.m.s"
    ]
    master_keywords = [
        "master", "msc", "m.sc", "ma", "m.a", "mcom", "m.com", "mca", "m.c.a",
        "mtech", "m.tech", "mba", "m.b.a"
    ]
    doctorate_keywords = [
        "phd", "ph.d", "doctor"
    ]

    for keyword in bachelor_keywords:
        if keyword in term:
            return "bachelor"
    for keyword in master_keywords:
        if keyword in term:
            return "master"
    for keyword in doctorate_keywords:
        if keyword in term:
            return "doctorate"
            
    return term # Return original if no mapping

def parse_boolean_query(query: str) -> Tuple[List[str], List[List[str]], List[str]]:
    and_terms, or_terms, not_terms = set(), [], set()
    
    raw_segments = [s.strip() for s in query.lower().split(',') if s.strip()]

    for segment in raw_segments:
        # Check for explicit NOT (e.g., "not java")
        not_match = re.match(r"not\s+(.+)", segment)
        if not_match:
            not_terms.add(not_match.group(1).strip())
            continue

        # Check for explicit OR (e.g., "python or java") - handles only two terms
        or_match = re.match(r"(.+)\s+or\s+(.+)", segment)
        if or_match:
            or_terms.append([or_match.group(1).strip(), or_match.group(2).strip()])
            continue
            
        # Check for explicit AND (e.g., "python and sql") - handles only two terms
        and_match = re.match(r"(.+)\s+and\s+(.+)", segment)
        if and_match:
            and_terms.add(and_match.group(1).strip())
            and_terms.add(and_match.group(2).strip())
            continue

        # If no explicit operator, treat the entire segment as an AND term
        and_terms.add(segment)

    return list(and_terms), or_terms, list(not_terms)

def calculate_relevance_score(resume: Resume, search_criteria: Dict[str, Any]) -> float:
    score = 0.0
    weights = {"skills": 0.5, "experience": 0.25, "job_title": 0.15, "location": 0.10}

    # ---- Skills ----
    all_search_skills = search_criteria.get("all_search_skills_combined", set())
    if all_search_skills:
        resume_skills = {s.lower() for s in (resume.skills or [])}
        matched = all_search_skills.intersection(resume_skills)
        score += weights["skills"] * (len(matched) / len(all_search_skills))

    # ---- Experience ----
    if resume.experience is not None:
        min_exp = search_criteria.get("min_exp", 0)
        max_exp = search_criteria.get("max_exp", 50)
        if min_exp <= resume.experience <= max_exp:
            score += weights["experience"]

    # ---- Job title / text ----
    if search_criteria.get("job_title"):
        jt = search_criteria["job_title"].lower()
        resume_combined_text = f"{resume.job_title or ''} {resume.resume_text or ''}".lower()
        if jt in resume_combined_text:
            score += weights["job_title"]

    # ---- Location ----
    search_location_str = search_criteria.get("location", "")
    if search_location_str and resume.location:
        search_locations = {loc.strip().lower() for loc in search_location_str.split(',')}
        resume_loc_lower = resume.location.lower()
        
        location_matched = False
        for search_loc in search_locations:
            if search_loc in resume_loc_lower:
                location_matched = True
                break
            if resume_loc_lower in CITY_STATE_MAP and CITY_STATE_MAP[resume_loc_lower] == search_loc:
                location_matched = True
                break
        
        if location_matched:
            score += weights["location"]
            
    return score * 100


def new_run_search(search_criteria: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    
    all_resumes = [Resume(**r) for r in retrieve_resumes()]
    all_resumes_with_scores_and_reasons = []
    debug_info = {}

    # Normalize search criteria
    search_job_title_lower = search_criteria.get("job_title", "").strip().lower()
    search_location_lower = search_criteria.get("location", "").strip().lower()
    search_education_keywords = search_criteria.get("education", [])
    search_min_exp = search_criteria.get("min_exp", 0)
    search_max_exp = search_criteria.get("max_exp", 50)
    search_all_skills_combined = search_criteria.get("all_search_skills_combined", set())
    search_boolean_or_terms = search_criteria.get("boolean_or_terms", [])
    search_boolean_not_terms = search_criteria.get("boolean_not_terms", [])

    for resume in all_resumes:
        reasons = []

        # Normalize resume fields
        resume_name = getattr(resume, "name", "N/A")
        resume_file_name = getattr(resume, "file_name", "N/A")
        resume_loc = (resume.location or "").strip().lower()
        resume_skills_lower = {s.strip().lower() for s in (resume.skills or [])}
        resume_job_title_lower = (resume.job_title or "").strip().lower()
        resume_full_text_lower = (resume.resume_text or "").lower()

        # Hard Filter: Boolean NOT terms
        hard_filter_passed = True
        if search_boolean_not_terms:
            for not_term in search_boolean_not_terms:
                if not_term.strip().lower() in resume_full_text_lower or not_term.strip().lower() in resume_skills_lower:
                    reasons.append(f"HARD FILTER: Boolean NOT Mismatch - Found excluded term '{not_term}'")
                    hard_filter_passed = False
                    break
        
        if not hard_filter_passed:
            key = f"{resume_name} - {resume_file_name}" if resume_name != 'N/A' else resume_file_name
            debug_info[key] = {
                "reasons": reasons,
                "Search Criteria": search_criteria,
                "Resume Data": {
                    "Name": resume.name, "File Name": resume.file_name, "Skills": resume.skills,
                    "Experience": resume.experience, "Location": resume.location,
                    "Job Title": resume.job_title, "Education": resume.education,
                }
            }
            continue

        # Soft checks for scoring - these add to reasons but don't filter directly
        if search_job_title_lower:
            if search_job_title_lower not in resume_job_title_lower and \
               search_job_title_lower not in resume_full_text_lower:
                reasons.append(f"INFO: Job Title mismatch (score will be lower): Required='{search_job_title_lower}', Found='{resume.job_title}'")

        if search_location_lower:
            location_matched = False
            if resume_loc:
                if search_location_lower in resume_loc:
                    location_matched = True
                elif search_location_lower in ["karnataka", "maharashtra", "tamil nadu", "telangana", "delhi", "west bengal", "gujarat", "uttar pradesh", "haryana"]:
                    if resume_loc in CITY_STATE_MAP and CITY_STATE_MAP[resume_loc] == search_location_lower:
                        location_matched = True
            if not location_matched:
                reasons.append(f"INFO: Location Mismatch (score will be lower): Required = '{search_location_lower}', Found = '{resume.location}'")

        if search_education_keywords:
            education_matched = False
            resume_education_list_lower = [edu_entry.lower() for edu_entry in (resume.education if isinstance(resume.education, list) else [])]
            for required_edu_keyword in search_education_keywords:
                mapped_search_level = _map_education_to_level(required_edu_keyword)
                if any(mapped_search_level in _map_education_to_level(resume_edu_entry) for resume_edu_entry in resume_education_list_lower):
                    education_matched = True
                    break
            if not education_matched:
                reasons.append(f"INFO: Education Mismatch (score will be lower): Required = {search_education_keywords}, Found = '{resume.education}'")

        if resume.experience is not None:
            if not (search_min_exp <= resume.experience <= search_max_exp):
                reasons.append(f"INFO: Experience Mismatch (score will be lower): Required = {search_min_exp}-{search_max_exp} years, Found = {resume.experience} years")

        if search_all_skills_combined:
            if not search_all_skills_combined:
                skill_score = 1.0
            else:
                matched = search_all_skills_combined.intersection(resume_skills_lower)
                skill_score = len(matched) / len(search_all_skills_combined)
            missing_skills = search_all_skills_combined - resume_skills_lower
            if skill_score < 0.5:
                reasons.append(f"INFO: Skill Match Below 50% (score will be lower): Required skills match was only {skill_score:.0%}. Missing: {sorted(list(missing_skills))}")
            elif missing_skills:
                reasons.append(f"INFO: Missing Some Required Skills (score will be lower): {sorted(list(missing_skills))}")

        if search_boolean_or_terms:
            or_passes = False
            for or_pair in search_boolean_or_terms:
                if any(term.strip().lower() in resume_full_text_lower or term.strip().lower() in resume_skills_lower for term in or_pair):
                    or_passes = True
                    break
            if not or_passes:
                reasons.append(f"INFO: Boolean OR Mismatch (score will be lower): No terms from '{search_boolean_or_terms}' found.")

        score = calculate_relevance_score(resume, search_criteria)
        all_resumes_with_scores_and_reasons.append({"resume": resume, "score": score, "reasons": reasons})
    
    # Final filtering by score
    filtered_and_scored_resumes = []
    for item in all_resumes_with_scores_and_reasons:
        if item["score"] >= 50.0:
            filtered_and_scored_resumes.append({"resume": item["resume"], "score": item["score"]})
        else:
            key = f"{item['resume'].name} - {item['resume'].file_name}" if item['resume'].name != 'N/A' else item['resume'].file_name
            debug_info[key] = {
                "reasons": item["reasons"] + [f"HARD FILTER: Overall score below 50% ({item['score']:.2f}%)"],
                "Search Criteria": search_criteria,
                "Resume Data": {
                    "Name": item['resume'].name, "File Name": item['resume'].file_name, "Skills": item['resume'].skills,
                    "Experience": item['resume'].experience, "Location": item['resume'].location,
                    "Job Title": item['resume'].job_title, "Education": item['resume'].education,
                }
            }

    # Sorting
    filtered_and_scored_resumes.sort(key=lambda x: x["score"] if x["score"] is not None else 0.0, reverse=True)
    
    return filtered_and_scored_resumes, debug_info

# --- Main Application ---
ensure_uploads_dir()
st.title("AI-Powered Resume Analysis")

with st.sidebar:
    st.header("Upload Resumes")
    if 'file_uploader_key' not in st.session_state: st.session_state.file_uploader_key = 0
    uploaded_files = st.file_uploader("Upload one or more PDF resumes", type="pdf", accept_multiple_files=True, key=f"file_uploader_{st.session_state.file_uploader_key}")
    if uploaded_files and st.button("Process Uploaded Resumes"):
        for uploaded_file in uploaded_files:
            delete_resume_by_filename(uploaded_file.name)
            print(f"✓ Deleted existing resume: {uploaded_file.name}")
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            print(f"✓ Saved file: {uploaded_file.name}")
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    add_resume_to_db(parse_resume(file_path).dict()); st.success(f"Processed & Updated '{uploaded_file.name}"); print(f"✓ Processed & added to database: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error with '{uploaded_file.name}': {e}")
                    if os.path.exists(file_path): os.remove(file_path); print(f"✗ Error processing {uploaded_file.name}, file removed")
        st.session_state.file_uploader_key += 1; print("✓ All resumes processed, refreshing interface"); st.rerun()

# --- UI Tabs ---
search_tab, recommend_tab, jobs_tab, resumes_tab = st.tabs(["🔎 Resume Search", "🤖 Job Recommendations", "📝 Manage Jobs", "🗂️ Manage Resumes"])

with search_tab:
    st.header("Search for Candidates")
    
    # Initialize session state for displaying full resume if not already present
    if "selected_resume_text_to_view" not in st.session_state:
        st.session_state.selected_resume_text_to_view = None
    if "selected_resume_name_to_view" not in st.session_state:
        st.session_state.selected_resume_name_to_view = None

    with st.form("search_form_new"):
        st.subheader("Recruiter Search Filters")
        
        col1, col2 = st.columns(2)
        with col1:
            search_job_title = st.text_input("Job Title / Designation", help="e.g., Data Analyst")
        with col2:
            search_location = st.text_input("Location (comma-separated)", help="e.g., Bengaluru, Karnataka, new york, delhi")
        
        col1, col2 = st.columns(2)
        with col1:
            min_exp = st.number_input("Min Experience (years)", 0, 50, 0)
        with col2:
            max_exp = st.number_input("Max Experience (years)", 0, 50, 50)
        
        education_filter = st.text_input("Education (comma-separated)", help="e.g., BCA, MCA, B.Tech")
        search_skills = st.text_input("Skills (comma-separated)", help="e.g., Python, SQL, Power BI")
        boolean_query = st.text_input("Advanced Boolean Search", help="e.g., Python AND (SQL OR PowerBI) NOT Java")
        
        if st.form_submit_button("Search Resumes"):
            # Call the new_run_search function with collected criteria
            search_criteria = {
                "job_title": search_job_title,
                "location": search_location,
                "min_exp": min_exp,
                "max_exp": max_exp,
                "education": [e.strip().lower() for e in education_filter.split(',') if e.strip()],
                "skills": [s.strip().lower() for s in search_skills.split(',') if s.strip()],
                "boolean_query": boolean_query,
            }
            
            # Combine skills from both fields for initial processing
            boolean_and_terms, boolean_or_terms, boolean_not_terms = parse_boolean_query(boolean_query)
            all_search_skills_combined = set(search_criteria["skills"]).union(boolean_and_terms)
            
            search_criteria["all_search_skills_combined"] = all_search_skills_combined
            search_criteria["boolean_or_terms"] = boolean_or_terms
            search_criteria["boolean_not_terms"] = boolean_not_terms

            st.session_state.search_results, st.session_state.last_search_debug = new_run_search(search_criteria)
            
    # Display search results outside the form
    if st.session_state.get("search_results"):
        st.success(f"Found and ranked {len(st.session_state.search_results)} matching resumes.")
        for item in st.session_state.search_results:
            resume, score = item["resume"], item["score"]
            exp_str = f"{resume.experience} years" if resume.experience is not None else "N/A"
            with st.expander(f"**{resume.name or 'N/A'}** | Score: **{score if score is not None else 0.0:.2f}%** | Status: **{resume.status}**"):
                st.markdown(f"**Email:** {resume.email or 'N/A'}")
                st.markdown(f"**Contact:** {resume.contact_number or 'N/A'}")
                st.markdown(f"**Location:** {resume.location or 'N/A'}")
                st.markdown(f"**Education:** {', '.join(resume.education) if resume.education else 'N/A'}")
                st.markdown(f"**Experience:** {exp_str}")
                st.markdown(f"**Detected Job Title:** {resume.job_title or 'N/A'}")
                st.markdown(f"**Skills:** {', '.join(resume.skills) if resume.skills else 'N/A'}")
                if st.button("View Full Resume", key=f"view_full_resume_{resume.id}"):
                    st.session_state.selected_resume_text_to_view = resume.resume_text
                    st.session_state.selected_resume_name_to_view = resume.name or "N/A" # Store name for header
        
        # Display full resume content if selected
        if st.session_state.get("selected_resume_text_to_view"):
            st.markdown("---")
            st.subheader(f"Full Resume Content: {st.session_state.selected_resume_name_to_view}")
            st.text_area("Resume Text", st.session_state.selected_resume_text_to_view, height=500, key="full_resume_text_display")
            if st.button("Clear Full Resume View", key="clear_full_resume_view"):
                st.session_state.selected_resume_text_to_view = None
                st.session_state.selected_resume_name_to_view = None
                st.rerun() # Rerun to clear the text area immediately
    elif st.session_state.get("search_results") is not None and not st.session_state.search_results: # No results found
        st.warning("No resumes found that meet your search criteria with at least 50% relevance.")



with recommend_tab:
    st.header("Get Recommendations for a Job")
    all_resumes_for_dashboard = [Resume(**r) for r in retrieve_resumes()]
    if all_resumes_for_dashboard:
        with st.expander("Show Database Overview"): display_dashboard(all_resumes_for_dashboard)
    all_jobs = retrieve_jobs()
    if not all_jobs: st.warning("No jobs found. Please add a job in the 'Manage Jobs' tab to get recommendations.")
    else:
        job_options = {job['job_title']: job['id'] for job in all_jobs}
        selected_job_title = st.selectbox("Select a Job to find candidates for:", options=list(job_options.keys()))
        num_recommendations = st.slider("Show Top N Recommendations", 1, 20, 5)
        if st.button("Get Recommendations"):
            with st.spinner("Analyzing resumes..."):
                selected_job_obj = next((job for job in all_jobs if job['id'] == job_options[selected_job_title]), None)
                if selected_job_obj:
                    print(f"🤖 Generating recommendations for job: {selected_job_obj['job_title']}")
                    recommendations = get_recommendations(Job(**selected_job_obj), [Resume(**r) for r in retrieve_resumes()])
                    print(f"✓ Generated {len(recommendations)} recommendations")
                    st.success(f"Displaying top {num_recommendations} matches for '{selected_job_obj['job_title']}':")
                    for i, rec in enumerate(recommendations[:num_recommendations]):
                        details, exp_str = rec['details'], f"{rec['details'].experience} years" if rec['details'].experience is not None else "N/A"
                        with st.expander(f"**{i+1}. {details.name or 'N/A'}** (Match Score: {rec['score']:.2f} | Exp: {exp_str})"):
                            st.markdown(f"**Email:** {details.email or 'N/A'}")
                            st.markdown(f"**Contact:** {details.contact_number or 'N/A'}")
                            st.markdown(f"**Location:** {details.location or 'N/A'}")
                            st.markdown(f"**Skills:** {', '.join(details.skills) if details.skills else 'N/A'}")

with jobs_tab:
    st.header("Manage Job Listings")
    with st.form("add_job_form", clear_on_submit=True):
        st.subheader("Add a New Job")
        job_title = st.text_input("Job Title")
        required_experience = st.number_input("Required Experience (years)", 0)
        education_level = st.text_input("Required Education")
        job_type = st.selectbox("Job Type", ['full-time', 'part-time', 'contract', 'internship', 'remote', 'hybrid'])
        skills = st.text_area("Required Skills (comma-separated)")
        job_description_text = st.text_area("Full Job Description")
        if st.form_submit_button("Add Job"):
            if job_title and skills:
                add_job_to_db(Job(job_title=job_title, required_experience=required_experience, education_level=education_level, job_type=job_type, skills=[s.strip() for s in skills.split(',')], job_description_text=job_description_text).model_dump())

                print(f"✓ Job added: {job_title} ({job_type})")
                st.success(f"Job '{job_title}' added successfully!"); st.rerun()
            else: st.warning("Job Title and Skills are required."); print("✗ Failed to add job: Missing Job Title or Skills")
    st.divider()
    st.subheader("Existing Jobs")
    all_jobs_display = retrieve_jobs()
    if not all_jobs_display: st.info("No jobs currently in the database.")
    else:
        for job in all_jobs_display:
            with st.expander(f"{job['job_title']} ({job['job_type']})"):
                st.markdown(f"**Experience:** {job['required_experience']} years")
                st.markdown(f"**Skills:** {', '.join(job['skills'])}")
                if st.button("Delete Job", key=f"del_{job['id']}"):
                    delete_job_from_db(job['id']); print(f"✓ Deleted job: {job['job_title']}"); st.rerun()

with resumes_tab:
    st.header("Manage All Resumes")

    # This logic is shared with the search tab, so initialize here for robustness
    if "selected_resume_text_to_view" not in st.session_state:
        st.session_state.selected_resume_text_to_view = None
    if "selected_resume_name_to_view" not in st.session_state:
        st.session_state.selected_resume_name_to_view = None
    
    # --- Display for Selected Resume ---
    if st.session_state.selected_resume_text_to_view:
        st.subheader(f"Full Resume Content: {st.session_state.selected_resume_name_to_view}")
        st.text_area("", st.session_state.selected_resume_text_to_view, height=400, key="resume_viewer_in_manage_tab")
        if st.button("Close View", key="close_resume_viewer_in_manage_tab"):
            st.session_state.selected_resume_text_to_view = None
            st.session_state.selected_resume_name_to_view = None
            st.rerun()
        st.divider()

    st.subheader("Existing Resumes in Database")
    all_resumes_display = retrieve_resumes()
    if not all_resumes_display:
        st.info("No resumes currently in the database.")
    else:
        for resume in all_resumes_display:
            with st.expander(f"{resume.get('name', 'N/A')} - {resume.get('job_title', 'N/A') or 'No Title'}"):
                st.markdown(f"**File:** `{resume['file_name']}`")
                st.markdown(f"**Title:** {resume.get('job_title', 'N/A')}") # Added Title field
                st.markdown(f"**Experience:** {resume.get('experience', 'N/A')} years | **Location:** {resume.get('location', 'N/A')}")
                
                # Using columns for buttons
                col1, col2, _ = st.columns([1, 1, 5])
                with col1:
                    if st.button("View", key=f"view_resume_{resume['id']}"):
                        st.session_state.selected_resume_text_to_view = resume.get('resume_text', 'Resume text not found.')
                        st.session_state.selected_resume_name_to_view = resume.get('name', 'N/A')
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"del_resume_{resume['id']}"):
                        # Clear view if the deleted resume was selected
                        if st.session_state.get('selected_resume_name_to_view') == resume.get('name', 'N/A'):
                            st.session_state.selected_resume_text_to_view = None
                            st.session_state.selected_resume_name_to_view = None
                        
                        delete_resume_from_db(resume['id'])
                        print(f"✓ Deleted resume from database: {resume['file_name']}")
                        file_to_delete = os.path.join("uploads", resume['file_name'])
                        if os.path.exists(file_to_delete):
                            os.remove(file_to_delete)
                            print(f"✓ Deleted file: {resume['file_name']}")
                        st.rerun()
    st.divider()
    st.subheader("⚠️ Danger Zone")
    with st.expander("Delete All Resumes and Uploaded Files"):
        st.warning("This action is irreversible and will delete ALL resumes from the database and all files from the 'uploads' folder.")
        if st.checkbox("I understand and want to delete all resumes", key="confirm_delete_all_resumes") and st.button("Delete ALL Resumes"):
            from database import delete_all_resumes
            print("🗑️  Starting deletion of all resumes...")
            deleted_count = delete_all_resumes()
            print(f"✓ Deleted {deleted_count} resumes from database")
            files_deleted = 0
            for filename in os.listdir("uploads"):
                file_path = os.path.join("uploads", filename)
                try:
                    if os.path.isfile(file_path): os.remove(file_path); files_deleted += 1
                except Exception as e: st.error(f"Error deleting file {filename}: {e}"); print(f"✗ Error deleting file {filename}: {e}")
            print(f"✓ Deleted {files_deleted} files from 'uploads' folder")
            st.success(f"Deleted {deleted_count} resumes from database and {files_deleted} files from 'uploads' folder.")
            st.rerun()