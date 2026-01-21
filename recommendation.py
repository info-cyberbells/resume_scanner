from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models import Job, Resume
from typing import List
from difflib import SequenceMatcher

# --- Scoring Configuration ---
WEIGHTS = {
    'skills': 0.50,
    'experience': 0.25,
    'education': 0.15,
    'job_title': 0.10,
}

def _calculate_skill_score(job_skills: List[str], resume_skills: List[str]) -> float:
    """
    Calculates a skill match score using TF-IDF and cosine similarity.
    """
    if not job_skills or not resume_skills:
        return 0.0

    # The vectorizer needs strings, so we join the lists of skills
    job_skills_str = " ".join(job_skills)
    resume_skills_str = " ".join(resume_skills)

    vectorizer = TfidfVectorizer().fit_transform([job_skills_str, resume_skills_str])
    vectors = vectorizer.toarray()
    
    similarity = cosine_similarity(vectors)
    # The similarity of the two items is in the off-diagonal
    score = similarity[0][1]
    
    return score * 100

def _calculate_experience_score(required_exp: int, resume_exp: int | None) -> float:
    """
    Calculates an experience match score.
    """
    if resume_exp is None or required_exp == 0:
        return 0.0 # No basis for comparison

    if resume_exp >= required_exp:
        # Bonus for exceeding requirement, capped at 20%
        bonus = min((resume_exp - required_exp) / required_exp, 0.20)
        return 100 * (1 + bonus)
    else:
        # Score is proportional to the experience they have vs what's required
        return (resume_exp / required_exp) * 100

def _calculate_education_score(required_edu: str | None, resume_edu: List[str]) -> float:
    """
    Calculates an education match score based on keyword matching.
    Improved to handle variations in education terminology.
    """
    if not required_edu or not resume_edu:
        return 0.0

    required_edu_lower = required_edu.lower()
    
    # Define common education level keywords
    bachelor_keywords = ["bachelor", "b.sc", "b.tech", "undergraduate"]
    master_keywords = ["master", "m.sc", "m.tech", "postgraduate"]
    phd_keywords = ["ph.d", "doctorate"]

    # Check for direct match or keyword match
    for edu_entry in resume_edu:
        edu_entry_lower = edu_entry.lower()
        if required_edu_lower in edu_entry_lower:
            return 100.0
        
        # More flexible matching for common degrees
        if any(keyword in required_edu_lower for keyword in bachelor_keywords) and \
           any(keyword in edu_entry_lower for keyword in bachelor_keywords):
            return 100.0
        
        if any(keyword in required_edu_lower for keyword in master_keywords) and \
           any(keyword in edu_entry_lower for keyword in master_keywords):
            return 100.0
            
        if any(keyword in required_edu_lower for keyword in phd_keywords) and \
           any(keyword in edu_entry_lower for keyword in phd_keywords):
            return 100.0
            
    return 0.0

def _calculate_job_title_score(job_title: str, resume_job_title: str | None) -> float:
    """
    Calculates a job title similarity score.
    """
    if not job_title or not resume_job_title:
        return 0.0
    
    similarity = SequenceMatcher(None, job_title.lower(), resume_job_title.lower()).ratio()
    return similarity * 100

def get_recommendations(job: Job, resumes: List[Resume]) -> List[dict]:
    """
    Get resume recommendations based on a weighted, multi-factor scoring model.
    """
    if not resumes:
        return []

    recommendations = []
    for resume in resumes:
        # --- Calculate all sub-scores ---
        skill_score = _calculate_skill_score(job.skills, resume.skills)
        experience_score = _calculate_experience_score(job.required_experience, resume.experience)
        education_score = _calculate_education_score(job.education_level, resume.education)
        job_title_score = _calculate_job_title_score(job.job_title, resume.job_title)

        # --- Calculate Final Weighted Score ---
        final_score = (
            (skill_score * WEIGHTS['skills']) +
            (experience_score * WEIGHTS['experience']) +
            (education_score * WEIGHTS['education']) +
            (job_title_score * WEIGHTS['job_title'])
        )
        
        # We cap the final score at 100 for consistency
        final_score = min(final_score, 100.0)

        recommendations.append({
            "score": final_score,
            "details": resume
        })

    # Sort by the final score in descending order
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return recommendations
