import re
import pdfplumber
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from models import Resume

# Load the spaCy model lazily (on first use)
nlp = None

def _get_nlp():
    """Lazy-load the spaCy model on first use"""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import en_core_web_sm
            nlp = en_core_web_sm.load()
    return nlp

# --- Keyword Lists for PhraseMatcher ---
SKILLS_LIST = sorted(list(set([
    "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "go", "swift",
    "react", "angular", "vue.js", "node.js", "express.js", "next.js",
    "fastapi", "django", "flask", "ruby on rails",
    "mongodb", "postgresql", "mysql", "sqlite", "redis", "cassandra", "graphql",
    "docker", "kubernetes", "jenkins", "git", "svn", "ci/cd",
    "aws", "azure", "google cloud platform", "gcp", "heroku", "netlify",
    "machine learning", "data science", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "natural language processing", "nlp", "computer vision", "opencv", "power bi", "data cleaning", "data cleansing", "data wrangling", "data preparation",
    "agile", "scrum", "jira", "tableau",
    "html", "css", "sass", "less",
    "rest", "soap", "json", "xml", "api", "sql",
    "excel", "jupyter notebook", "data visualization", "data analytics",
    "figma", "adobe xd", "wireframing", "prototyping", "user research", "ui design", "ux design"
])))

EDUCATION_KEYWORDS = [
    "b.sc", "m.sc", "b.tech", "m.tech", "ph.d", "bachelor", "master", "degree",
    "bachelor of science", "master of science", "bachelor of technology", "master of technology",
    "doctor of philosophy"
]

JOB_TITLES_LIST = [
    "software engineer", "data scientist", "product manager", "backend developer", 
    "frontend developer", "full stack developer", "full stack python developer", "devops engineer", "qa engineer",
    "machine learning engineer", "data analyst", "project manager"
]

INDIAN_CITIES = [
    "bangalore", "bengaluru", "mumbai", "pune", "chennai", "hyderabad", "delhi", 
    "kolkata", "ahmedabad", "noida", "gurgaon",  "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
]

INDIAN_STATES = [
    "karnataka", "maharashtra", "tamil nadu", "telangana", "delhi", "west bengal",
    "gujarat", "uttar pradesh", "haryana"
]

# Map common aliases / spellings to canonical names
CITY_ALIASES = {
    "bengaluru": "Bangalore",
    "bangalore": "Bangalore",
    "mumbai": "Mumbai",
    "chennai": "Chennai",
    "hyderabad": "Hyderabad",
    "delhi": "Delhi",
    "kolkata": "Kolkata",
    "ahmedabad": "Ahmedabad",
    "noida": "Noida",
    "gurgaon": "Gurgaon",
    "pune": "Pune",
}

# Map cities to their respective states for broader location matching
CITY_STATE_MAP = {
    "bangalore": "karnataka",
    "bengaluru": "karnataka",
    "mumbai": "maharashtra",
    "pune": "maharashtra",
    "chennai": "tamil nadu",
    "hyderabad": "telangana",
    "delhi": "delhi", # Delhi is also a state/UT
    "noida": "uttar pradesh",
    "gurgaon": "haryana",
    "kolkata": "west bengal",
    "ahmedabad": "gujarat",
}

def _extract_name(doc) -> str | None:
    """Extracts the name from the spaCy Doc (prioritizing top of resume and filtering by skills)."""
    # Process only the first few lines for name extraction (e.g., first 500 chars)
    top_text_doc = nlp(doc.text[:500])
    for ent in top_text_doc.ents:
        if ent.label_ == "PERSON":
            # Simple heuristic: prioritize multi-word names
            if len(ent.text.split()) > 1:
                return ent.text
            # If it's a single word, check if it's also a skill (case-insensitive)
            if ent.text.lower() not in [skill.lower() for skill in SKILLS_LIST]:
                return ent.text
    return None

def _extract_email(doc) -> str | None:
    """Extracts the email from the spaCy Doc."""
    for token in doc:
        if token.like_email:
            return token.text
    return None

def _extract_location(doc) -> str | None:
    """Extract the location from the resume text.

    Strategy (in order):
    1. Look for explicit location phrases ("Location:", "Based in", "Current location", "Address") using regex.
    2. Use spaCy NER to find a `GPE` entity.
    3. Fallback to searching known city/state keywords (with word boundaries).
    4. Normalize common aliases (e.g., Bengaluru -> Bangalore).
    """
    text = doc.text or ""

    # 1) Explicit location patterns
    patterns = [r"location[:\-\s]*([A-Za-z0-9 ,.-]+)",
                r"based in[:\-\s]*([A-Za-z0-9 ,.-]+)",
                r"current location[:\-\s]*([A-Za-z0-9 ,.-]+)",
                r"address[:\-\s]*([A-Za-z0-9 ,.-]+)",
                r"lives in[:\-\s]*([A-Za-z0-9 ,.-]+)"]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).split('\n')[0].strip()
            # Trim trailing separators and common words
            candidate = re.split(r",|;|-|\(|\)", candidate)[0].strip()
            if candidate:
                return candidate

    # 2) NER using spaCy - prefer GPE entities
    skill_tokens = {s.lower() for s in SKILLS_LIST}
    jobtitle_tokens = {jt.lower() for jt in JOB_TITLES_LIST}
    for ent in doc.ents:
        if ent.label_ == "GPE":
            cand = ent.text.strip()
            if not cand:
                continue
            key = cand.lower()
            # Skip entities that look like skills or job titles (avoid techs being treated as places)
            if key in skill_tokens or key in jobtitle_tokens:
                continue
            if key in CITY_ALIASES:
                return CITY_ALIASES[key]
            return cand

    # 3) Keyword search for cities/states using word boundaries to avoid substrings
    text_lower = text.lower()
    all_places = sorted(set(INDIAN_CITIES + INDIAN_STATES), key=lambda x: -len(x))
    for place in all_places:
        if re.search(r"\b" + re.escape(place) + r"\b", text_lower):
            if place in CITY_ALIASES:
                return CITY_ALIASES[place]
            return place.capitalize()

    return None
def _extract_contact_number(doc) -> str | None:
    """Extracts the contact number from the spaCy Doc, supporting Indian and international formats."""
    text = doc.text or ""
    
    # Try to find phone numbers using regex patterns
    # Pattern for Indian phone: +91-XXXXX-XXXXX or 91XXXXXXXXXX or +91 XXXXX XXXXX
    patterns = [
        r"\+91[\s-]?\d{5}[\s-]?\d{5}",  # +91-XXXXX-XXXXX
        r"\+91[\s-]?\d{4}[\s-]?\d{6}",  # +91-XXXX-XXXXXX
        r"\d{10}",  # 10-digit without country code
        r"\+1\s?\d{3}[\s-]?\d{3}[\s-]?\d{4}",  # US format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None

def _extract_skills(full_text: str) -> list:
    """Extracts skills by searching the full text of the resume with flexible matching."""
    skills = set()
    full_text_lower = full_text.lower()
    
    for skill_item in SKILLS_LIST:
        # Create a more flexible regex pattern for multi-word skills
        # Allow hyphens and spaces to be optional/interchangeable
        escaped_skill = re.escape(skill_item)
        pattern = r"\b" + escaped_skill.replace(r"\ ", r"[\s\-]?") + r"\b"
        
        if re.search(pattern, full_text_lower, re.IGNORECASE):
            skills.add(skill_item.lower())
    
    # Also try to extract skills from common phrases like "Skills:" sections
    skills_section_patterns = [
        r"skills?[:\-]?\s*([^\n]+)",
        r"technical skills?[:\-]?\s*([^\n]+)",
        r"core competencies[:\-]?\s*([^\n]+)",
    ]
    
    for pattern in skills_section_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            skills_text = match.group(1)
            # Split by comma, slash, or other delimiters
            for item in re.split(r"[,/;]", skills_text):
                item_clean = item.strip().lower()
                # Check if any known skill matches
                for skill in SKILLS_LIST:
                    if skill.lower() in item_clean or item_clean in skill.lower():
                        skills.add(skill.lower())
            
    return list(skills)


def _extract_education(doc) -> list:
    """Extracts education from the spaCy Doc and text patterns."""
    text = doc.text or ""
    education = set()
    
    # Try PhraseMatcher for known education keywords
    try:
        nlp_obj = _get_nlp()
        if nlp_obj:
            matcher = PhraseMatcher(nlp_obj.vocab, attr="LOWER")
            patterns = [nlp_obj.make_doc(ed_text) for ed_text in EDUCATION_KEYWORDS]
            matcher.add("EDUCATION", patterns)
            
            matches = matcher(doc)
            for _, start, end in matches:
                education.add(doc[start:end].text)
    except Exception:
        pass
    
    # Also search for education patterns in text
    edu_patterns = [
        r"education[:\-]?\s*([^\n]+)",
        r"(?:bachelor|master|phd|b\.?tech|m\.?tech|b\.?sc|m\.?sc)[^\n]*",
    ]
    
    for pattern in edu_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            edu_item = match.group(1) if match.lastindex else match.group(0)
            edu_item = edu_item.strip()
            if edu_item and len(edu_item) < 150:
                education.add(edu_item)
    
    return list(education)

def _extract_job_title(doc) -> str | None:
    """Extracts job title from the spaCy Doc and text patterns."""
    text = doc.text or ""
    
    # Try regex patterns from job titles list
    for job_title in JOB_TITLES_LIST:
        pattern = r"\b" + re.escape(job_title) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return job_title
    
    # Try to extract from "Experience" or "Work Experience" sections
    exp_pattern = r"(?:work experience|experience)[:\-]?\s*([^\n]+)"
    match = re.search(exp_pattern, text, re.IGNORECASE | re.MULTILINE)
    if match:
        # Extract the first line which usually contains the job title
        job_line = match.group(1).strip()
        # Clean up common markers
        job_line = re.sub(r"^[-•*]", "", job_line).strip()
        if job_line and len(job_line) < 100:  # Reasonable length for a job title
            return job_line
    
    return None


def _extract_experience(doc) -> int:
    """Extracts total years of experience from the spaCy Doc and text patterns."""
    text = doc.text or ""
    years = []
    
    # Try regex patterns first for flexible matching
    # Pattern: "X years" or "X+ years" or "X-Y years"
    patterns = [
        r"(\d+)\+?\s*(?:years?|yrs?|year|yr)",  # "5 years", "5+ years", "5yr"
        r"experience[:\-]?\s*(\d+)\s*(?:years?|yrs?)",  # "Experience: 5 years"
        r"total\s+experience[:\-]?\s*(\d+)",  # "Total experience: 5"
        r"\(\s*(\d+)\s*(?:years?|yrs?)\s*\)",  # "(5 years)"
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                year_count = int(match.group(1))
                if 0 < year_count < 70:  # Reasonable experience range
                    years.append(year_count)
            except (ValueError, IndexError):
                pass
    
    # Also parse year ranges like "2020-2024"
    date_pattern = r"(\d{4})\s*[–\-]\s*(\d{4})"
    matches = re.finditer(date_pattern, text)
    for match in matches:
        try:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            if end_year > start_year > 1900 and end_year < 2100:
                years.append(end_year - start_year)
        except ValueError:
            pass
    
    return max(years) if years else 0

def parse_resume(file_path: str) -> Resume:
    """
    Parses a resume from a PDF file and extracts key information.
    """
    nlp = _get_nlp()  # Get the model (lazy-loaded on first call)
    
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    
    doc = nlp(full_text)

    # --- Extract Information ---
    name = _extract_name(doc)
    email = _extract_email(doc)
    contact_number = _extract_contact_number(doc)
    skills = _extract_skills(full_text)
    education = _extract_education(doc)
    job_title = _extract_job_title(doc)
    location = _extract_location(doc)
    experience = _extract_experience(doc)


    return Resume(
        name=name,
        email=email,
        contact_number=contact_number,
        skills=skills,
        education=education,
        job_title=job_title,
        experience=experience,
        location=location,
        resume_text=full_text,
        file_name=file_path.split("/")[-1]
    )
