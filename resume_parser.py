import re
import pdfplumber
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from models import Resume

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import en_core_web_sm
    nlp = en_core_web_sm.load()

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
    """Extracts the contact number from the spaCy Doc."""
    matcher = Matcher(nlp.vocab)
    pattern = [{"TEXT": {"REGEX": r"(\(?\d{3}\)?[-.\s]?)?(\d{3}[-.\s]?\d{4})"}}]
    matcher.add("PHONE_NUMBER", [pattern])
    
    matches = matcher(doc)
    for _, start, end in matches:
        return doc[start:end].text
    return None

def _extract_skills(full_text: str) -> list:
    """Extracts skills by searching the full text of the resume."""
    skills = set()
    full_text_lower = full_text.lower()
    
    for skill_item in SKILLS_LIST:
        # Create a more flexible regex pattern for multi-word skills
        pattern = r"\b" + re.escape(skill_item).replace(r"\ ", r"[\s-]?") + r"\b"
        
        if re.search(pattern, full_text_lower):
            skills.add(skill_item.lower())
            
    return list(skills)


def _extract_education(doc) -> list:
    """Extracts education from the spaCy Doc."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in EDUCATION_KEYWORDS]
    matcher.add("EDUCATION", patterns)
    
    matches = matcher(doc)
    education = set()
    for _, start, end in matches:
        education.add(doc[start:end].text)
    return list(education)

def _extract_job_title(doc) -> str | None:
    """Extracts job title from the spaCy Doc."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in JOB_TITLES_LIST]
    matcher.add("JOB_TITLE", patterns)
    
    matches = matcher(doc)
    if matches:
        _, start, end = matches[0]
        return doc[start:end].text
    return None


def _extract_experience(doc) -> int:
    """Extracts total years of experience from the spaCy Doc."""
    matcher = Matcher(nlp.vocab)
    pattern = [
        {"LIKE_NUM": True},
        {"OP": "+", "TEXT": {"IN": ["+", "years", "year", "yrs", "yr"]}},
        {"OP": "*", "LOWER": {"IN": ["of", "in", "with"]}},
        {"OP": "*", "LOWER": "experience"}
    ]
    matcher.add("EXPERIENCE", [pattern])
    
    matches = matcher(doc)
    years = []
    for _, start, end in matches:
        span = doc[start:end]
        for token in span:
            if token.like_num:
                try:
                    years.append(int(token.text))
                    break
                except ValueError:
                    continue
    
    return max(years) if years else 0

def parse_resume(file_path: str) -> Resume:
    """
    Parses a resume from a PDF file and extracts key information.
    """
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
