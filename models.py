from typing import List, Optional, Literal
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

class Company(BaseModel):
    id: Optional[str] = None
    company_name: str
    email: EmailStr
    password: str
    phone_no: Optional[str] = None
    staff: Optional[int] = None

class User(BaseModel):
    username: str
    email: EmailStr
    password: str
    user_type: Literal['seeker', 'recruiter']
    company_id: Optional[str] = None  # Link to a company if user_type is 'recruiter'
    id: Optional[str] = None

class Resume(BaseModel):
    id: Optional[str] = None
    file_name: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    contact_number: Optional[str] = None
    education: List[str] = []
    job_title: Optional[str] = None
    skills: List[str] = []
    experience: Optional[int] = None
    location: Optional[str] = None
    updated_at: Optional[datetime] = None
    status: Optional[str] = "New"
    notes: Optional[str] = ""
    visible_to_companies: List[str] = [] # Replaces visible_to_recruiters
    hidden_from: List[str] = [] # New field
    resume_text: Optional[str] = None

class Job(BaseModel):
    id: Optional[str] = None
    job_title: str
    required_experience: int = Field(..., ge=0)
    education_level: Optional[str] = None
    job_type: Literal['full-time', 'part-time', 'contract', 'internship', 'remote', 'hybrid']
    skills: List[str] = []
    company_id: Optional[str] = None # Replaces recruiter_id
    job_description_text: str

class UnifiedSignupRequest(BaseModel):
    user_type: Literal['seeker', 'recruiter']
    username: str
    email: EmailStr
    password: str
    company_name: Optional[str] = None
    company_email: Optional[EmailStr] = None
    company_phone_no: Optional[str] = None
    company_staff: Optional[int] = None
