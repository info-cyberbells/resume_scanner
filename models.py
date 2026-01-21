from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

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
    resume_text: Optional[str] = None

class Job(BaseModel):
    id: Optional[str] = None
    job_title: str
    required_experience: int = Field(..., ge=0)
    education_level: Optional[str] = None
    job_type: Literal['full-time', 'part-time', 'contract', 'internship', 'remote', 'hybrid']
    skills: List[str] = []
    job_description_text: str
