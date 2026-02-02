from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv # Import load_dotenv
from typing import List, Optional
import os
import logging
import bcrypt
from passlib.context import CryptContext
from pymongo import MongoClient
from bson.objectid import ObjectId

# Import models and functions
from models import Resume, Job, User, Company, UnifiedSignupRequest
from resume_parser import parse_resume
from recommendation import get_recommendations

# Import database helpers
from database import (
    add_resume, retrieve_resumes, retrieve_resumes_by_user, 
    update_resume_workflow, delete_resume, resume_helper, resume_collection
)
from job_database import (
    retrieve_jobs, submit_job_application, get_job_applications, 
    update_application_status, retrieve_applications_by_user
)
from user_database import (
    add_company, retrieve_company_by_email, retrieve_company_by_name, add_user, retrieve_user_by_email,
    database as db # Use the same db connection
)

load_dotenv() # Load environment variables from .env file
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

logging.basicConfig(level=logging.INFO)

# --- API Key Authentication ---
# In a production environment, this should be loaded from environment variables or a secure configuration management system.
# For demonstration purposes, we'll use an environment variable with a fallback.
API_KEY_SECRET = os.getenv("FASTAPI_API_KEY", "your-super-secret-api-key") # Default value for testing

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY_SECRET == "your-super-secret-api-key":
        logging.warning("Using default API key. Please set FASTAPI_API_KEY environment variable in production.")
    if not x_api_key or x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return x_api_key
    
def hash_password(password: str) -> str:
    truncated = _truncate_password_to_72(password)
    if truncated is None:
        truncated = ""
    b = truncated.encode("utf-8", errors="ignore")
    if len(b) > 72:
        b = b[:72]
    hashed = bcrypt.hashpw(b, bcrypt.gensalt())
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        truncated = _truncate_password_to_72(plain_password)
        if truncated is None:
            truncated = ""
        b = truncated.encode("utf-8", errors="ignore")
        if len(b) > 72:
            b = b[:72]
        return bcrypt.checkpw(b, hashed_password.encode("utf-8"))
    except Exception:
        logging.exception("Error verifying password")
        return False

def _truncate_password_to_72(password: str) -> str:
    if password is None:
        return password
    try:
        b = password.encode("utf-8")
    except Exception:
        b = str(password).encode("utf-8", errors="ignore")
    if len(b) <= 72:
        return password
    truncated_bytes = b[:72]
    return truncated_bytes.decode("utf-8", errors="ignore")

# --- User Signup (Seeker/Recruiter) ---
@app.post("/auth/signup", response_model=dict, dependencies=[Depends(verify_api_key)])
async def signup(user: User):
    try:
        existing_user = retrieve_user_by_email(user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        if user.user_type == 'recruiter':
            if not user.company_id:
                raise HTTPException(status_code=400, detail="Recruiter must be associated with a company_id")
            try:
                user.company_id = ObjectId(user.company_id)
            except Exception: # Catch any exception during ObjectId conversion
                raise HTTPException(status_code=400, detail="Invalid company_id format.")

        hashed_password = hash_password(user.password)
        user_data = user.dict()
        user_data["password"] = hashed_password
        
        new_user = add_user(user_data)
        return {"message": "User created successfully", "user_id": new_user["id"]}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in signup endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# --- Company Authentication ---

@app.post("/auth/company/signup", response_model=dict, dependencies=[Depends(verify_api_key)])
async def company_signup(company: Company):
    try:
        # Check for existing company by email OR name
        existing_company_by_email = retrieve_company_by_email(company.email)
        if existing_company_by_email:
            raise HTTPException(status_code=400, detail=f"Company with email '{company.email}' already exists.")
        
        existing_company_by_name = retrieve_company_by_name(company.company_name)
        if existing_company_by_name:
            raise HTTPException(status_code=400, detail=f"Company with name '{company.company_name}' already exists.")

        hashed_password = hash_password(company.password)
        company_data = company.dict()
        company_data["password"] = hashed_password
        
        new_company = add_company(company_data)
        return {"message": "Company created successfully", "company_id": new_company["id"]}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in company_signup endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/company/login", response_model=dict, dependencies=[Depends(verify_api_key)])
async def company_login(email: str, password: str):
    try:
        company = retrieve_company_by_email(email)
        if not company or not verify_password(password, company.get("password", "")):
            raise HTTPException(status_code=401, detail="Invalid company email or password")

        return {
            "message": "Company login successful",
            "company_id": str(company["_id"]),
            "company_name": company["company_name"],
            "email": company["email"],
            "user_type": "company"
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in company_login endpoint")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/auth/login", response_model=dict, dependencies=[Depends(verify_api_key)])
async def login(email: str, password: str):
    try:
        user = retrieve_user_by_email(email)
        if not user or not verify_password(password, user.get("password", "")):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        return {
            "message": "Login successful",
            "user_id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"],
            "user_type": user["user_type"],
            "company_id": user.get("company_id")
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in login endpoint")
        raise HTTPException(status_code=500, detail=str(e))

# (The rest of the file remains the same for now)

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...), user_id: str = None):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        file_location = f"temp_{file.filename}"
        try:
            with open(file_location, "wb+") as file_object:
                content = await file.read()
                file_object.write(content)
            
            resume_data = parse_resume(file_location)
            resume_dict = resume_data.dict()
            resume_dict["user_id"] = user_id
            resume_result = add_resume(resume_dict)
            
            return resume_result
        
        finally:
            if os.path.exists(file_location):
                os.remove(file_location)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in upload_resume endpoint")
        raise HTTPException(status_code=500, detail=f"Resume upload failed: {str(e)}")

@app.post("/recommend_resumes/", response_model=List[dict])
async def recommend_resumes(job: Job, user_id: str = None):
    try:
        if user_id:
            resumes = retrieve_resumes_by_user(user_id)
        else:
            resumes = retrieve_resumes()
        
        recommendations = get_recommendations(job.description, resumes)
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in recommend_resumes endpoint")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.get("/resumes/", response_model=List[dict])
async def list_resumes(user_id: str = None):
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id query parameter is required")
        resumes = retrieve_resumes_by_user(user_id)
        return resumes
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error listing resumes")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resumes/{id}")
async def get_resume(id: str):
    try:
        from bson.objectid import ObjectId
        doc = resume_collection.find_one({"_id": ObjectId(id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Resume not found")
        return resume_helper(doc)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error getting resume")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/resumes/by_company/{company_id}", response_model=List[dict], dependencies=[Depends(verify_api_key)])
async def list_resumes_by_company(company_id: str):
    try:
        resumes = retrieve_resumes_by_company(company_id)
        return resumes
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error listing resumes by company")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/resumes/{id}")
async def update_resume(id: str, payload: dict):
    try:
        status = payload.get("status")
        notes = payload.get("notes")
        ok = update_resume_workflow(id, status, notes)
        if not ok:
            raise HTTPException(status_code=404, detail="Resume not found or update failed")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error updating resume")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/resumes/{id}")
async def delete_resume_endpoint(id: str):
    try:
        ok = delete_resume(id)
        if not ok:
            raise HTTPException(status_code=404, detail="Resume not found or delete failed")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error deleting resume")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/db", response_model=dict)
async def debug_db():
    try:
        collections = db.list_collection_names()
        return {"ok": True, "collections": collections}
    except Exception as e:
        logging.exception("Error in debug_db endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/", response_model=List[dict])
async def list_all_jobs():
    try:
        jobs = retrieve_jobs()
        return jobs
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error retrieving jobs")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobs/", response_model=dict, dependencies=[Depends(verify_api_key)])
async def create_job(job: Job):
    try:
        # Validate that company_id is provided for a new job listing
        if not job.company_id:
            raise HTTPException(status_code=400, detail="Job must be associated with a company_id.")
            
        new_job = add_job(job.dict())
        return {"message": "Job created successfully", "job_id": new_job["id"]}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in create_job endpoint")
        raise HTTPException(status_code=500, detail=f"Job creation failed: {str(e)}")

@app.post("/jobs/{job_id}/apply")
async def apply_for_job(job_id: str, user_id: str = None):
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        resumes = retrieve_resumes_by_user(user_id)
        if not resumes:
            raise HTTPException(status_code=400, detail="You must upload a resume before applying")
        
        resume_data = resumes[0]
        application = submit_job_application(job_id, user_id, resume_data)
        
        return {
            "message": "Successfully applied for the job!",
            "application": application
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error applying for job")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/applications")
async def get_job_applicants(job_id: str):
    try:
        applications = get_job_applications(job_id)
        return applications
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error retrieving job applications")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/applications/{application_id}/status", response_model=dict)
async def update_application_status_endpoint(
    application_id: str,
    new_status: str,
    notes: str = ""
):
    try:
        ok = update_application_status(application_id, new_status, notes)
        if not ok:
            raise HTTPException(status_code=404, detail="Application not found or update failed")
        
        return {"message": "Application status updated successfully", "ok": True}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error updating application status")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/applications/seeker/{user_id}", response_model=List[dict])
async def list_seeker_applications(user_id: str):
    try:
        applications = retrieve_applications_by_user(user_id)
        return applications
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error listing seeker applications")
        raise HTTPException(status_code=500, detail=str(e))