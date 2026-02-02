import os
import logging
from pymongo import MongoClient
from bson.objectid import ObjectId
from models import Job
from datetime import datetime
from database import resume_collection # Import resume_collection

# Setup logging
logger = logging.getLogger(__name__)

# --- Database Configuration ---
MONGO_DETAILS = "mongodb+srv://infocyberbells:URgCpmEAgksetiiI@cyberbellsmongocluster.vy8xm.mongodb.net/resumes?retryWrites=true&w=majority"
client = MongoClient(MONGO_DETAILS)
database = client.resumes  # Using the same database
job_collection = database.get_collection("job_collection")
application_collection = database.get_collection("job_applications")  # New collection for applications


# --- Helper Functions ---

def job_helper(job) -> dict:
    """
    Converts a job document from MongoDB to a dictionary.
    """
    return {
        "id": str(job["_id"]),
        "job_title": job.get("job_title"),
        "required_experience": job.get("required_experience"),
        "education_level": job.get("education_level"),
        "job_type": job.get("job_type"),
        "skills": job.get("skills", []),
        "company_id": job.get("company_id"),
        "job_description_text": job.get("job_description_text"),
    }


# --- CRUD Operations ---

def retrieve_jobs():
    """
    Retrieves all jobs present in the database.
    """
    jobs = []
    for job in job_collection.find():
        jobs.append(job_helper(job))
    return jobs


def add_job(job_data: dict) -> dict:
    """
    Adds a new job to the database.
    
    Args:
        job_data (dict): Job data to insert
        
    Returns:
        dict: The inserted job with generated ID
        
    Raises:
        Exception: If database insertion fails
    """
    try:
        if not job_data:
            raise ValueError("Job data cannot be empty")
        # Ensure company_id is present if it's part of the model
        if "company_id" not in job_data:
            logger.warning("company_id missing in job_data for add_job")
            # For now, we'll let it be None if not provided, and expect upstream to provide it.
        job = job_collection.insert_one(job_data)
        new_job = job_collection.find_one({"_id": job.inserted_id})
        return job_helper(new_job)
    except Exception as e:
        logger.error(f"Error adding job: {e}")
        raise


def update_job(id: str, job_data: dict) -> dict:
    """
    Updates an existing job in the database.
    
    Args:
        id (str): The job ID to update
        job_data (dict): Updated job data
        
    Returns:
        dict: The updated job
        
    Raises:
        Exception: If job not found or update fails
    """
    try:
        if not id or not job_data:
            raise ValueError("Job ID and data cannot be empty")
        result = job_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": job_data}
        )
        if result.matched_count == 0:
            raise ValueError(f"Job with ID {id} not found")
        updated_job = job_collection.find_one({"_id": ObjectId(id)})
        return job_helper(updated_job)
    except Exception as e:
        logger.error(f"Error updating job {id}: {e}")
        raise


def delete_job(id: str) -> bool:
    """
    Deletes a job from the database.
    
    Args:
        id (str): The job ID to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        if not id:
            logger.warning("delete_job called with empty ID")
            return False
            
        job = job_collection.find_one({"_id": ObjectId(id)})
        if job:
            job_collection.delete_one({"_id": ObjectId(id)})
            return True
        else:
            logger.warning(f"Job with ID {id} not found")
            return False
    except Exception as e:
        logger.error(f"Error deleting job {id}: {e}")
        return False


def submit_job_application(job_id: str, user_id: str, resume_data: dict) -> dict:
    """
    Submits a job application with the candidate's resume.
    """
    try:
        if not job_id or not user_id:
            raise ValueError("Job ID and User ID are required")

        # 1. Retrieve company_id from job_collection
        job_doc = job_collection.find_one({"_id": ObjectId(job_id)})
        if not job_doc:
            raise ValueError(f"Job with ID {job_id} not found.")
        company_id = job_doc.get("company_id")

        if not company_id:
            logger.warning(f"Job {job_id} has no company_id. Resume visibility won't be changed.")
        else:
            # 2. Update the original seeker's resume in resume_collection
            # Add company_id to the 'visible_to_companies' array on the seeker's resume
            seeker_resume_id = resume_data.get("id")
            if seeker_resume_id:
                resume_collection.update_one(
                    {"_id": ObjectId(seeker_resume_id)},
                    {"$addToSet": {"visible_to_companies": company_id}}
                )
                logger.info(f"Resume {seeker_resume_id} made visible to company {company_id} due to application.")
            else:
                logger.warning(f"Resume data for job_id {job_id} is missing original ID. Cannot update visibility.")
        
        application_data = {
            "job_id": job_id,
            "user_id": user_id,
            "resume": resume_data,
            "applied_at": datetime.utcnow(),
            "status": "applied"
        }
        
        result = application_collection.insert_one(application_data)
        application = application_collection.find_one({"_id": result.inserted_id})
        
        return {
            "id": str(application["_id"]),
            "job_id": application["job_id"],
            "user_id": application["user_id"],
            "applied_at": str(application["applied_at"]),
            "status": application["status"]
        }
    except Exception as e:
        logger.error(f"Error submitting job application: {e}")
        raise

def update_application_status(application_id: str, new_status: str, notes: str) -> bool:
    """
    Updates the status and notes of a job application.
    """
    try:
        result = application_collection.update_one(
            {"_id": ObjectId(application_id)},
            {"$set": {"status": new_status, "notes": notes}}
        )
        return result.matched_count > 0
    except Exception as e:
        logger.error(f"Error updating application {application_id}: {e}")
        return False


def retrieve_applications_by_user(user_id: str) -> list:
    """
    Retrieves all applications submitted by a specific user (job seeker).
    """
    try:
        applications = []
        for app in application_collection.find({"user_id": user_id}):
            # Also fetch the job details for each application
            job_doc = job_collection.find_one({"_id": ObjectId(app["job_id"])})
            job_title = job_doc.get("job_title", "N/A") if job_doc else "N/A"
            applications.append({
                "id": str(app["_id"]),
                "job_id": app["job_id"],
                "job_title": job_title, # Include job title for display
                "user_id": app["user_id"],
                "resume_id": str(app.get("resume", {}).get("id")),
                "applied_at": str(app.get("applied_at", "")),
                "status": app.get("status", "applied"),
                "notes": app.get("notes", "") # Include recruiter notes
            })
        return applications
    except Exception as e:
        logger.error(f"Error retrieving applications for user {user_id}: {e}")
        return []


def get_job_applications(job_id: str) -> list:
    """
    Retrieves all applications for a specific job.
    
    Args:
        job_id (str): The job ID
        
    Returns:
        list: List of applications with candidate details
    """
    try:
        applications = []
        for app in application_collection.find({"job_id": job_id}):
            applications.append({
                "id": str(app["_id"]),
                "job_id": app["job_id"],
                "user_id": app["user_id"],
                "resume_id": str(app.get("resume", {}).get("id")),
                "resume": app.get("resume", {}),
                "candidate_name": app.get("resume", {}).get("name", "N/A"),
                "candidate_email": app.get("resume", {}).get("email", "N/A"),
                "candidate_phone": app.get("resume", {}).get("contact_number", "N/A"),
                "candidate_skills": app.get("resume", {}).get("skills", []),
                "candidate_experience": app.get("resume", {}).get("experience", "N/A"),
                "applied_at": str(app.get("applied_at", "")),
                "status": app.get("status", "applied")
            })
        return applications
    except Exception as e:
        logger.error(f"Error retrieving job applications: {e}")
        return []
