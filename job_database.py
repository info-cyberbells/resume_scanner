import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from models import Job

# --- Database Configuration ---
MONGO_DETAILS = "mongodb+srv://infocyberbells:URgCpmEAgksetiiI@cyberbellsmongocluster.vy8xm.mongodb.net/resumes?retryWrites=true&w=majority"
client = MongoClient(MONGO_DETAILS)
database = client.resumes  # Using the same database
job_collection = database.get_collection("job_collection")


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
    """
    job = job_collection.insert_one(job_data)
    new_job = job_collection.find_one({"_id": job.inserted_id})
    return job_helper(new_job)




def delete_job(id: str):
    """
    Deletes a job from the database.
    """
    try:
        job = job_collection.find_one({"_id": ObjectId(id)})
        if job:
            job_collection.delete_one({"_id": ObjectId(id)})
            return True
        return False
    except Exception:
        return False
