import os
import ast # This import is crucial for ast.literal_eval
from pymongo import MongoClient
from bson.objectid import ObjectId
from models import Resume

# --- Database Configuration ---
MONGO_DETAILS = "mongodb+srv://infocyberbells:URgCpmEAgksetiiI@cyberbellsmongocluster.vy8xm.mongodb.net/resumes?retryWrites=true&w=majority"
client = MongoClient(MONGO_DETAILS)
database = client.get_database()  
resume_collection = database.get_collection("resume_collection")


# --- Helper Functions ---

def resume_helper(resume) -> dict:
    """
    Converts a resume document from MongoDB to a dictionary.
    Handles potential string representation of education field.
    """
    education_data = resume.get("education", [])
    if isinstance(education_data, str):
        try:
            # Safely evaluate string representation of a list back to a list
            education_data = ast.literal_eval(education_data)
            # Ensure it's a list even if evaluation results in something else
            if not isinstance(education_data, list):
                education_data = [] 
        except (ValueError, SyntaxError):
            # Fallback if string is not a valid list representation
            education_data = [] 
    elif not isinstance(education_data, list):
        # Ensure it's a list if it's not a string but also not a list
        education_data = []

    return {
        "id": str(resume["_id"]),
        "file_name": resume.get("file_name"),
        "resume_text": resume.get("resume_text"),
        "skills": resume.get("skills", []),
        "experience": resume.get("experience"),
        "name": resume.get("name"),
        "email": resume.get("email"),
        "contact_number": resume.get("contact_number"),
        "education": education_data,
        "job_title": resume.get("job_title"),
        "location": resume.get("location"),
        "status": resume.get("status") or "New",
        "notes": resume.get("notes", ""),
    }


# --- CRUD Operations ---

def retrieve_resumes():
    """
    Retrieves all resumes present in the database.
    """
    resumes = []
    for resume in resume_collection.find():
        resumes.append(resume_helper(resume))
    return resumes


def add_resume(resume_data: dict) -> dict:
    """
    Adds a new resume to the database.
    """
    resume = resume_collection.insert_one(resume_data)
    new_resume = resume_collection.find_one({"_id": resume.inserted_id})
    return resume_helper(new_resume)

def update_resume_workflow(id: str, status: str, notes: str):
    """
    Updates the status and notes of a resume.
    """
    try:
        resume_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"status": status, "notes": notes}}
        )
        return True
    except Exception:
        return False


def delete_resume(id: str):
    """
    Deletes a resume from the database.
    """
    try:
        resume = resume_collection.find_one({"_id": ObjectId(id)})
        if resume:
            resume_collection.delete_one({"_id": ObjectId(id)})
            return True
        return False
    except Exception:
        return False


def delete_resume_by_filename(filename: str):
    """
    Deletes all resume entries from the database by filename.
    """
    result = resume_collection.delete_many({"file_name": filename})
    return result.deleted_count > 0


def delete_all_resumes():
    """
    Deletes all resumes from the database.
    """
    result = resume_collection.delete_many({})
    return result.deleted_count
