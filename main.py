from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import os
from database import add_resume, retrieve_resumes
from models import Resume, JobDescription
from resume_parser import parse_resume
from recommendation import get_recommendations

app = FastAPI()

@app.post("/upload_resume/", response_model=Resume)
async def upload_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
    
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.read())
    
    try:
        resume_data = parse_resume(file_location)
        resume_id = await add_resume(resume_data.dict())
        return {**resume_data.dict(), "id": resume_id}
    finally:
        os.remove(file_location)

@app.post("/recommend_resumes/", response_model=List[dict])
async def recommend_resumes(job_description: JobDescription):
    resumes = await retrieve_resumes()
    recommendations = get_recommendations(job_description, resumes)
    return recommendations
