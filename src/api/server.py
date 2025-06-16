from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import tempfile
import shutil
import os
from typing import List
import json
from datetime import datetime

from src.core.resume_parser import ResumeParser
from config.settings import settings

app = FastAPI(title="Resume Parser API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize parser
parser = ResumeParser()

@app.get("/")
async def root():
    return {"status": "online", "service": "resume-parser"}

@app.post("/parse")
async def parse_resume(file: UploadFile = File(...)):
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # Parse resume
        result = parser.parse_resume_file(temp_path)
        
        if not result:
            raise HTTPException(status_code=400, detail="Failed to parse resume")
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

@app.post("/parse-batch")
async def parse_batch(files: List[UploadFile] = File(...)):
    results = []
    temp_files = []
    
    try:
        # Save all files temporarily
        for file in files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
            shutil.copyfileobj(file.file, temp_file)
            temp_files.append(temp_file.name)
        
        # Process each file
        for temp_path in temp_files:
            try:
                result = parser.parse_resume_file(temp_path)
                if result:
                    results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "filename": Path(temp_path).name
                })
                
        return {
            "total_files": len(files),
            "processed": len(results),
            "results": results
        }
        
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 