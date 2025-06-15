from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime

class ResumeData(BaseModel):
    # Personal Information
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    
    # Contact Information
    primary_email: EmailStr = ""
    secondary_email: Optional[EmailStr] = ""
    phone: str = ""
    secondary_phone_number: str = ""
    
    # Location
    city: str = ""
    state: str = ""
    zip: str = ""
    
    # Professional Information
    designation: str = ""
    experience: str = ""  # In years
    skills: str = ""  # Comma-separated
    work_authority: str = ""
    
    # Resume Metadata
    resume_link: str = ""
    raw_resume: str = ""
    source_by: str = ""
    tax_term: str = ""
    
    # Processing Metadata
    payrate: str = ""
    comment: str = "--"
    added_by: str = ""
    processed_at: Optional[datetime] = None
    confidence_score: Optional[float] = None
    
    @validator('phone', 'secondary_phone_number')
    def validate_phone(cls, v):
        # Clean and validate phone numbers
        if v:
            cleaned = ''.join(filter(str.isdigit, v))
            if len(cleaned) >= 10:
                return cleaned
        return v

    def to_dict(self) -> dict:
        """Convert to the required JSON structure"""
        return {
            "name": f"{self.first_name} {self.middle_name} {self.last_name}".strip(),
            "email": self.primary_email,
            "secondary_email": self.secondary_email,
            "city": self.city,
            "state": self.state,
            "zip": self.zip,
            "work_authority": self.work_authority,
            "phone": self.phone,
            "resume_link": self.resume_link,
            "raw_resume": self.raw_resume,
            "tax_term": self.tax_term,
            "source_by": self.source_by,
            "skills": self.skills,
            "designation": self.designation,
            "experience": self.experience
        }

class ProcessingMetrics(BaseModel):
    total_files: int = 0
    processed: int = 0
    failed: int = 0
    avg_processing_time: float = 0.0
    memory_usage: float = 0.0 