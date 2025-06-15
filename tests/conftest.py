import pytest
import os
from pathlib import Path
from src.core.data_models import ResumeData
from src.core.resume_parser import ResumeParser
from src.core.document_reader import DocumentReader

@pytest.fixture
def test_data_dir():
    """Fixture to provide test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_resume_path(test_data_dir):
    """Fixture to provide path to sample resume"""
    return test_data_dir / "sample_resume.pdf"

@pytest.fixture
def sample_resume_text():
    """Fixture to provide sample resume text"""
    return """
    John Doe
    Software Engineer
    john.doe@example.com
    (555) 123-4567
    
    Location: New York, NY 10001
    Work Authorization: US Citizen
    
    Experience:
    Senior Software Engineer
    Tech Company Inc.
    2018 - Present
    
    Skills:
    Python, Java, Machine Learning, NLP, Docker
    
    Education:
    Bachelor of Science in Computer Science
    University of Technology
    2014 - 2018
    """

@pytest.fixture
def resume_parser():
    """Fixture to provide ResumeParser instance"""
    return ResumeParser()

@pytest.fixture
def document_reader():
    """Fixture to provide DocumentReader instance"""
    return DocumentReader()

@pytest.fixture
def sample_resume_data():
    """Fixture to provide sample ResumeData instance"""
    return ResumeData(
        first_name="John",
        last_name="Doe",
        primary_email="john.doe@example.com",
        phone="(555) 123-4567",
        city="New York",
        state="NY",
        zip="10001",
        work_authority="US Citizen",
        designation="Software Engineer",
        experience="5",
        skills="Python, Java, Machine Learning, NLP, Docker",
        confidence_score=0.95
    )

@pytest.fixture
def mock_ats_response():
    """Fixture to provide mock ATS API response"""
    return {
        "status": "success",
        "message": "Resume processed successfully",
        "data": {
            "resume_id": "12345",
            "processed_at": "2024-03-20T10:00:00Z"
        }
    } 