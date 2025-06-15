import pytest
from src.core.resume_parser import ResumeParser
from src.core.data_models import ResumeData

def test_parse_resume_basic(resume_parser, sample_resume_text):
    """Test basic resume parsing functionality"""
    # Create a temporary file with sample text
    with open("temp_resume.txt", "w") as f:
        f.write(sample_resume_text)
    
    # Parse the resume
    result = resume_parser.parse_resume("temp_resume.txt")
    
    # Verify basic fields
    assert result is not None
    assert isinstance(result, ResumeData)
    assert result.first_name == "John"
    assert result.last_name == "Doe"
    assert result.primary_email == "john.doe@example.com"
    assert result.phone == "(555) 123-4567"
    assert result.city == "New York"
    assert result.state == "NY"
    assert result.zip == "10001"
    assert result.work_authority == "US Citizen"
    assert result.designation == "Software Engineer"
    assert "Python" in result.skills
    assert "Java" in result.skills
    assert result.confidence_score > 0.8

def test_parse_resume_missing_fields(resume_parser):
    """Test parsing resume with missing fields"""
    incomplete_text = """
    John Doe
    john.doe@example.com
    """
    
    with open("temp_incomplete.txt", "w") as f:
        f.write(incomplete_text)
    
    result = resume_parser.parse_resume("temp_incomplete.txt")
    
    assert result is not None
    assert result.first_name == "John"
    assert result.last_name == "Doe"
    assert result.primary_email == "john.doe@example.com"
    assert result.confidence_score < 0.8  # Lower confidence due to missing fields

def test_parse_resume_invalid_file(resume_parser):
    """Test parsing non-existent file"""
    result = resume_parser.parse_resume("non_existent_file.txt")
    assert result is None

def test_extract_contact_info(resume_parser):
    """Test contact information extraction"""
    text = """
    Contact:
    Email: test@example.com
    Phone: (555) 999-8888
    Secondary Email: test2@example.com
    """
    
    resume_data = ResumeData()
    resume_parser._extract_contact_info(text, resume_data)
    
    assert resume_data.primary_email == "test@example.com"
    assert resume_data.secondary_email == "test2@example.com"
    assert resume_data.phone == "(555) 999-8888"

def test_extract_professional_info(resume_parser):
    """Test professional information extraction"""
    text = """
    Current Position: Senior Developer
    Experience: 8 years
    Work Authorization: H1B
    """
    
    resume_data = ResumeData()
    resume_parser._extract_professional_info(text, resume_data)
    
    assert resume_data.designation == "Senior Developer"
    assert resume_data.experience == "8"
    assert resume_data.work_authority == "H1B"

def test_extract_skills(resume_parser):
    """Test skills extraction"""
    text = """
    Technical Skills:
    - Python
    - Java
    - Machine Learning
    - Docker
    """
    
    resume_data = ResumeData()
    resume_parser._extract_skills(text, resume_data)
    
    skills = resume_data.skills.lower()
    assert "python" in skills
    assert "java" in skills
    assert "machine learning" in skills
    assert "docker" in skills

def test_calculate_confidence(resume_parser):
    """Test confidence score calculation"""
    resume_data = ResumeData(
        first_name="John",
        last_name="Doe",
        primary_email="john@example.com",
        phone="(555) 123-4567",
        skills="Python, Java"
    )
    
    resume_parser._calculate_confidence(resume_data, used_ocr=False)
    assert 0.7 <= resume_data.confidence_score <= 1.0
    
    # Test with OCR
    resume_parser._calculate_confidence(resume_data, used_ocr=True)
    assert 0.6 <= resume_data.confidence_score <= 0.9 