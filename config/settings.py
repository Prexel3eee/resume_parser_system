from pydantic_settings import BaseSettings
from typing import Optional
import multiprocessing
from pathlib import Path
import os

class Settings(BaseSettings):
    # Processing
    BATCH_SIZE: int = 500
    NUM_WORKERS: int = min(4, multiprocessing.cpu_count() - 1)
    MAX_MEMORY_PERCENT: int = 80
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    INPUT_DIR: Path = BASE_DIR / "data" / "input"
    OUTPUT_DIR: Path = BASE_DIR / "data" / "output"
    ERROR_DIR: Path = BASE_DIR / "data" / "errors"
    LOG_DIR: Path = BASE_DIR / "data" / "logs"
    
    # Database
    DATABASE_URL: str = "sqlite:///data/resumes.db"
    REDIS_URL: Optional[str] = None
    
    # API
    ATS_API_URL: Optional[str] = None
    ATS_API_KEY: Optional[str] = None
    API_BATCH_SIZE: int = 100
    
    # Models
    SPACY_MODEL: str = "en_core_web_trf"
    NER_MODEL: str = "dslim/bert-base-NER"
    SKILL_MODEL: str = "jjzha/jobbert-base-cased"
    
    # OCR Settings
    ENABLE_OCR: bool = True
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    OCR_LANGUAGE: str = "eng"
    OCR_CONFIG: str = "--oem 3 --psm 6"
    TESSERACT_PATH: str = os.environ.get('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    USE_GPU: bool = False
    OCR_PREPROCESSING: bool = True
    OCR_DENOISE: bool = True
    OCR_ADAPTIVE_THRESHOLD: bool = True
    
    # Document Processing
    MAX_DOCUMENT_SIZE: int = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS: list = ["pdf", "docx", "doc", "txt"]
    PDF_EXTRACTION_METHODS: list = ["pdfplumber", "pdfminer", "pypdf2"]
    DOCX_EXTRACTION_METHODS: list = ["mammoth", "python-docx"]
    
    # Performance
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600
    
    class Config:
        env_file = ".env"

settings = Settings() 