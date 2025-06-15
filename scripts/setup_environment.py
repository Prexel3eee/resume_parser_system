#!/usr/bin/env python3
"""
Setup script to download required models and create necessary directories
"""

import subprocess
import sys
from pathlib import Path
import logging
from transformers import AutoModel, AutoTokenizer
import spacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    base_dir = Path(__file__).parent.parent
    directories = [
        base_dir / "data" / "input",
        base_dir / "data" / "output",
        base_dir / "data" / "errors",
        base_dir / "data" / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_spacy_model():
    """Download spaCy model"""
    try:
        logger.info("Downloading spaCy model...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_trf"],
            check=True
        )
        logger.info("SpaCy model downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading spaCy model: {e}")
        sys.exit(1)

def download_huggingface_models():
    """Download Hugging Face models"""
    models = [
        "dslim/bert-base-NER",
        "jjzha/jobbert-base-cased",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    for model in models:
        try:
            logger.info(f"Downloading {model}...")
            AutoModel.from_pretrained(model)
            AutoTokenizer.from_pretrained(model)
            logger.info(f"{model} downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading {model}: {e}")
            sys.exit(1)

def verify_installation():
    """Verify all required components are installed"""
    try:
        # Check spaCy
        nlp = spacy.load("en_core_web_trf")
        logger.info("SpaCy model loaded successfully")
        
        # Check transformers
        from transformers import pipeline
        ner = pipeline("token-classification", model="dslim/bert-base-NER")
        logger.info("Transformers models loaded successfully")
        
        # Check OCR
        import pytesseract
        logger.info("Tesseract OCR is available")
        
        # Check PDF processing
        import pdfplumber
        logger.info("PDF processing libraries are available")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        sys.exit(1)

def main():
    """Main setup function"""
    logger.info("Starting environment setup...")
    
    # Create directories
    create_directories()
    
    # Download models
    download_spacy_model()
    download_huggingface_models()
    
    # Verify installation
    verify_installation()
    
    logger.info("Environment setup completed successfully!")

if __name__ == "__main__":
    main() 