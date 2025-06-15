import sys
import os
from pathlib import Path
import logging
import time
import json
from typing import List, Dict, Any
import shutil
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.processors.two_pass_processor import TwoPassProcessor
from src.utils.quality_monitor import QualityMonitor
from src.core.resume_parser import ResumeParser
from config.logging_config import setup_logging
from config.settings import settings
from src.core.document_reader import DocumentReader

# Setup logging
setup_logging()

logger = logging.getLogger(__name__)

def get_processed_files() -> set:
    """Get set of already processed files"""
    processed_files = set()
    processed_log = Path('logs/processed_files.txt')
    if processed_log.exists():
        with open(processed_log, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f)
    return processed_files

def mark_file_as_processed(file_path: str):
    """Mark a file as processed"""
    processed_log = Path('logs/processed_files.txt')
    with open(processed_log, 'a', encoding='utf-8') as f:
        f.write(f"{file_path}\n")

def move_to_processed(file_path: str, success: bool = True):
    """Move processed file to the appropriate directory"""
    try:
        source_path = Path(file_path)
        if not source_path.exists():
            logger.warning(f"Source file {file_path} does not exist")
            return

        # Determine destination directory
        if success:
            dest_dir = Path('data/processed')
        else:
            dest_dir = Path('data/errors')
        
        # Create destination directory if it doesn't exist
        dest_dir.mkdir(exist_ok=True)
        
        # Handle duplicate filenames
        dest_path = dest_dir / source_path.name
        if dest_path.exists():
            # Add timestamp to filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dest_path = dest_dir / f"{source_path.stem}_{timestamp}{source_path.suffix}"
        
        # Move the file
        shutil.move(str(source_path), str(dest_path))
        logger.info(f"Moved {file_path} to {dest_dir.name} directory")
        
        # If successful, mark as processed
        if success:
            mark_file_as_processed(file_path)
    except Exception as e:
        logger.error(f"Error moving file {file_path}: {e}")

def test_single_resume(resume_path: str) -> Dict[str, Any]:
    """Test quality extraction on a single resume"""
    print(f"\nTesting quality extraction on: {resume_path}")
    
    try:
        # Initialize processor
        processor = TwoPassProcessor()
        
        # Process the resume
        start_time = time.time()
        result = processor.process_resume_file(resume_path)
        processing_time = time.time() - start_time
        
        if result:
            print(f"\nProcessing time: {processing_time:.2f} seconds")
            
            # Helper function to extract value from ExtractedValue objects
            def extract_value(value):
                if hasattr(value, 'value'):
                    return value.value
                return value
            
            # Convert to the desired JSON format
            json_output = {
                "first_name": extract_value(result.get('first_name')) or None,
                "last_name": extract_value(result.get('last_name')) or None,
                "primary_email": extract_value(result.get('primary_email')) or None,
                "secondary_email": extract_value(result.get('secondary_email')) or None,
                "phone": extract_value(result.get('phone')) or None,
                "city": extract_value(result.get('city')) or None,
                "state": extract_value(result.get('state')) or None,
                "zip": extract_value(result.get('zip')) or None,
                "work_authority": extract_value(result.get('work_authority')) or None,
                "resume_link": result.get('resume_link') or None,
                "raw_resume": extract_value(result.get('raw_resume')) or None,
                "tax_term": extract_value(result.get('tax_term')) or None,
                "source_by": extract_value(result.get('source_by')) or None,
                "skills": clean_skills(result.get('skills', {})),
                "designation": extract_value(result.get('designation')) or None,
                "experience": extract_value(result.get('experience')) or None,
                "education": extract_value(result.get('education')) or None,
                "certifications": extract_value(result.get('certifications')) or None
            }
            
            # Save JSON to file
            output_dir = Path("data/output")
            output_dir.mkdir(exist_ok=True)
            json_filename = Path(resume_path).stem + ".json"
            json_path = output_dir / json_filename
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            print(f"Saved extracted JSON to {json_path}")
            
            # Move file to processed directory
            move_to_processed(resume_path, success=True)
            
            return json_output
        else:
            print(f"\nFailed to process resume: {resume_path}")
            # Move file to errors directory
            move_to_processed(resume_path, success=False)
            return None
    except Exception as e:
        logger.error(f"Error processing resume {resume_path}: {e}")
        # Move file to errors directory
        move_to_processed(resume_path, success=False)
        return None

def test_batch_resumes(resume_dir: str, batch_size: int = 5) -> None:
    """Test quality extraction on a batch of resumes"""
    print(f"\nTesting batch processing on directory: {resume_dir}")
    
    # Get list of resume files
    resume_files = []
    for ext in ['.pdf', '.doc', '.docx', '.txt']:
        resume_files.extend(list(Path(resume_dir).glob(f'*{ext}')))
    
    if not resume_files:
        print(f"No resume files found in {resume_dir}")
        return
    
    # Filter out already processed files
    processed_files = get_processed_files()
    resume_files = [f for f in resume_files if str(f) not in processed_files]
    
    if not resume_files:
        print("All files have been processed")
        return
    
    print(f"Found {len(resume_files)} new resume files to process")
    
    # Initialize processor
    processor = TwoPassProcessor()
    all_skills = set()
    
    # Process in batches
    for i in range(0, len(resume_files), batch_size):
        batch = resume_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} resumes)")
        
        start_time = time.time()
        results = processor.process_resumes([str(f) for f in batch])
        batch_time = time.time() - start_time
        
        print(f"Batch processing time: {batch_time:.2f} seconds")
        print(f"Average time per resume: {batch_time/len(batch):.2f} seconds")
        
        # Helper function to extract value from ExtractedValue objects
        def get_value(obj):
            if hasattr(obj, 'value'):
                return obj.value
            return obj
        
        # Process results
        for result, file_path in zip(results, batch):
            if result:
                print(f"\nResume: {get_value(result.get('resume_link', 'unknown'))}")
                print(f"Confidence: {result.get('confidence_score', 0):.2f}")
                
                # Convert to the desired JSON format
                json_output = {
                    "name": f"{get_value(result.get('first_name', ''))} {get_value(result.get('last_name', ''))}".strip(),
                    "email": get_value(result.get('primary_email', '')),
                    "secondary_email": get_value(result.get('secondary_email', '')),
                    "city": get_value(result.get('city', '')),
                    "state": get_value(result.get('state', '')),
                    "zip": get_value(result.get('zip', '')),
                    "work_authority": get_value(result.get('work_authority', '')),
                    "phone": get_value(result.get('phone', '')),
                    "resume_link": get_value(result.get('resume_link', '')),
                    "raw_resume": get_value(result.get('raw_resume', '')),
                    "tax_term": get_value(result.get('tax_term', '')),
                    "source_by": get_value(result.get('source_by', '')),
                    "skills": get_value(result.get('skills', '')),
                    "designation": get_value(result.get('designation', '')),
                    "experience": get_value(result.get('experience', ''))
                }
                
                # Print the JSON output
                json_output["skills"] = remove_empty(json_output["skills"])
                print("\nExtracted JSON:")
                print(json.dumps(json_output, indent=2))
                print("\nRAW RESUME TEXT:\n", json_output["raw_resume"])
                print("---")
                # Save JSON to file
                output_dir = "data/output"
                os.makedirs(output_dir, exist_ok=True)
                # Use the original input filename instead of resume_link
                json_filename = os.path.splitext(os.path.basename(str(file_path)))[0] + ".json"
                json_path = os.path.join(output_dir, json_filename)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False)
                print(f"Saved extracted JSON to {json_path}")
                # Merge all unique skills
                skills = json_output["skills"]
                if isinstance(skills, str):
                    for skill in skills.split(","):
                        skill = skill.strip()
                        if skill:
                            all_skills.add(skill)
                elif isinstance(skills, dict):
                    def collect_skills(d):
                        for v in d.values():
                            if isinstance(v, dict):
                                collect_skills(v)
                            elif isinstance(v, list):
                                for skill in v:
                                    if isinstance(skill, str) and skill:
                                        all_skills.add(skill.strip())
                    collect_skills(skills)
    # Print all unique skills found in the batch
    if all_skills:
        print("\nAll unique skills found in batch:")
        print(sorted(all_skills))

def remove_empty(d):
    """Remove empty values from dictionary or list"""
    if isinstance(d, dict):
        return {k: remove_empty(v) for k, v in d.items() if remove_empty(v) not in [None, {}, []]}
    elif isinstance(d, list):
        return [remove_empty(v) for v in d if remove_empty(v) not in [None, {}, []]]
    else:
        return d

def extract_value(value):
    """Extract clean value from ExtractedValue or string"""
    if hasattr(value, 'value'):
        return value.value
    return value

def clean_skills(skills_dict):
    """Clean skills dictionary to only include skill names."""
    if not skills_dict:
        return {}
    
    cleaned_skills = {}
    for category, skills in skills_dict.items():
        if isinstance(skills, list):
            # If skills is a list, just use the skill names
            cleaned_skills[category] = [skill["name"] if isinstance(skill, dict) else skill for skill in skills]
        elif isinstance(skills, dict):
            # If skills is a dict, extract just the skill names
            cleaned_skills[category] = [skill["name"] for skill in skills.values() if isinstance(skill, dict) and "name" in skill]
    
    return cleaned_skills

def main():
    """Main function to test quality extraction"""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting test script...")

    # Initialize processor
    processor = TwoPassProcessor()

    # Ensure output directory exists
    output_dir = Path(settings.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file in the input directory
    input_dir = Path(settings.INPUT_DIR)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return

    for file_path in input_dir.glob("*"):
        if file_path.is_file():
            logger.info(f"Processing {file_path}")
            try:
                # Process the file
                result = processor.process_resume_file(str(file_path))
                
                if result:
                    # Create output filename
                    output_file = output_dir / f"{file_path.stem}.json"
                    
                    # Convert result to JSON-serializable format with null for empty values
                    json_data = {
                        'first_name': extract_value(result.get('first_name')) or None,
                        'last_name': extract_value(result.get('last_name')) or None,
                        'primary_email': extract_value(result.get('primary_email')) or None,
                        'secondary_email': extract_value(result.get('secondary_email')) or None,
                        'phone': extract_value(result.get('phone')) or None,
                        'city': extract_value(result.get('city')) or None,
                        'state': extract_value(result.get('state')) or None,
                        'zip': extract_value(result.get('zip')) or None,
                        'work_authority': extract_value(result.get('work_authority')) or None,
                        'resume_link': result.get('resume_link') or None,
                        'raw_resume': extract_value(result.get('raw_resume')) or None,
                        'tax_term': extract_value(result.get('tax_term')) or None,
                        'source_by': extract_value(result.get('source_by')) or None,
                        'skills': clean_skills(result.get('skills', {})),
                        'designation': extract_value(result.get('designation')) or None,
                        'experience': extract_value(result.get('experience')) or None
                    }
                    
                    # Save to JSON file
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Successfully processed {file_path}")
                    logger.info(f"Extracted information: {result}")
                else:
                    logger.error(f"Failed to process {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main()
    print("Test script completed.") 