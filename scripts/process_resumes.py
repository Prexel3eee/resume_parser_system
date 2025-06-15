#!/usr/bin/env python3
"""
Main script to process all resumes
Usage: python process_resumes.py --input-dir data/input --output-dir data/output
"""

import click
import asyncio
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import List
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processors.batch_processor import BatchProcessor
from src.api.ats_client import ATSAPIClient
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data/logs/processing_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--input-dir', default='data/input', help='Input directory with resumes')
@click.option('--output-dir', default='data/output', help='Output directory for JSON')
@click.option('--batch-size', default=500, help='Batch size for processing')
@click.option('--num-workers', default=4, help='Number of parallel workers')
@click.option('--push-to-api', is_flag=True, help='Push results to ATS API')
def main(input_dir: str, 
         output_dir: str, 
         batch_size: int,
         num_workers: int,
         push_to_api: bool):
    """Process all resumes in the input directory"""
    
    start_time = datetime.now()
    logger.info(f"Starting resume processing at {start_time}")
    
    # Setup
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all resume files
    extensions = ['.pdf', '.docx', '.doc', '.txt']
    resume_files = []
    
    for ext in extensions:
        resume_files.extend(input_path.glob(f'**/*{ext}'))
    
    resume_files = [str(f) for f in resume_files]
    logger.info(f"Found {len(resume_files)} resume files")
    
    if not resume_files:
        logger.error("No resume files found!")
        return
    
    # Initialize processor
    processor = BatchProcessor(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Process resumes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"resumes_{timestamp}.json"
    metrics = processor.process_to_file(resume_files, output_file)
    
    # Generate summary report
    summary = {
        "processing_time": metrics["processing_time"],
        "total_files": metrics["total_files"],
        "processed": metrics["processed"],
        "failed": metrics["failed"],
        "success_rate": metrics["success_rate"],
        "files_per_second": metrics["files_per_second"],
        "output_file": str(output_file),
        "timestamp": timestamp
    }
    
    summary_file = output_path / f"processing_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Push to API if requested
    if push_to_api and settings.ATS_API_URL:
        logger.info("Pushing results to ATS API...")
        
        api_client = ATSAPIClient(
            api_url=settings.ATS_API_URL,
            api_key=settings.ATS_API_KEY
        )
        
        # Run async push
        api_results = asyncio.run(api_client.push_all_resumes(resume_files))
        logger.info(f"API push results: {api_results}")
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main() 