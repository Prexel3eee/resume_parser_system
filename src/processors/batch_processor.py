import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Generator, Optional
import psutil
import gc
from pathlib import Path
import json
from tqdm import tqdm
import logging
from datetime import datetime

from src.core.resume_parser import ResumeParser
from config.settings import settings

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Memory-efficient batch processing with monitoring"""
    
    def __init__(self, 
                 batch_size: int = None,
                 num_workers: int = None,
                 max_memory_percent: int = None):
        self.batch_size = batch_size or settings.BATCH_SIZE
        self.num_workers = num_workers or settings.NUM_WORKERS
        self.max_memory_percent = max_memory_percent or settings.MAX_MEMORY_PERCENT
        self.parser = None  # Initialize in worker process
        
    def _init_worker(self):
        """Initialize parser in worker process"""
        global parser
        parser = ResumeParser()
        
    def _process_single(self, file_path: str) -> Optional[Dict]:
        """Process single resume in worker"""
        try:
            result = parser.parse_resume_file(file_path)
            return result if result else None
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
            
    def check_memory(self):
        """Monitor and manage memory usage"""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > self.max_memory_percent:
            logger.warning(f"High memory usage: {memory_percent}%")
            gc.collect()
            
            # If still high, reduce workers
            if memory_percent > 90:
                self.num_workers = max(1, self.num_workers - 1)
                logger.warning(f"Reduced workers to {self.num_workers}")
                
    def process_batch_generator(self, 
                               file_paths: List[str]) -> Generator[Dict, None, None]:
        """Process files as a generator to save memory"""
        
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=self._init_worker
        ) as executor:
            
            # Process in chunks
            for i in range(0, len(file_paths), self.batch_size):
                batch = file_paths[i:i + self.batch_size]
                
                # Submit batch
                futures = {
                    executor.submit(self._process_single, fp): fp 
                    for fp in batch
                }
                
                # Process results as they complete
                for future in tqdm(
                    as_completed(futures), 
                    total=len(futures),
                    desc=f"Batch {i//self.batch_size + 1}"
                ):
                    try:
                        result = future.result()
                        if result:
                            yield result
                    except Exception as e:
                        logger.error(f"Future failed: {e}")
                    
                    # Check memory after each result
                    self.check_memory()
                    
                # Force garbage collection after each batch
                gc.collect()
                
    def _convert_extracted_values(self, obj):
        """Convert ExtractedValue objects to their values for JSON serialization"""
        if hasattr(obj, 'value'):
            return obj.value
        elif isinstance(obj, dict):
            return {k: self._convert_extracted_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_extracted_values(item) for item in obj]
        return obj

    def process_to_file(self, 
                       file_paths: List[str],
                       output_file: Path) -> Dict:
        """Process files and save results to JSON files"""
        start_time = datetime.now()
        total_files = len(file_paths)
        processed = 0
        failed = 0
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create individual files directory
        individual_dir = output_file.parent / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        # Ensure output file has .json extension
        if not output_file.suffix:
            output_file = output_file.with_suffix('.json')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('[\n')  # Start JSON array
                first = True
                
                # Process in batches
                for result in self.process_batch_generator(file_paths):
                    if result:
                        # Convert ExtractedValue objects to their values
                        result = self._convert_extracted_values(result)
                        
                        # Save individual file
                        try:
                            # Get filename from original file path
                            original_filename = Path(result.get('file_path', 'unknown')).stem
                            if original_filename == 'unknown':
                                # Try to construct filename from name fields
                                first_name = result.get('first_name', '').strip()
                                last_name = result.get('last_name', '').strip()
                                designation = result.get('designation', '').strip()
                                if first_name and last_name:
                                    original_filename = f"{first_name} {last_name}"
                                    if designation:
                                        original_filename += f" - {designation}"
                                else:
                                    original_filename = f"resume_{processed + 1}"
                            
                            # Clean filename to be safe for filesystem
                            safe_filename = "".join(c for c in original_filename if c.isalnum() or c in (' ', '-', '_')).strip()
                            individual_file = individual_dir / f"{safe_filename}.json"
                            
                            # Write individual file
                            with open(individual_file, 'w', encoding='utf-8') as ind_f:
                                json.dump(result, ind_f, indent=2)
                        except Exception as e:
                            logger.error(f"Error writing individual file for {result.get('file_path', 'unknown')}: {e}")
                        
                        # Write to combined file
                        if not first:
                            f.write(',\n')
                        json.dump(result, f, indent=2)
                        first = False
                        processed += 1
                    else:
                        failed += 1
                        
                    # Log progress
                    if (processed + failed) % 100 == 0:
                        logger.info(
                            f"Progress: {processed + failed}/{total_files} "
                            f"({(processed + failed)/total_files*100:.1f}%)"
                        )
                
                f.write('\n]')  # End JSON array
        except Exception as e:
            logger.error(f"Error writing to output file {output_file}: {e}")
            raise
        
        # Calculate metrics
        duration = (datetime.now() - start_time).total_seconds()
        metrics = {
            "total_files": total_files,
            "processed": processed,
            "failed": failed,
            "success_rate": processed / total_files * 100 if total_files > 0 else 0,
            "processing_time": duration,
            "files_per_second": total_files / duration if duration > 0 else 0,
            "output_file": str(output_file),
            "individual_files_dir": str(individual_dir)
        }
        
        logger.info(f"Processing complete: {metrics}")
        return metrics 