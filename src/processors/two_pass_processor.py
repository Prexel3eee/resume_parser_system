import logging
import time
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.core.resume_parser import ResumeParser
from src.core.document_reader import DocumentReader
from src.utils.quality_monitor import QualityMonitor
from src.core.extracted_value import ExtractedValue
import re

logger = logging.getLogger(__name__)

class TwoPassProcessor:
    def __init__(self, max_workers: int = 4, fast_confidence_threshold: float = 0.8):
        """Initialize the two-pass processor"""
        self.max_workers = max_workers
        self.fast_confidence_threshold = fast_confidence_threshold
        self.document_reader = DocumentReader()
        self.parser = ResumeParser(use_full_text=True)
        self.quality_monitor = QualityMonitor()
    
    def process_resumes(self, resume_paths: List[str]) -> List[Dict[str, Any]]:
        """Process a list of resumes with two-pass approach"""
        logger.info(f"Starting two-pass processing of {len(resume_paths)} resumes")
        results = []
        start_time = time.time()
        
        # First pass: Quick extraction
        high_confidence = []
        need_quality_pass = []
        
        for resume_path in resume_paths:
            try:
                # Read document
                text, used_ocr = self.document_reader.read_document(resume_path)
                if not text:
                    logger.error(f"No text extracted from {resume_path}")
                    continue
                
                # Process with parser using parse_resume_text instead of parse_resume
                result = self.parser.parse_resume_text(text, file_path=resume_path, used_ocr=used_ocr)
                if result:
                    confidence = result.get('confidence_score', 0)
                    if confidence >= self.fast_confidence_threshold:
                        high_confidence.append((resume_path, result))
                    else:
                        need_quality_pass.append(resume_path)
                    
                    # Log extraction
                    self.quality_monitor.log_extraction(
                        resume_path,
                        result,
                        used_ocr
                    )
                else:
                    logger.error(f"Failed to parse {resume_path}")
            except Exception as e:
                logger.error(f"Error processing {resume_path}: {e}")
        
        # Add high confidence results
        results.extend([r for _, r in high_confidence])
        
        # Second pass: Quality extraction for low confidence
        if need_quality_pass:
            logger.info(f"Starting quality pass for {len(need_quality_pass)} resumes")
            for resume_path in need_quality_pass:
                try:
                    # Read document with quality settings
                    text, used_ocr = self.document_reader.read_document(resume_path)
                    if not text:
                        logger.error(f"No text extracted from {resume_path}")
                        continue
                    
                    # Process with parser using parse_resume_text
                    result = self.parser.parse_resume_text(text, file_path=resume_path, used_ocr=used_ocr)
                    if result:
                        results.append(result)
                        # Log extraction
                        self.quality_monitor.log_extraction(
                            resume_path,
                            result,
                            used_ocr
                        )
                    else:
                        logger.error(f"Failed to parse {resume_path}")
                except Exception as e:
                    logger.error(f"Error processing {resume_path}: {e}")
        
        # Generate quality report
        self.quality_monitor.generate_report()
        
        logger.info(f"First pass completed: {len(high_confidence)} high confidence, {len(need_quality_pass)} need quality pass")
        return results
    
    def process_resume_file(self, resume_path: str, max_chars: int = 50000) -> Dict[str, Any]:
        """Read and parse a single resume file, ensuring correct separation of file path and content."""
        try:
            text, used_ocr = self.document_reader.read_document(resume_path, max_chars=max_chars)
            if not text:
                logger.error(f"No text extracted from {resume_path}")
                return None

            # Split text into smaller chunks if needed
            max_tokens = 512  # Maximum tokens for the model
            chunks = self._split_into_chunks(text, max_tokens)
            
            # Process each chunk and merge results
            merged_result = None
            for chunk in chunks:
                chunk_result = self.parser.parse_resume_text(chunk, file_path=resume_path, used_ocr=used_ocr)
                if chunk_result:
                    if merged_result is None:
                        merged_result = chunk_result
                    else:
                        # Merge results, preferring higher confidence values
                        for key, value in chunk_result.items():
                            if key not in merged_result or (
                                hasattr(value, 'confidence') and 
                                hasattr(merged_result[key], 'confidence') and 
                                value.confidence > merged_result[key].confidence
                            ):
                                merged_result[key] = value

            if merged_result:
                merged_result['resume_link'] = str(Path(resume_path).name)
                merged_result['used_ocr'] = used_ocr
            return merged_result
        except Exception as e:
            logger.error(f"Error processing {resume_path}: {e}")
            return None
    
    def _split_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks that fit within token limit"""
        # Simple splitting by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            # Rough estimate of tokens (words + punctuation)
            para_tokens = len(para.split()) + len(re.findall(r'[^\w\s]', para))
            
            if current_length + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(para)
            current_length += para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _fast_extract(self, resume_path: str) -> Dict[str, Any]:
        """Fast extraction pass (now always uses quality extraction)"""
        try:
            logger.debug(f"[TwoPassProcessor] _fast_extract called with resume_path: {resume_path} (type: {type(resume_path)})")
            return self.process_resume_file(resume_path, max_chars=50000)
        except Exception as e:
            logger.error(f"Error in fast extraction for {resume_path}: {e}")
            return None
    
    def _quality_extract(self, resume_path: str) -> Dict[str, Any]:
        """Quality extraction pass (uses high max_chars)"""
        try:
            logger.debug(f"[TwoPassProcessor] _quality_extract called with resume_path: {resume_path} (type: {type(resume_path)})")
            return self.process_resume_file(resume_path, max_chars=50000)
        except Exception as e:
            logger.error(f"Error in quality extraction for {resume_path}: {e}")
            return None
    
    def _apply_alternative_methods(self, result: Dict[str, Any], text: str):
        """Apply alternative extraction methods for low confidence fields"""
        # Try different extraction methods for each low confidence field
        for field, value in result.items():
            if hasattr(value, 'confidence') and value.confidence < self.fast_confidence_threshold:
                # Try context-based extraction
                new_value = self.parser.extract_using_context(text, field)
                if new_value and new_value.confidence > value.confidence:
                    result[field] = new_value
                
                # Try pattern-based extraction
                new_value = self.parser.extract_using_patterns(text, field)
                if new_value and new_value.confidence > value.confidence:
                    result[field] = new_value 