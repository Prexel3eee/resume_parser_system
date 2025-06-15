import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any
import json
from pathlib import Path
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class QualityMonitor:
    def __init__(self, log_dir: str = "logs"):
        """Initialize the quality monitor."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.resume_data = {}
        self.error_files = set()
        self.metrics = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "success_rate": 0.0,
            "ocr_usage_percentage": 0.0,
            "extraction_times": [],
            "empty_fields": {},
            "field_confidence": {}
        }
        self.start_time = None
        
    def log_extraction(self, resume_path: str, result: Dict[str, Any], used_ocr: bool = False):
        """Log extraction results for a resume"""
        self.resume_data[resume_path] = {
            "result": result,
            "used_ocr": used_ocr,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics["total_processed"] += 1
        
        if result:
            self.metrics["successful_extractions"] += 1
            if used_ocr:
                self.metrics["ocr_usage_percentage"] = (
                    (self.metrics["ocr_usage_percentage"] * (self.metrics["total_processed"] - 1) + 1) 
                    / self.metrics["total_processed"]
                )
        else:
            self.metrics["failed_extractions"] += 1
        
        # Update success rate
        self.metrics["success_rate"] = (
            (self.metrics["successful_extractions"] / self.metrics["total_processed"] * 100)
            if self.metrics["total_processed"] > 0 else 0.0
        )
        
        # Update field statistics
        for field, value in result.items():
            if hasattr(value, "value"):
                if not value.value:
                    self.metrics["empty_fields"][field] = self.metrics["empty_fields"].get(field, 0) + 1
                if hasattr(value, "confidence"):
                    if field not in self.metrics["field_confidence"]:
                        self.metrics["field_confidence"][field] = []
                    self.metrics["field_confidence"][field].append(value.confidence)
        
        # Track extraction time
        if self.start_time is None:
            self.start_time = time.time()
        extraction_time = time.time() - self.start_time
        self.metrics["extraction_times"].append(extraction_time)
        
    def log_error(self, resume_path: str, error: str):
        """Log error for a resume"""
        self.error_files.add(resume_path)
        logger.error(f"Error processing {resume_path}: {error}")
    
    def get_error_files(self) -> set:
        """Get set of files that had errors"""
        return self.error_files
    
    def generate_report(self):
        """Generate a quality report for all processed resumes."""
        report_dict = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_processed": self.metrics["total_processed"],
                "successful_extractions": self.metrics["successful_extractions"],
                "failed_extractions": self.metrics["failed_extractions"],
                "success_rate": self.metrics["success_rate"],
                "ocr_usage_percentage": self.metrics["ocr_usage_percentage"],
                "avg_extraction_time": np.mean(self.metrics["extraction_times"]) if self.metrics["extraction_times"] else 0.0
            },
            "field_analysis": {
                "empty_fields": self.metrics["empty_fields"],
                "field_confidence": self.metrics["field_confidence"]
            },
            "skills_analysis": {
                "categories": {
                    "technical_skills": {"count": 0, "skills": []},
                    "business_skills": {"count": 0, "skills": []},
                    "creative_skills": {"count": 0, "skills": []},
                    "communication_skills": {"count": 0, "skills": []},
                    "industry_skills": {"count": 0, "skills": []},
                    "soft_skills": {"count": 0, "skills": []},
                    "other_skills": {"count": 0, "skills": []}
                },
                "total_skills": 0,
                "unique_skills": 0
            },
            "resume_details": []
        }

        all_skills = set()
        
        # Process each resume
        for resume_path, resume_data in self.resume_data.items():
            resume_detail = {
                "resume_path": str(resume_path),
                "confidence_score": resume_data["result"].get("confidence_score", 0.0),
                "used_ocr": resume_data["result"].get("used_ocr", False),
                "extracted_fields": {}
            }
            
            # Process skills
            skills_data = resume_data["result"].get("skills", {})
            if isinstance(skills_data, dict):
                # Add skills to resume detail
                resume_detail["extracted_fields"]["skills"] = skills_data
                
                # Update skills analysis
                for category, skills in skills_data.items():
                    if isinstance(skills, dict):
                        for subcategory, skill_list in skills.items():
                            if skill_list:  # If there are skills in this subcategory
                                category_key = f"{category}_{subcategory}"
                                if category_key not in report_dict["skills_analysis"]["categories"]:
                                    report_dict["skills_analysis"]["categories"][category_key] = {"count": 0, "skills": []}
                                report_dict["skills_analysis"]["categories"][category_key]["skills"].extend(skill_list)
                                report_dict["skills_analysis"]["categories"][category_key]["count"] = len(skill_list)
                                all_skills.update(skill_list)
            
            # Process other fields
            for field, value in resume_data["result"].items():
                if field not in ["skills", "confidence_score", "used_ocr"]:
                    if hasattr(value, "to_dict"):
                        resume_detail["extracted_fields"][field] = value.to_dict()
                    else:
                        resume_detail["extracted_fields"][field] = value
            
            report_dict["resume_details"].append(resume_detail)
        
        # Update skills statistics
        report_dict["skills_analysis"]["total_skills"] = len(all_skills)
        report_dict["skills_analysis"]["unique_skills"] = len(set(all_skills))
        
        # Sort skills in each category
        for category in report_dict["skills_analysis"]["categories"].values():
            category["skills"] = sorted(set(category["skills"]))
        
        # Save report to file
        report_file = self.log_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Log summary
        logger.info(f"Quality Report Generated: {report_file}")
        logger.info(f"Total Processed: {self.metrics['total_processed']}")
        logger.info(f"Success Rate: {self.metrics['success_rate']:.2f}%")
        logger.info(f"Average Extraction Time: {np.mean(self.metrics['extraction_times']) if self.metrics['extraction_times'] else 0.0:.2f}s")
        
        return report_dict
    
    def get_field_quality(self, field: str) -> Dict[str, float]:
        """Get quality metrics for a specific field"""
        if field not in self.metrics['field_confidence']:
            return {
                'mean_confidence': 0,
                'min_confidence': 0,
                'max_confidence': 0,
                'empty_count': self.metrics['empty_fields'].get(field, 0)
            }
        
        scores = self.metrics['field_confidence'][field]
        return {
            'mean_confidence': np.mean(scores),
            'min_confidence': np.min(scores),
            'max_confidence': np.max(scores),
            'empty_count': self.metrics['empty_fields'].get(field, 0)
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "success_rate": 0.0,
            "ocr_usage_percentage": 0.0,
            "extraction_times": [],
            "empty_fields": {},
            "field_confidence": {}
        }
        self.resume_data = {}
        self.error_files = set()
        self.start_time = None 