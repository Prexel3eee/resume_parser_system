"""Data models for the resume parser."""

from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ExtractedValue:
    """Class to hold extracted values with confidence scores and metadata."""
    
    def __init__(self, value: Any, confidence: float, method: str, structured_data: Optional[Dict] = None):
        self.value = value
        self.confidence = confidence
        self.method = method
        self.structured_data = structured_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'value': self.value,
            'confidence': self.confidence,
            'method': self.method,
            'structured_data': self.structured_data
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.value} (confidence: {self.confidence:.2f}, method: {self.method})" 