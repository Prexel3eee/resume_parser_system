from typing import Any, Dict, Optional
from datetime import datetime

class ExtractedValue:
    """Class to store extracted information with confidence scores and methods."""
    
    def __init__(self, value: Any = None, confidence: float = 0.0, method: str = "unknown"):
        """
        Initialize an ExtractedValue object.
        
        Args:
            value: The extracted value
            confidence: Confidence score (0.0 to 1.0)
            method: The method used for extraction (e.g., 'ner', 'regex', 'hybrid')
        """
        self.value = value
        self.confidence = confidence
        self.method = method
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the ExtractedValue to a dictionary."""
        return {
            "value": self.value,
            "confidence": self.confidence,
            "method": self.method,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation of the ExtractedValue."""
        if self.value is None:
            return ""
        return f"{self.value} (confidence: {self.confidence:.2f}, method: {self.method})"
    
    def __bool__(self) -> bool:
        """Return True if the value is not None."""
        return self.value is not None 