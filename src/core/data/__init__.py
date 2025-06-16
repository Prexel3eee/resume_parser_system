"""Data package for resume parser."""

from .models import ExtractedValue
from .visa_states import US_VISAS, US_STATES, US_STATE_ABBR, US_TAX_TERMS
from .skills import COMMON_SKILLS

__all__ = [
    'ExtractedValue',
    'US_VISAS',
    'US_STATES',
    'US_STATE_ABBR',
    'US_TAX_TERMS',
    'COMMON_SKILLS'
] 