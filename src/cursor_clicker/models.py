"""
Pydantic models for cursor_clicker.

This module contains the data models used for structured data handling
in the cursor_clicker package.
"""

from typing import List, Literal
from pydantic import BaseModel, Field

class ButtonLocation(BaseModel):
    """Model for button location on screen."""
    x1: int = Field(..., description="Left coordinate of the button")
    y1: int = Field(..., description="Top coordinate of the button")
    x2: int = Field(..., description="Right coordinate of the button") 
    y2: int = Field(..., description="Bottom coordinate of the button")
    text: str = Field(..., description="Text on the button")

class ScreenAnalysisResult(BaseModel):
    """Model for screen analysis results."""
    error_type: Literal["none", "tool_call_limit", "anthropic_unavailable"] = Field(
        ..., description="Type of error detected in the screenshot"
    )
    message: str = Field("", description="The error message text if detected")
    buttons: List[ButtonLocation] = Field(
        default_factory=list, 
        description="Locations of action buttons like 'Try Again' or 'Continue'"
    ) 