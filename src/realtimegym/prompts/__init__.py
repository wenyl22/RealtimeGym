"""
Prompts module for RealtimeGym.

This module contains game-specific prompt templates and state-to-description
conversion logic. Each game has:
- A YAML file in configs/prompts/ with prompt templates (data)
- A Python module here with the conversion logic (code)

Available game modules:
- overcooked: Cooperative cooking game prompts
- freeway: Traffic avoidance game prompts
- snake: Snake game prompts
"""

from . import freeway, overcooked, snake

__all__ = ["overcooked", "freeway", "snake"]
