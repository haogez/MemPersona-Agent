from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class BigFive(BaseModel):
    openness: int = Field(..., ge=0, le=100)
    conscientiousness: int = Field(..., ge=0, le=100)
    extraversion: int = Field(..., ge=0, le=100)
    agreeableness: int = Field(..., ge=0, le=100)
    neuroticism: int = Field(..., ge=0, le=100)


class Persona(BaseModel):
    name: str
    age: int
    gender: str
    occupation: str
    hobby: str
    skill: str
    values: str
    living_habit: str
    dislike: str
    language_style: str
    appearance: str
    family_status: str
    education: str
    social_pattern: str
    favorite_thing: str
    usual_place: str
    past_experience: List[str]
    background: str
    speech_style: str
    personality: BigFive

    @field_validator("past_experience")
    @classmethod
    def ensure_non_empty(cls, value: List[str]):
        return value or []


__all__ = ["Persona", "BigFive"]
