from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from pathlib import Path

router = APIRouter()

# Response models
class Chapter(BaseModel):
    id: str
    title: str
    module_id: str
    module_title: str
    order: int
    has_urdu: bool
    created_at: str

class ChapterDetail(BaseModel):
    id: str
    title: str
    content: str
    module_id: str
    module_title: str
    order: int
    has_urdu: bool
    created_at: str
    updated_at: str

class ChapterUrdu(BaseModel):
    id: str
    title_urdu: str
    content_urdu: str

# Mock data for chapters (in real implementation, this would come from a database)
def load_chapters():
    """Load chapter information from the textbook files"""
    chapters_data = []

    # Define the modules and their titles
    modules = {
        "module-1": "Module 1: ROS 2 (Robotic Nervous System)",
        "module-2": "Module 2: Gazebo & Unity (Digital Twin)",
        "module-3": "Module 3: NVIDIA Isaac (AI-Robot Brain)",
        "module-4": "Module 4: Vision-Language-Action (VLA)"
    }

    # Chapter titles and their corresponding files
    chapter_files = [
        ("Introduction to Physical AI & Humanoid Robotics", "introduction.md"),
        ("ROS 2 Foundations - The Robotic Nervous System", "ros2-foundations.md"),
        ("ROS 2 Communication Patterns - Connecting the Robotic Body", "ros2-communication.md"),
        ("ROS 2 Navigation - Moving with Purpose", "ros2-navigation.md"),
        ("Simulation Basics - Creating Digital Twins", "simulation-basics.md"),
        ("Digital Twins - Bridging Physical and Virtual Worlds", "digital-twins.md"),
        ("Physics Simulation - Realistic Robot Dynamics", "physics-simulation.md"),
        ("NVIDIA Isaac AI Integration - The Robot's Digital Brain", "isaac-ai-integration.md"),
        ("NVIDIA Isaac Control Systems - Orchestrating Robot Behavior", "isaac-control-systems.md"),
        ("Vision Systems in VLA - Seeing the World Through AI Eyes", "vla-vision-systems.md"),
        ("Language Integration in VLA - Communicating with AI Systems", "vla-language-integration.md"),
        ("Conclusion - The Future of Physical AI & Humanoid Robotics", "conclusion.md")
    ]

    for i, (title, filename) in enumerate(chapter_files, 1):
        # Determine module based on index
        if i == 1:  # Introduction
            module_id = "module-0"
            module_title = "Introduction"
        elif i <= 4:  # ROS 2 module (intro + 3 chapters)
            module_id = "module-1"
            module_title = modules["module-1"]
        elif i <= 7:  # Simulation module (3 chapters)
            module_id = "module-2"
            module_title = modules["module-2"]
        elif i <= 9:  # Isaac module (2 chapters)
            module_id = "module-3"
            module_title = modules["module-3"]
        elif i <= 11:  # VLA module (2 chapters)
            module_id = "module-4"
            module_title = modules["module-4"]
        else:  # Conclusion
            module_id = "module-5"
            module_title = "Conclusion"

        chapters_data.append({
            "id": f"chapter-{i}",
            "title": title,
            "module_id": module_id,
            "module_title": module_title,
            "order": i,
            "has_urdu": True if i % 2 == 0 else False,  # Alternate chapters with Urdu support
            "created_at": "2025-12-15T10:00:00Z",
            "filename": filename
        })

    return chapters_data

@router.get("/chapters", response_model=dict)
async def get_chapters():
    """
    Get all chapters with module information
    """
    chapters = load_chapters()

    # Format response to match the specification
    response = {
        "chapters": []
    }

    for chapter in chapters:
        response["chapters"].append({
            "id": chapter["id"],
            "title": chapter["title"],
            "module_id": chapter["module_id"],
            "module_title": chapter["module_title"],
            "order": chapter["order"],
            "has_urdu": chapter["has_urdu"],
            "created_at": chapter["created_at"]
        })

    return response

@router.get("/chapters/{chapter_id}", response_model=ChapterDetail)
async def get_chapter_detail(chapter_id: str):
    """
    Get a specific chapter by ID
    """
    chapters = load_chapters()

    # Find the chapter
    chapter = None
    for ch in chapters:
        if ch["id"] == chapter_id:
            chapter = ch
            break

    if not chapter:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Chapter not found")

    # For demo purposes, we'll return a sample content
    sample_content = f"""# {chapter['title']}

This is a sample chapter content for the Physical AI & Humanoid Robotics textbook.
The actual content would be loaded from the corresponding Markdown file: {chapter['filename']}.

## Overview

This chapter covers important concepts related to the textbook's topic. In a full implementation,
this would be replaced with the actual content from the textbook.

## Learning Objectives

- Understand the key concepts
- Apply the principles to real-world scenarios
- Connect with other chapters in the textbook

## Summary

This chapter is part of the comprehensive curriculum designed to teach Physical AI and Humanoid Robotics.
"""

    return ChapterDetail(
        id=chapter["id"],
        title=chapter["title"],
        content=sample_content,
        module_id=chapter["module_id"],
        module_title=chapter["module_title"],
        order=chapter["order"],
        has_urdu=chapter["has_urdu"],
        created_at=chapter["created_at"],
        updated_at=chapter["created_at"]
    )

@router.get("/chapters/{chapter_id}/urdu", response_model=ChapterUrdu)
async def get_chapter_urdu(chapter_id: str):
    """
    Get Urdu translation of a chapter
    """
    chapters = load_chapters()

    # Find the chapter
    chapter = None
    for ch in chapters:
        if ch["id"] == chapter_id:
            chapter = ch
            break

    if not chapter or not chapter["has_urdu"]:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Urdu translation not available")

    # Return sample Urdu content
    sample_urdu_content = f"""# {chapter['title']} - اردو ترجمہ

یہ فزیکل ای آئی اور ہیومنوائڈ روبوٹکس کے بارے میں ٹیکسٹ بک کا ایک نمونہ章节 کا مواد ہے۔
اصل مواد متعلقہ مارک ڈاؤن فائل سے لوڈ ہوگا: {chapter['filename']}

## جائزہ

یہ chapter ٹیکسٹ بک کے موضوع سے متعلق اہم تصورات پر مشتمل ہے۔ مکمل نفاذ کاری میں،
اسے ٹیکسٹ بک کے اصل مواد کے ساتھ تبدیل کر دیا جائے گا۔

## سیکھنے کے مقاصد

- کلیدی تصورات کو سمجھنا
- حقیقی دنیا کے منظار ناموں میں اصول لاگو کرنا
- ٹیکسٹ بک کے دیگر chapters کے ساتھ ربطہ قائم کرنا

## خلاصہ

یہ chapter فزیکل ای آئی اور ہیومنوائڈ روبوٹکس سیکھنے کے منصوبے کا ایک حصہ ہے۔
"""

    return ChapterUrdu(
        id=chapter["id"],
        title_urdu=f"{chapter['title']} - اردو ترجمہ",
        content_urdu=sample_urdu_content
    )