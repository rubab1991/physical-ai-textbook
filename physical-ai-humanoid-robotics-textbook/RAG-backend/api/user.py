from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
from ..api.auth import get_current_user

router = APIRouter()

# Request/Response models
class UserPreferences(BaseModel):
    difficulty_level: Optional[str] = "intermediate"
    learning_style: Optional[str] = "balanced"
    language_preference: Optional[str] = "en"
    content_focus: Optional[str] = "conceptual"

class UserPreferencesUpdate(BaseModel):
    difficulty_level: Optional[str] = None
    learning_style: Optional[str] = None
    language_preference: Optional[str] = None
    content_focus: Optional[str] = None

class UserProgress(BaseModel):
    total_chapters: int
    completed_chapters: int
    overall_progress: int
    module_progress: list
    recent_activity: list

class UpdateProgressRequest(BaseModel):
    progress_percentage: int
    time_spent: int

# Mock user preferences and progress storage (in real implementation, this would be a database)
user_preferences_db = {}
user_progress_db = {}

@router.get("/user/preferences", response_model=UserPreferences)
async def get_user_preferences(current_user: dict = Depends(get_current_user)):
    """
    Get current user's personalization preferences
    """
    user_id = current_user["id"]

    # Check if user preferences exist, otherwise return defaults
    if user_id not in user_preferences_db:
        # Create default preferences for new users
        default_prefs = {
            "difficulty_level": "intermediate",
            "learning_style": "balanced",
            "language_preference": "en",
            "content_focus": "conceptual"
        }
        user_preferences_db[user_id] = default_prefs

    user_prefs = user_preferences_db[user_id]

    return UserPreferences(
        difficulty_level=user_prefs.get("difficulty_level", "intermediate"),
        learning_style=user_prefs.get("learning_style", "balanced"),
        language_preference=user_prefs.get("language_preference", "en"),
        content_focus=user_prefs.get("content_focus", "conceptual")
    )

@router.put("/user/preferences", response_model=UserPreferences)
async def update_user_preferences(
    preferences: UserPreferencesUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user's personalization preferences
    """
    user_id = current_user["id"]

    # Get existing preferences or create new ones
    if user_id not in user_preferences_db:
        user_preferences_db[user_id] = {}

    # Update only the fields that were provided
    if preferences.difficulty_level is not None:
        user_preferences_db[user_id]["difficulty_level"] = preferences.difficulty_level

    if preferences.learning_style is not None:
        user_preferences_db[user_id]["learning_style"] = preferences.learning_style

    if preferences.language_preference is not None:
        user_preferences_db[user_id]["language_preference"] = preferences.language_preference

    if preferences.content_focus is not None:
        user_preferences_db[user_id]["content_focus"] = preferences.content_focus

    # Ensure all required fields exist in the final result
    updated_prefs = UserPreferences(
        difficulty_level=user_preferences_db[user_id].get("difficulty_level", "intermediate"),
        learning_style=user_preferences_db[user_id].get("learning_style", "balanced"),
        language_preference=user_preferences_db[user_id].get("language_preference", "en"),
        content_focus=user_preferences_db[user_id].get("content_focus", "conceptual")
    )

    return updated_prefs

@router.get("/user/progress", response_model=UserProgress)
async def get_user_progress(current_user: dict = Depends(get_current_user)):
    """
    Get user's progress through the textbook
    """
    user_id = current_user["id"]

    # Initialize progress data if not exists
    if user_id not in user_progress_db:
        user_progress_db[user_id] = {
            "total_chapters": 12,
            "completed_chapters": 0,
            "overall_progress": 0,
            "module_progress": [
                {
                    "module_id": "module-1",
                    "module_title": "Module 1: ROS 2 (Robotic Nervous System)",
                    "completed_chapters": 0,
                    "total_chapters": 4,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-2",
                    "module_title": "Module 2: Gazebo & Unity (Digital Twin)",
                    "completed_chapters": 0,
                    "total_chapters": 3,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-3",
                    "module_title": "Module 3: NVIDIA Isaac (AI-Robot Brain)",
                    "completed_chapters": 0,
                    "total_chapters": 2,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-4",
                    "module_title": "Module 4: Vision-Language-Action (VLA)",
                    "completed_chapters": 0,
                    "total_chapters": 2,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-0",
                    "module_title": "Introduction",
                    "completed_chapters": 0,
                    "total_chapters": 1,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-5",
                    "module_title": "Conclusion",
                    "completed_chapters": 0,
                    "total_chapters": 1,
                    "progress_percentage": 0
                }
            ],
            "recent_activity": []
        }

    # Calculate overall progress
    total_chapters = user_progress_db[user_id]["total_chapters"]
    completed_chapters = sum(
        module["completed_chapters"] for module in user_progress_db[user_id]["module_progress"]
    )
    overall_progress = int((completed_chapters / total_chapters) * 100) if total_chapters > 0 else 0

    # Update overall progress in db
    user_progress_db[user_id]["overall_progress"] = overall_progress
    user_progress_db[user_id]["completed_chapters"] = completed_chapters

    return UserProgress(
        total_chapters=total_chapters,
        completed_chapters=completed_chapters,
        overall_progress=overall_progress,
        module_progress=user_progress_db[user_id]["module_progress"],
        recent_activity=user_progress_db[user_id]["recent_activity"]
    )

@router.put("/user/progress/{chapter_id}", response_model=dict)
async def update_chapter_progress(
    chapter_id: str,
    progress_update: UpdateProgressRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update progress for a specific chapter
    """
    user_id = current_user["id"]

    # Initialize user progress data if not exists
    if user_id not in user_progress_db:
        # Get default progress structure
        user_progress_db[user_id] = {
            "total_chapters": 12,
            "completed_chapters": 0,
            "overall_progress": 0,
            "module_progress": [
                {
                    "module_id": "module-1",
                    "module_title": "Module 1: ROS 2 (Robotic Nervous System)",
                    "completed_chapters": 0,
                    "total_chapters": 4,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-2",
                    "module_title": "Module 2: Gazebo & Unity (Digital Twin)",
                    "completed_chapters": 0,
                    "total_chapters": 3,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-3",
                    "module_title": "Module 3: NVIDIA Isaac (AI-Robot Brain)",
                    "completed_chapters": 0,
                    "total_chapters": 2,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-4",
                    "module_title": "Module 4: Vision-Language-Action (VLA)",
                    "completed_chapters": 0,
                    "total_chapters": 2,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-0",
                    "module_title": "Introduction",
                    "completed_chapters": 0,
                    "total_chapters": 1,
                    "progress_percentage": 0
                },
                {
                    "module_id": "module-5",
                    "module_title": "Conclusion",
                    "completed_chapters": 0,
                    "total_chapters": 1,
                    "progress_percentage": 0
                }
            ],
            "recent_activity": []
        }

    # Update chapter progress
    now_iso = datetime.utcnow().isoformat() + "Z"

    # Add to recent activity
    recent_activity_entry = {
        "chapter_id": chapter_id,
        "chapter_title": f"Chapter {chapter_id}",  # In a real implementation, get the actual title
        "last_accessed": now_iso,
        "progress_percentage": progress_update.progress_percentage
    }

    # Add to beginning of recent activity list (most recent first)
    user_progress_db[user_id]["recent_activity"].insert(0, recent_activity_entry)

    # Keep only the 10 most recent activities
    user_progress_db[user_id]["recent_activity"] = user_progress_db[user_id]["recent_activity"][:10]

    # Determine which module this chapter belongs to (simplified logic)
    # In a real implementation, chapter-to-module mapping would be stored in the database
    if "ros2" in chapter_id.lower():
        module_id = "module-1"
    elif "simulation" in chapter_id.lower() or "digital" in chapter_id.lower() or "physics" in chapter_id.lower():
        module_id = "module-2"
    elif "isaac" in chapter_id.lower():
        module_id = "module-3"
    elif "vla" in chapter_id.lower():
        module_id = "module-4"
    elif "introduction" in chapter_id.lower():
        module_id = "module-0"
    elif "conclusion" in chapter_id.lower():
        module_id = "module-5"
    else:
        module_id = "module-1"  # Default

    # Find the module and update its progress
    for module in user_progress_db[user_id]["module_progress"]:
        if module["module_id"] == module_id:
            # If this chapter is now completed, increment completed chapters for this module
            if progress_update.progress_percentage >= 100:
                if module["completed_chapters"] < module["total_chapters"]:
                    module["completed_chapters"] += 1

            # Update module progress percentage
            module["progress_percentage"] = int((module["completed_chapters"] / module["total_chapters"]) * 100)
            break

    # Calculate overall progress
    total_chapters = user_progress_db[user_id]["total_chapters"]
    completed_chapters = sum(
        module["completed_chapters"] for module in user_progress_db[user_id]["module_progress"]
    )
    overall_progress = int((completed_chapters / total_chapters) * 100) if total_chapters > 0 else 0

    # Update overall stats
    user_progress_db[user_id]["overall_progress"] = overall_progress
    user_progress_db[user_id]["completed_chapters"] = completed_chapters

    return {
        "chapter_id": chapter_id,
        "progress_percentage": progress_update.progress_percentage,
        "time_spent": progress_update.time_spent,
        "completed": progress_update.progress_percentage >= 100,
        "updated_at": now_iso
    }

# Additional user-related endpoints can be added here