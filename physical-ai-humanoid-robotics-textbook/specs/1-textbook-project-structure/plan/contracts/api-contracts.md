# API Contracts: Physical AI & Humanoid Robotics Textbook

## Authentication API

### POST /api/auth/register
Register a new user account

**Request**:
```json
{
  "email": "user@example.com",
  "name": "John Doe",
  "password": "secure_password"
}
```

**Response (201 Created)**:
```json
{
  "id": "uuid-string",
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2025-12-15T10:00:00Z"
}
```

**Response (400 Bad Request)**:
```json
{
  "error": "Validation error",
  "details": ["Email is already registered"]
}
```

### POST /api/auth/login
Authenticate user and return session token

**Request**:
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response (200 OK)**:
```json
{
  "token": "jwt-token-string",
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

**Response (401 Unauthorized)**:
```json
{
  "error": "Invalid credentials"
}
```

### GET /api/auth/me
Get current user information

**Headers**:
```
Authorization: Bearer {token}
```

**Response (200 OK)**:
```json
{
  "id": "uuid-string",
  "email": "user@example.com",
  "name": "John Doe",
  "preferences": {
    "difficulty_level": "intermediate",
    "learning_style": "balanced",
    "language_preference": "en"
  }
}
```

**Response (401 Unauthorized)**:
```json
{
  "error": "Authentication required"
}
```

## Chapter API

### GET /api/chapters
Get all chapters with module information

**Response (200 OK)**:
```json
{
  "chapters": [
    {
      "id": "uuid-string",
      "title": "Introduction to ROS 2",
      "module_id": "module-uuid",
      "module_title": "ROS 2 (Robotic Nervous System)",
      "order": 1,
      "has_urdu": true,
      "created_at": "2025-12-15T10:00:00Z"
    }
  ]
}
```

### GET /api/chapters/{id}
Get a specific chapter by ID

**Response (200 OK)**:
```json
{
  "id": "uuid-string",
  "title": "Introduction to ROS 2",
  "content": "# Chapter Content in Markdown...",
  "module_id": "module-uuid",
  "module_title": "ROS 2 (Robotic Nervous System)",
  "order": 1,
  "has_urdu": true,
  "created_at": "2025-12-15T10:00:00Z",
  "updated_at": "2025-12-15T10:00:00Z"
}
```

**Response (404 Not Found)**:
```json
{
  "error": "Chapter not found"
}
```

### GET /api/chapters/{id}/urdu
Get Urdu translation of a chapter

**Response (200 OK)**:
```json
{
  "id": "uuid-string",
  "title_urdu": "ROS 2 کا تعارف",
  "content_urdu": "# Urdu chapter content in Markdown..."
}
```

**Response (404 Not Found)**:
```json
{
  "error": "Urdu translation not available"
}
```

## RAG Chat API

### POST /api/chat
Send a message to the RAG chatbot and get a response

**Headers**:
```
Authorization: Bearer {token} (optional)
```

**Request**:
```json
{
  "message": "What is ROS 2?",
  "session_id": "optional-session-uuid",
  "user_id": "optional-user-uuid"
}
```

**Response (200 OK)**:
```json
{
  "response": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot applications...",
  "session_id": "session-uuid",
  "sources": [
    {
      "chapter_id": "chapter-uuid",
      "chapter_title": "Introduction to ROS 2",
      "relevance_score": 0.85
    }
  ],
  "confidence": 0.92
}
```

**Response (400 Bad Request)**:
```json
{
  "error": "Message is required"
}
```

## Personalization API

### GET /api/user/preferences
Get user's personalization preferences

**Headers**:
```
Authorization: Bearer {token}
```

**Response (200 OK)**:
```json
{
  "difficulty_level": "intermediate",
  "learning_style": "balanced",
  "language_preference": "en",
  "content_focus": "conceptual"
}
```

### PUT /api/user/preferences
Update user's personalization preferences

**Headers**:
```
Authorization: Bearer {token}
```

**Request**:
```json
{
  "difficulty_level": "advanced",
  "learning_style": "practical",
  "language_preference": "ur",
  "content_focus": "mathematical"
}
```

**Response (200 OK)**:
```json
{
  "difficulty_level": "advanced",
  "learning_style": "practical",
  "language_preference": "ur",
  "content_focus": "mathematical",
  "updated_at": "2025-12-15T10:00:00Z"
}
```

## User Progress API

### GET /api/user/progress
Get user's progress through the textbook

**Headers**:
```
Authorization: Bearer {token}
```

**Response (200 OK)**:
```json
{
  "total_chapters": 10,
  "completed_chapters": 3,
  "overall_progress": 30,
  "module_progress": [
    {
      "module_id": "module-uuid",
      "module_title": "ROS 2 (Robotic Nervous System)",
      "completed_chapters": 2,
      "total_chapters": 3,
      "progress_percentage": 67
    }
  ],
  "recent_activity": [
    {
      "chapter_id": "chapter-uuid",
      "chapter_title": "Introduction to ROS 2",
      "last_accessed": "2025-12-15T10:00:00Z",
      "progress_percentage": 100
    }
  ]
}
```

### PUT /api/user/progress/{chapterId}
Update progress for a specific chapter

**Headers**:
```
Authorization: Bearer {token}
```

**Request**:
```json
{
  "progress_percentage": 75,
  "time_spent": 1200
}
```

**Response (200 OK)**:
```json
{
  "chapter_id": "chapter-uuid",
  "progress_percentage": 75,
  "time_spent": 1200,
  "completed": false,
  "updated_at": "2025-12-15T10:00:00Z"
}
```

## Module API

### GET /api/modules
Get all textbook modules

**Response (200 OK)**:
```json
{
  "modules": [
    {
      "id": "module-uuid",
      "title": "ROS 2 (Robotic Nervous System)",
      "description": "Foundational concepts of Robot Operating System 2",
      "order": 1,
      "chapter_count": 3,
      "created_at": "2025-12-15T10:00:00Z"
    }
  ]
}
```

## Error Response Format

All error responses follow this format:

```json
{
  "error": "Error message",
  "timestamp": "2025-12-15T10:00:00Z",
  "path": "/api/endpoint",
  "status_code": 400
}
```

## Common Headers

### Authentication
All authenticated endpoints require:
```
Authorization: Bearer {token}
```

### Content Type
All requests and responses use:
```
Content-Type: application/json
```

## Rate Limiting

All API endpoints are subject to rate limiting:
- Unauthenticated requests: 100 requests/hour
- Authenticated requests: 1000 requests/hour
- RAG chat endpoint: 50 requests/hour (due to computational cost)

Rate limit response headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1642345600
```