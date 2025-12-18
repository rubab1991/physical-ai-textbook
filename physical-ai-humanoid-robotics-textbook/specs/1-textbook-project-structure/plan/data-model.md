# Data Model: Physical AI & Humanoid Robotics Textbook

## Entity: User

**Description**: Represents a registered user of the textbook system

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the user |
| email | String(255) | Unique, Not Null | User's email address |
| name | String(255) | Not Null | User's full name |
| preferences | JSON | Nullable | User preferences for personalization |
| created_at | Timestamp | Not Null | Account creation timestamp |
| updated_at | Timestamp | Not Null | Last update timestamp |

**Validation Rules**:
- Email must be valid email format
- Name must be 2-255 characters
- Preferences must be valid JSON with defined schema

**State Transitions**:
- New Registration → Active Account
- Account Deletion → Deleted (soft delete)

## Entity: Module

**Description**: Represents a major section of the textbook (e.g., ROS 2, Gazebo & Unity)

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the module |
| title | String(255) | Not Null | Title of the module |
| description | Text | Nullable | Detailed description of the module |
| order | Integer | Not Null | Display order of the module |
| created_at | Timestamp | Not Null | Creation timestamp |
| updated_at | Timestamp | Not Null | Last update timestamp |

**Validation Rules**:
- Title must be 5-255 characters
- Order must be positive integer
- Title must be unique across modules

## Entity: Chapter

**Description**: Represents a chapter within a module of the textbook

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the chapter |
| title | String(255) | Not Null | Title of the chapter |
| content | Text | Not Null | Main content of the chapter in Markdown |
| module_id | UUID | Foreign Key, Not Null | Reference to parent module |
| order | Integer | Not Null | Display order within the module |
| urdu_content | Text | Nullable | Urdu translation of the content |
| has_personalization | Boolean | Not Null, Default: false | Whether personalization is available |
| created_at | Timestamp | Not Null | Creation timestamp |
| updated_at | Timestamp | Not Null | Last update timestamp |

**Validation Rules**:
- Title must be 5-255 characters
- Content must be valid Markdown
- Order must be positive integer
- module_id must reference existing module
- Urdu content, if present, must be translation of English content

**Relationships**:
- Chapter belongs to one Module (module_id → Module.id)
- Module has many Chapters

## Entity: ChatSession

**Description**: Represents a conversation session with the RAG chatbot

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the session |
| user_id | UUID | Foreign Key, Nullable | Reference to user (null for anonymous) |
| messages | JSON | Not Null | Array of messages in the conversation |
| active | Boolean | Not Null, Default: true | Whether session is active |
| created_at | Timestamp | Not Null | Creation timestamp |
| updated_at | Timestamp | Not Null | Last update timestamp |

**Validation Rules**:
- Messages must be valid JSON array
- Each message must have role (user/assistant) and content
- Session cannot exceed maximum message count

**Relationships**:
- ChatSession belongs to one User (user_id → User.id, optional)

## Entity: DocumentChunk

**Description**: Represents a chunk of textbook content stored in the vector database

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for the chunk |
| chapter_id | UUID | Foreign Key, Not Null | Reference to source chapter |
| content | Text | Not Null | The actual content text |
| embedding | JSON | Nullable | Vector embedding for similarity search |
| metadata | JSON | Not Null | Additional metadata about the chunk |
| created_at | Timestamp | Not Null | Creation timestamp |

**Validation Rules**:
- Content must be 50-1000 characters (optimal for RAG)
- Metadata must include source chapter and section info
- Embedding must be valid vector format when present

**Relationships**:
- DocumentChunk belongs to one Chapter (chapter_id → Chapter.id)

## Entity: UserProgress

**Description**: Tracks user progress through the textbook

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for progress record |
| user_id | UUID | Foreign Key, Not Null | Reference to user |
| chapter_id | UUID | Foreign Key, Not Null | Reference to chapter |
| progress_percentage | Integer | Not Null, Min: 0, Max: 100 | Progress through the chapter |
| time_spent | Integer | Not Null, Default: 0 | Time spent in seconds |
| completed | Boolean | Not Null, Default: false | Whether chapter is completed |
| last_accessed | Timestamp | Not Null | Last time chapter was accessed |
| created_at | Timestamp | Not Null | Creation timestamp |
| updated_at | Timestamp | Not Null | Last update timestamp |

**Validation Rules**:
- Progress percentage must be 0-100
- Time spent must be non-negative
- User and chapter references must exist

**Relationships**:
- UserProgress belongs to one User (user_id → User.id)
- UserProgress belongs to one Chapter (chapter_id → Chapter.id)

## Entity: PersonalizationSetting

**Description**: Stores user-specific personalization preferences

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for setting |
| user_id | UUID | Foreign Key, Not Null | Reference to user |
| setting_key | String(100) | Not Null | Key for the setting (e.g., difficulty_level) |
| setting_value | String(255) | Not Null | Value for the setting |
| created_at | Timestamp | Not Null | Creation timestamp |
| updated_at | Timestamp | Not Null | Last update timestamp |

**Validation Rules**:
- Setting key must be from predefined list
- Setting value must be valid for the key
- User reference must exist
- Combination of user_id and setting_key must be unique

**Predefined Setting Keys**:
- difficulty_level: beginner, intermediate, advanced
- learning_style: theoretical, practical, balanced
- language_preference: en, ur
- content_focus: mathematical, conceptual, application-oriented

**Relationships**:
- PersonalizationSetting belongs to one User (user_id → User.id)

## Entity: Citation

**Description**: Stores citation information for textbook content

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | Primary Key, Not Null | Unique identifier for citation |
| chapter_id | UUID | Foreign Key, Not Null | Reference to chapter containing citation |
| citation_text | Text | Not Null | The actual citation in APA format |
| doi | String(255) | Nullable | Digital Object Identifier if applicable |
| url | String(500) | Nullable | URL to source |
| access_date | Date | Nullable | Date source was accessed |
| created_at | Timestamp | Not Null | Creation timestamp |

**Validation Rules**:
- Citation text must follow APA 7th edition format
- If DOI is provided, it must be valid format
- If URL is provided, it must be valid format
- Chapter reference must exist

**Relationships**:
- Citation belongs to one Chapter (chapter_id → Chapter.id)