# Agent Context for Physical AI & Humanoid Robotics Textbook

## Technology Stack
- Docusaurus v3 for static site generation
- FastAPI for backend RAG services
- Qdrant for vector database storage
- Neon Postgres for metadata and user data
- Better-Auth for authentication
- OpenAI Agents SDK for conversational interfaces
- ROS 2, Gazebo, Unity, NVIDIA Isaac for robotics frameworks
- Vision-Language-Action (VLA) systems

## Project Structure
- Frontend: Docusaurus-based textbook interface
- Backend: FastAPI API services
- Database: Qdrant (vector) + Neon Postgres (metadata)
- Authentication: Better-Auth integration
- Chatbot: RAG-based conversational interface

## Architecture Overview
- Static frontend with embedded RAG chatbot
- FastAPI backend handling document retrieval and processing
- Qdrant vector database for textbook content indexing
- Neon Postgres for user state and metadata management
- GitHub Actions for CI/CD and deployment to GitHub Pages

## Key Components
1. Docusaurus textbook interface with 10 chapters across 4 modules
2. RAG chatbot with content-aware responses
3. User authentication and personalization system
4. Urdu translation capabilities
5. Progress tracking and user analytics

## Implementation Notes
- Follows Spec-Driven Development methodology
- All content must meet constitutional requirements for accuracy
- Zero tolerance for hallucinated facts or unverifiable claims
- Flesch-Kincaid Grade Level 10-12 readability requirement
- APA 7th edition citation standards required

---
# END OF SPEC-KIT PLUS AGENT CONTEXT
# Do not modify the marker above - it's used by automation
---