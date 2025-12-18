# ADR 1: Textbook Technology Stack

## Status
Accepted

## Date
2025-12-15

## Context
The Physical AI & Humanoid Robotics textbook requires a technology stack that supports static content delivery, interactive learning experiences, and advanced features like RAG chatbot, personalization, and multilingual support. The solution must be cost-effective, scalable, and aligned with the project's educational goals while maintaining high standards for technical accuracy and accessibility.

## Decision
We will use the following integrated technology stack:

**Frontend**: Docusaurus v3 static site generator with classic preset
- Provides excellent documentation site capabilities
- Strong Markdown support with extended features
- Built-in search functionality
- Responsive design out of the box
- Plugin ecosystem for enhanced functionality
- GitHub Pages deployment integration

**Backend Services**: FastAPI for RAG and user management
- Async capabilities for efficient API handling
- Excellent performance characteristics
- Python ecosystem integration
- Type safety through Pydantic

**Data Storage**:
- Qdrant vector database for content indexing (free tier)
- Neon Serverless Postgres for user state and metadata

**Authentication**: Better-Auth for user authentication
- Lightweight integration with Docusaurus
- Multiple authentication methods support
- Educational use case optimization

**AI Integration**: OpenAI Agents SDK for conversational interface
- Natural language processing capabilities
- Integration with RAG system

## Alternatives Considered
1. **GitBook + custom backend**: Good for books but less flexible for custom features
2. **mdBook + Node.js backend**: Rust-based, good for technical content but limited interactivity
3. **Custom React app + full-stack framework**: Maximum flexibility but significant development overhead
4. **VuePress + different backend stack**: Alternative static site generator but smaller ecosystem

## Consequences
**Positive:**
- Industry standard for documentation sites, excellent for textbook content
- Cost-effective deployment via GitHub Pages
- Scalable architecture with clear separation of concerns
- Strong community support and documentation
- Efficient development workflow

**Negative:**
- Learning curve for team members unfamiliar with Python backend
- Dependency on multiple third-party services
- Potential limitations with free-tier vector database

## References
- plan.md: Technical Context section
- research.md: Docusaurus v3 Integration Research
- data-model.md: Entity definitions for User, Chapter, etc.