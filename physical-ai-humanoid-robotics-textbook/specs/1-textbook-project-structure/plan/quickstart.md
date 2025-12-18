# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Development Environment Setup

### Prerequisites
- Node.js v18+
- Python 3.9+
- Git
- Docker (for local development with all services)

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd physical-ai-humanoid-robotics-textbook
   ```

2. **Install frontend dependencies**:
   ```bash
   cd physical-ai-humanoid-robotics-textbook
   npm install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start the Docusaurus development server**:
   ```bash
   npm start
   ```

### Backend Setup (RAG Services)

1. **Navigate to RAG backend directory**:
   ```bash
   cd physical-ai-humanoid-robotics-textbook/RAG-backend
   ```

2. **Create Python virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install fastapi uvicorn qdrant-client openai python-dotenv
   ```

4. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

## Project Structure Overview

```
physical-ai-humanoid-robotics-textbook/
├── docs/                    # Textbook chapters and content
│   └── chapters/            # Individual chapter files
├── src/                     # Custom React components and styling
│   ├── pages/              # Additional pages (bonus features)
│   └── css/                # Custom styles
├── specs/                   # Specifications and planning documents
├── RAG-backend/            # RAG chatbot backend code
├── .github/                # GitHub Actions workflows
├── docusaurus.config.js    # Docusaurus configuration
├── sidebars.js             # Navigation sidebar configuration
└── package.json            # Frontend dependencies
```

## Adding New Content

### Adding a New Chapter

1. **Create the chapter file**:
   ```bash
   # In docs/chapters/
   touch chapterX.md
   ```

2. **Add frontmatter to the chapter**:
   ```markdown
   ---
   sidebar_position: X
   description: Brief description of the chapter
   keywords: [keyword1, keyword2]
   ---

   # Chapter Title

   Chapter content in Markdown...
   ```

3. **Update sidebar configuration** in `sidebars.js`:
   ```javascript
   module.exports = {
     docs: [
       {
         type: 'category',
         label: 'Module Name',
         items: ['chapters/chapter1', 'chapters/chapter2', 'chapters/chapterX'],
       },
     ],
   };
   ```

### Adding Urdu Translation

1. **Add Urdu content to the same chapter file**:
   ```markdown
   ---
   sidebar_position: X
   description: Brief description of the chapter
   keywords: [keyword1, keyword2]
   ---

   # Chapter Title

   English content in Markdown...

   <details>
   <summary>اردو ترجمہ دیکھیں / Show Urdu Translation</summary>

   # عنوان باب

   اردو میں باب کا مواد...

   </details>
   ```

## Working with the RAG System

### Indexing Content

1. **Prepare content for indexing**:
   - Ensure content is in the correct format
   - Verify citations and references
   - Check for technical accuracy

2. **Run the indexing script**:
   ```bash
   cd RAG-backend
   python index_content.py
   ```

### Testing the Chat Interface

1. **Start both frontend and backend servers**

2. **Access the chat interface**:
   - Navigate to the chapter page
   - Use the embedded chat widget
   - Ask questions related to the textbook content

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info

### Content
- `GET /api/chapters` - Get all chapters
- `GET /api/chapters/{id}` - Get specific chapter
- `GET /api/chapters/{id}/urdu` - Get Urdu translation

### Chat
- `POST /api/chat` - Send message to RAG chatbot

### User Preferences
- `GET /api/user/preferences` - Get user preferences
- `PUT /api/user/preferences` - Update user preferences

## Running Tests

### Frontend Tests
```bash
npm test
```

### Backend Tests
```bash
cd RAG-backend
python -m pytest tests/
```

## Deployment

### Local Development
```bash
npm start  # Runs development server with hot reload
```

### Production Build
```bash
npm run build  # Creates optimized build in build/ directory
npm run serve  # Serves the built application locally
```

### GitHub Pages Deployment
The site is automatically deployed via GitHub Actions when changes are pushed to the main branch. The workflow is defined in `.github/workflows/deploy.yml`.

## Troubleshooting

### Common Issues

1. **Frontend not loading**:
   - Check that all dependencies are installed: `npm install`
   - Verify port 3000 is available
   - Clear browser cache and try again

2. **Chatbot not responding**:
   - Verify backend server is running
   - Check that Qdrant vector database is accessible
   - Ensure API keys are properly configured

3. **Content not appearing**:
   - Verify file is in correct directory
   - Check that sidebar configuration includes the new content
   - Ensure frontmatter is properly formatted

### Development Tips

- Use `npm run docusaurus className` to check for broken links
- Enable debug mode by setting `DEBUG=true` in environment variables
- Use the Docusaurus debug plugin during development