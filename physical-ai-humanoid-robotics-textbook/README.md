# Physical AI & Humanoid Robotics Textbook

[![Vercel Deployment](https://vercelbadge.vercel.app/api/physical-ai-humanoid-robotics-textbook)](https://physical-ai-humanoid-robotics-textbook.vercel.app)

A comprehensive textbook exploring the intersection of artificial intelligence and robotics, focusing on humanoid robots that can perceive, reason, and act in complex environments.

## Getting Started

1. Install dependencies: `npm install`
2. Start the development server: `npm start`
3. Build for production: `npm run build`

## Deployment

This project is configured for deployment on Vercel. The site will automatically build and deploy when changes are pushed to the repository.

The live deployment is available at:
- Primary: https://physical-ai-humanoid-robotics-textbook-g1t8p039q.vercel.app
- Alias: https://physical-ai-humanoid-robotics-textbook-lilac.vercel.app

### Manual Deployment to Vercel

If you want to deploy this project manually to Vercel:

1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Navigate to the project directory:
   ```bash
   cd physical-ai-humanoid-robotics-textbook
   ```

3. Deploy to Vercel:
   ```bash
   vercel
   ```

### GitHub Integration

To deploy directly from GitHub:

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project" and select your GitHub repository
3. Vercel will automatically detect this is a Docusaurus project and use the configuration in `vercel.json`
4. The site will deploy automatically on pushes to the main branch

## Project Structure

- `docs/` - Textbook chapters and content
- `src/` - Custom React components and styling
- `specs/` - Specifications and requirements documentation
- `RAG-backend/` - Retrieval-Augmented Generation backend code
- `src/pages/index.tsx` - Landing page with animated hero section
- `docusaurus.config.js` - Site configuration
- `vercel.json` - Vercel deployment configuration