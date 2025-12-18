# Deployment Guide

## Vercel Deployment

The Physical AI & Humanoid Robotics textbook is deployed on Vercel using the following configuration:

### Build Settings
- Build Command: `npm run build`
- Output Directory: `build`
- Install Command: `npm install`

### Configuration Files
- `vercel.json`: Contains Vercel-specific deployment configuration
- `docusaurus.config.js`: Contains baseUrl and URL settings for production

### Environment Variables
No environment variables are required for basic deployment.

### Deployment Process
1. Push changes to the main branch
2. Vercel automatically detects changes and builds the site
3. The build output is served from the `build` directory
4. CDN automatically serves assets for optimal performance

### Custom Domain
If using a custom domain:
1. Add your domain in the Vercel project settings
2. Update DNS records as instructed by Vercel
3. SSL certificate is automatically provisioned

### Troubleshooting
- If pages don't load correctly, check the baseUrl in `docusaurus.config.js`
- If assets don't load, verify the paths are relative to the root when baseUrl is "/"
- For build errors, check that all dependencies are properly defined in `package.json`