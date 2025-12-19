# Docusaurus Deployment Guide for Vercel

## Required Project Structure
- package.json with docusaurus dependencies and build script
- docusaurus.config.js configuration file
- src/ directory with page components
- docs/ directory with documentation content
- static/ directory (if any static assets)
- vercel.json (optional but recommended for explicit configuration)

## vercel.json Configuration
```json
{
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

## Build Command and Output Directory
- Build Command: `npm run build`
- Output Directory: `build`

## Step-by-Step Deployment Instructions

### Prerequisites
- A Docusaurus project pushed to a Git repository (GitHub, GitLab, or Bitbucket)
- A Vercel account (sign up at https://vercel.com)

### Deployment Steps
1. Go to https://vercel.com and sign in to your account
2. Click "New Project" and select your Git repository containing the Docusaurus site
3. Vercel will automatically detect that it's a Docusaurus project
4. In the configuration step:
   - Build Command: `npm run build` (should be auto-detected)
   - Output Directory: `build` (should be auto-detected)
   - Root Directory: project root (default)
5. Click "Deploy" to start the deployment process
6. Your site will be available at a URL like `https://your-project-name.vercel.app`
7. For subsequent deployments, pushes to your Git repository will automatically trigger new builds

### Post-Deployment
- Your site is now deployed and accessible via the provided URL
- You can customize the domain name in Project Settings > Domains
- Build logs are available in the Vercel dashboard for troubleshooting

## Common Deployment Issues and Fixes

### 1. Build fails with "Docusaurus found broken links" error
- **Issue**: Docusaurus build fails due to internal broken links
- **Solution**: Check all internal links in your documentation and ensure they match the actual file paths and routing configuration

### 2. Base URL configuration issues
- **Issue**: Site doesn't work properly after deployment (e.g., 404 errors on navigation)
- **Solution**: Ensure your `docusaurus.config.js` has the correct base URL. For Vercel, set `baseUrl: '/'`

### 3. Assets not loading after deployment
- **Issue**: Images, CSS, or JS files don't load
- **Solution**: Check that your base URL is correctly set in `docusaurus.config.js`

### 4. Vercel doesn't recognize the build output
- **Issue**: Vercel tries to serve files from wrong directory
- **Solution**: Create a `vercel.json` file with explicit build configuration (as shown above)

### 5. Custom domain issues
- **Issue**: Problems when setting up custom domains
- **Solution**: After deployment, go to Project Settings > Domains and add your custom domain

## Troubleshooting Tips

- Always test your build locally with `npm run build` before deploying
- Check the Vercel build logs if deployment fails
- Ensure your `docusaurus.config.js` has correct URL and base URL settings
- For GitHub Pages migration, change base URL from `/repository-name/` to `/`