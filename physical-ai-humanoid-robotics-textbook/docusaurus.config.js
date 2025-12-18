// Docusaurus configuration for Vercel deployment
const path = require('path');

module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive textbook on Physical AI and Humanoid Robotics',
  url: 'https://physical-ai-humanoid-robotics-textbook.vercel.app',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'your-organization', // Used for GitHub URL
  projectName: 'physical-ai-humanoid-robotics-textbook',
  trailingSlash: false,
  // Remove GitHub Pages specific settings
  themes: [
    // ... other themes
  ],
  plugins: [
    // Custom root component for translation context
    function(context, options) {
      return {
        name: 'client-root-component',
        getClientModules() {
          return [path.resolve(__dirname, 'src/components/ClientRoot')];
        },
      };
    },
  ],
  themeConfig: {
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'doc',
          docId: 'chapters/introduction',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/your-organization/physical-ai-humanoid-robotics-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Textbook',
              to: '/docs/chapters/introduction',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};