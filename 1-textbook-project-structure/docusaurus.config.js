// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'An AI-Native Textbook for Embodied Intelligence',
  favicon: 'img/favicon.ico',

  url: 'https://YOUR_GITHUB_USERNAME.github.io',
  baseUrl: '/physical-ai-humanoid-robotics-textbook/',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      ({
        docs: {
          routeBasePath: '/', // ✅ DOCS = HOME PAGE
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/YOUR_GITHUB_USERNAME/physical-ai-humanoid-robotics-textbook/edit/main/',
        },
        blog: false, // ❌ Disable blog for textbook
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: ({
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          label: 'Textbook',
          position: 'left',
        },
        {
          href: 'https://github.com/YOUR_GITHUB_USERNAME/physical-ai-humanoid-robotics-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright:
        `© ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  }),
};

export default config;
