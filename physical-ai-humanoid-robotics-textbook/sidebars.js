// Sidebar configuration for Docusaurus
module.exports = {
  docs: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'chapters/introduction',
        'chapters/introduction.ur',
        'chapters/sample-translation'
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 (Robotic Nervous System)',
      items: [
        'chapters/ros2-foundations',
        'chapters/ros2-communication',
        'chapters/ros2-navigation'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo & Unity (Digital Twin)',
      items: [
        'chapters/simulation-basics',
        'chapters/digital-twins',
        'chapters/physics-simulation'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac (AI-Robot Brain)',
      items: [
        'chapters/isaac-ai-integration',
        'chapters/isaac-control-systems'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'chapters/vla-vision-systems',
        'chapters/vla-language-integration'
      ],
    },
    {
      type: 'category',
      label: 'Conclusion',
      items: ['chapters/conclusion'],
    },
  ],
};