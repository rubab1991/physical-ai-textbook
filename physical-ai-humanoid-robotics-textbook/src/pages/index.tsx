import React from 'react';
import { motion } from 'framer-motion';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  const buttonVariants = {
    hover: {
      scale: 1.05,
      boxShadow: "0 0 20px rgba(56, 182, 255, 0.5)",
      transition: { duration: 0.2 }
    },
    tap: {
      scale: 0.95
    }
  };

  const floatingVariants = {
    animate: {
      y: [0, -10, 0],
      transition: {
        duration: 4,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };

  return (
    <motion.header
      className={clsx('hero hero--primary', styles.heroBanner)}
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      <div className="container">
        <div className={styles.heroContent}>
          <motion.div
            className={styles.heroText}
            variants={itemVariants}
          >
            <motion.h1
              className={styles.heroTitle}
              variants={itemVariants}
            >
              {siteConfig.title}
            </motion.h1>
            <motion.p
              className={styles.heroSubtitle}
              variants={itemVariants}
            >
              A Modern Textbook on Embodied Intelligence, Simulation, and AI Agents
            </motion.p>
            <motion.div
              className={styles.buttons}
              variants={itemVariants}
            >
              <motion.div variants={itemVariants} whileHover="hover" whileTap="tap" animate="animate">
                <Link
                  className="button button--secondary button--lg"
                  to="/docs/chapters/introduction">
                  Read the Book
                </Link>
              </motion.div>
              <motion.div variants={itemVariants} whileHover="hover" whileTap="tap" animate="animate">
                <Link
                  className="button button--outline button--secondary button--lg"
                  to="/docs/chapters/introduction">
                  View Chapters
                </Link>
              </motion.div>
            </motion.div>
          </motion.div>
          <motion.div
            className={styles.heroVisual}
            variants={floatingVariants}
            animate="animate"
          >
            <div className={styles.robotIllustration}>
              <div className={styles.robotHead}>
                <div className={styles.robotFace}>
                  <div className={styles.robotEyeLeft}>
                    <div className={styles.robotEyeHighlight}></div>
                  </div>
                  <div className={styles.robotEyeRight}>
                    <div className={styles.robotEyeHighlight}></div>
                  </div>
                </div>
                <div className={styles.robotMouth}></div>
              </div>
              <div className={styles.robotNeck}></div>
              <div className={styles.robotBody}>
                <div className={styles.robotChestPanel}></div>
              </div>
              <div className={styles.robotShoulders}></div>
              <div className={styles.robotArmLeft}></div>
              <div className={styles.robotArmRight}></div>
              <div className={styles.robotShoulderJointLeft}>
                <div className={styles.robotJoint}></div>
              </div>
              <div className={styles.robotShoulderJointRight}>
                <div className={styles.robotJoint}></div>
              </div>
              <div className={styles.robotElbowJointLeft}>
                <div className={styles.robotJoint}></div>
              </div>
              <div className={styles.robotElbowJointRight}>
                <div className={styles.robotJoint}></div>
              </div>
              <div className={styles.robotHandLeft}></div>
              <div className={styles.robotHandRight}></div>
              <div className={styles.circuitPattern}></div>
              <div className={styles.glowEffect}></div>
              <div className={styles.aiAura}></div>
            </div>
          </motion.div>
        </div>
      </div>
      <div className={styles.backgroundElements}>
        <div className={styles.gridLines}></div>
        <div className={styles.gradientBlob}></div>
      </div>
    </motion.header>
  );
}

function AboutSection() {
  return (
    <section className={styles.aboutSection}>
      <div className="container padding-horiz--md">
        <div className="row">
          <div className="col col--4">
            <h2>What is Physical AI?</h2>
            <p>
              Physical AI refers to the integration of artificial intelligence capabilities with physical systems,
              enabling robots to understand and interact with the physical world. Unlike traditional AI systems that
              operate in virtual environments, Physical AI systems must navigate the challenges of real-world physics,
              uncertainty, and dynamic environments.
            </p>
          </div>
          <div className="col col--4">
            <h2>Why Humanoid Robotics?</h2>
            <p>
              Humanoid robots, with their human-like form and capabilities, represent one of the most ambitious goals
              in robotics. These systems aim to combine the mobility and dexterity of humans with the computational
              power and reliability of machines, creating more intuitive human-robot interaction.
            </p>
          </div>
          <div className="col col--4">
            <h2>How This Book Helps</h2>
            <p>
              This textbook provides a structured approach from fundamental concepts to advanced implementations.
              Each module builds upon the previous one, with practical examples, code implementations, and
              exercises to reinforce learning and understanding.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function CurriculumSection() {
  const modules = [
    {
      title: 'Module 1: ROS 2 (Robotic Nervous System)',
      description: 'Foundation for robot communication and control',
      chapters: ['ROS 2 Foundations', 'Communication Patterns', 'Navigation Systems']
    },
    {
      title: 'Module 2: Gazebo & Unity (Digital Twin)',
      description: 'Simulation and testing environments',
      chapters: ['Simulation Basics', 'Digital Twins', 'Physics Simulation']
    },
    {
      title: 'Module 3: NVIDIA Isaac (AI-Robot Brain)',
      description: 'AI integration and control systems',
      chapters: ['AI Integration', 'Control Systems']
    },
    {
      title: 'Module 4: Vision-Language-Action (VLA)',
      description: 'Advanced integration of perception and action',
      chapters: ['Vision Systems', 'Language Integration']
    },
    {
      title: 'Module 5: Integration & Applications',
      description: 'Putting it all together with real-world applications',
      chapters: ['System Integration', 'Advanced Applications', 'Future Directions']
    }
  ];

  return (
    <section className={clsx('hero hero--secondary', styles.curriculumSection)}>
      <div className="container padding-horiz--md">
        <h2 className={styles.sectionTitle}>Curriculum Overview</h2>
        <div className="row">
          {modules.map((module, index) => (
            <div className="col col--4 margin-bottom--lg" key={index}>
              <div className={styles.moduleCard}>
                <h3>{module.title}</h3>
                <p className={styles.moduleDescription}>{module.description}</p>
                <ul className={styles.chapterList}>
                  {module.chapters.map((chapter, idx) => (
                    <li key={idx}>{chapter}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics textbook - A Modern Textbook on Embodied Intelligence, Simulation, and AI Agents">
      <HomepageHeader />
      <main>
        <AboutSection />
        <CurriculumSection />
      </main>
    </Layout>
  );
}