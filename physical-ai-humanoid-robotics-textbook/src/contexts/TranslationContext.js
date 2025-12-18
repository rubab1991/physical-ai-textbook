import React, { createContext, useContext, useState, useEffect } from 'react';

const TranslationContext = createContext();

export const useTranslation = () => {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error('useTranslation must be used within a TranslationProvider');
  }
  return context;
};

export const TranslationProvider = ({ children }) => {
  const [currentLanguage, setCurrentLanguage] = useState('en');

  // Load language preference from localStorage on initial load
  useEffect(() => {
    const savedLanguage = localStorage.getItem('preferredLanguage');
    if (savedLanguage && (savedLanguage === 'en' || savedLanguage === 'ur')) {
      setCurrentLanguage(savedLanguage);
    }
  }, []);

  // Save language preference to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('preferredLanguage', currentLanguage);
  }, [currentLanguage]);

  const toggleLanguage = () => {
    setCurrentLanguage(prev => prev === 'en' ? 'ur' : 'en');
  };

  const setLanguage = (lang) => {
    if (lang === 'en' || lang === 'ur') {
      setCurrentLanguage(lang);
    }
  };

  const value = {
    currentLanguage,
    toggleLanguage,
    setLanguage,
    isUrdu: currentLanguage === 'ur',
    isEnglish: currentLanguage === 'en'
  };

  return (
    <TranslationContext.Provider value={value}>
      {children}
    </TranslationContext.Provider>
  );
};