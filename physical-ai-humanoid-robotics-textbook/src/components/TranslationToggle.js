import React from 'react';
import { useTranslation } from '../contexts/TranslationContext';

const TranslationToggle = () => {
  try {
    const { currentLanguage, toggleLanguage, isUrdu, isEnglish } = useTranslation();

    return (
      <div className="translation-toggle" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        marginBottom: '1rem',
        padding: '0.5rem',
        backgroundColor: '#f8f9fa',
        borderRadius: '4px'
      }}>
        <span style={{
          marginRight: '0.5rem',
          fontSize: '0.9rem',
          color: '#495057'
        }}>
          {isEnglish ? 'Language:' : 'زبان:'}
        </span>
        <button
          onClick={toggleLanguage}
          style={{
            padding: '0.25rem 0.75rem',
            border: '1px solid #007bff',
            backgroundColor: currentLanguage === 'ur' ? '#007bff' : '#fff',
            color: currentLanguage === 'ur' ? '#fff' : '#007bff',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '0.9rem',
            fontWeight: currentLanguage === 'ur' ? 'bold' : 'normal'
          }}
          aria-label={isEnglish ? 'Switch to Urdu' : 'اردو میں تبدیل کریں'}
        >
          {isEnglish ? 'UR' : 'EN'}
        </button>
      </div>
    );
  } catch (error) {
    // If TranslationProvider is not available, return a placeholder
    console.warn('TranslationProvider not available, translation toggle will not function');
    return (
      <div className="translation-toggle" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        marginBottom: '1rem',
        padding: '0.5rem',
        backgroundColor: '#f8f9fa',
        borderRadius: '4px',
        opacity: 0.6
      }}>
        <span style={{
          fontSize: '0.9rem',
          color: '#6c757d'
        }}>
          Language: EN
        </span>
      </div>
    );
  }
};

export default TranslationToggle;