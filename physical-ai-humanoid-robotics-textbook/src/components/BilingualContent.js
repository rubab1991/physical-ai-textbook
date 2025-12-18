import React from 'react';
import { useTranslation } from '../contexts/TranslationContext';
import TranslationToggle from '../components/TranslationToggle';

const BilingualContent = ({ children, en, ur }) => {
  // Handle case where TranslationProvider is not available
  let isUrdu = false;
  try {
    const translationContext = useTranslation();
    isUrdu = translationContext.isUrdu;
  } catch (error) {
    console.warn('TranslationProvider not available in BilingualContent');
  }

  // If content is provided as children, use it; otherwise use the props
  if (children) {
    const childArray = React.Children.toArray(children);
    const enContent = childArray[0];
    const urContent = childArray[1] || enContent; // Fallback to English if Urdu not provided

    return (
      <div className="bilingual-content">
        <TranslationToggle />
        <div className="content">
          {isUrdu ? urContent : enContent}
        </div>
      </div>
    );
  }

  // If content is provided as props
  return (
    <div className="bilingual-content">
      <TranslationToggle />
      <div className="content">
        {isUrdu && ur ? ur : en}
      </div>
    </div>
  );
};

export default BilingualContent;