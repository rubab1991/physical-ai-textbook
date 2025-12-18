import React from 'react';
import { useTranslation } from '../contexts/TranslationContext';

const Translation = ({ en, ur }) => {
  const { isUrdu } = useTranslation();

  return <>{isUrdu && ur ? ur : en}</>;
};

export default Translation;