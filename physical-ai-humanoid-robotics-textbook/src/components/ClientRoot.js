import React from 'react';
import { TranslationProvider } from '../contexts/TranslationContext';

// Client-only root component to wrap the app with providers
const ClientRoot = ({ children }) => {
  return (
    <TranslationProvider>
      {children}
    </TranslationProvider>
  );
};

export default ClientRoot;