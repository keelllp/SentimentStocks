import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStock } from '../context/StockContext';

const LoadingBar: React.FC = () => {
  const { state } = useStock();

  return (
    <AnimatePresence>
      {state.loading && (
        <motion.div
          key="loading-bar-track"
          className="fixed top-0 left-0 right-0 z-50 h-1 bg-primary-100 dark:bg-primary-900 overflow-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          <motion.div
            className="h-full w-1/2 bg-gradient-to-r from-primary-400 via-primary-500 to-primary-400"
            initial={{ x: '-100%' }}
            animate={{ x: '250%' }}
            transition={{
              repeat: Infinity,
              duration: 1.4,
              ease: 'easeInOut',
            }}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default LoadingBar;
