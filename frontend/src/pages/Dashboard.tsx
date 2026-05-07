import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useStock } from '../context/StockContext';
import StockSelector from '../components/StockSelector';
import PredictionForm from '../components/PredictionForm';
import PredictionResult from '../components/PredictionResult';
import StockChart from '../components/StockChart';
import { checkHealth } from '../services/api';

const Dashboard: React.FC = () => {
  const { dispatch } = useStock();

  useEffect(() => {
    // Check API health on component mount
    const checkAPIHealth = async () => {
      try {
        const health = await checkHealth();
        console.log('✅ API Health Check:', health);
      } catch (error) {
        console.error('❌ API Health Check Failed:', error);
        dispatch({ 
          type: 'SET_ERROR', 
          payload: 'Backend API is not responding. Please ensure the Flask server is running.' 
        });
      }
    };

    checkAPIHealth();
  }, [dispatch]);

  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Welcome to <span className="text-gradient">SentimentStocks</span>
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
          Leverage the power of our optimized LightGBM model to predict stock prices with 
          exceptional accuracy. Get real-time predictions for tomorrow's market movements.
        </p>
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Controls */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="space-y-6"
        >
          <StockSelector />
          <PredictionForm />
        </motion.div>

        {/* Right Column - Results and Chart */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="lg:col-span-2 space-y-6"
        >
          <PredictionResult />
          <StockChart />
        </motion.div>
      </div>

      {/* Model Performance Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
        className="card bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200 dark:from-blue-900/20 dark:to-indigo-900/20 dark:border-blue-800"
      >
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Model Performance Metrics
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-100 dark:border-gray-700">
              <div className="text-3xl font-bold text-primary-600 mb-2">12.65</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">RMSE</div>
              <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">Lower is better</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-100 dark:border-gray-700">
              <div className="text-3xl font-bold text-success-600 mb-2">0.9490</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">R² Score</div>
              <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">Higher is better</div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-100 dark:border-gray-700">
              <div className="text-3xl font-bold text-warning-600 mb-2">11.18</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">MAE</div>
              <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">Lower is better</div>
            </div>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
            Our model has been trained on extensive historical data and optimized through 
            10 iterations for maximum accuracy.
          </p>
        </div>
      </motion.div>

      {/* Features Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.8 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        <div className="card text-center">
          <div className="w-12 h-12 bg-primary-100 dark:bg-primary-900/30 rounded-lg flex items-center justify-center mx-auto mb-4">
            <svg className="w-6 h-6 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Real-time Predictions</h3>
          <p className="text-gray-600 dark:text-gray-400">
            Get instant predictions for tomorrow's stock prices using our AI model
          </p>
        </div>

        <div className="card text-center">
          <div className="w-12 h-12 bg-success-100 dark:bg-success-900/30 rounded-lg flex items-center justify-center mx-auto mb-4">
            <svg className="w-6 h-6 text-success-600 dark:text-success-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Interactive Charts</h3>
          <p className="text-gray-600 dark:text-gray-400">
            Visualize stock trends and predictions with beautiful, interactive charts
          </p>
        </div>

        <div className="card text-center">
          <div className="w-12 h-12 bg-warning-100 dark:bg-warning-900/30 rounded-lg flex items-center justify-center mx-auto mb-4">
            <svg className="w-6 h-6 text-warning-600 dark:text-warning-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">High Accuracy</h3>
          <p className="text-gray-600 dark:text-gray-400">
            Powered by our optimized LightGBM model with 94.90% accuracy
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
