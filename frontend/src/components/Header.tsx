import React from 'react';
import { TrendingUp, BarChart3, Zap } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gradient">
                SentimentStocks
              </h1>
              <p className="text-sm text-gray-600">
                Powered by Optimized LightGBM
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2 text-gray-600">
              <BarChart3 className="w-5 h-5" />
              <span className="text-sm font-medium">Model Performance</span>
            </div>
            <div className="flex items-center space-x-2 text-gray-600">
              <Zap className="w-5 h-5" />
              <span className="text-sm font-medium">Real-time Predictions</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
