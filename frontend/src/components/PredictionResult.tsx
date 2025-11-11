import React, { useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, Target, BarChart3, Clock } from 'lucide-react';
import { useStock } from '../context/StockContext';
import { PredictionResult as PredictionResultType } from '../context/StockContext';

const PredictionResult: React.FC = () => {
  const { state } = useStock();
  const result = state.predictionResult as PredictionResultType;

  if (!result) return null;

  useEffect(() =>{
    console.log("Rendering PredictionResult with data:", result);
  })

  const getPriceChangeIcon = () => {
    if (result.price_change > 0) {
      return <TrendingUp className="w-6 h-6 text-success-600" />;
    } else if (result.price_change < 0) {
      return <TrendingDown className="w-6 h-6 text-danger-600" />;
    } else {
      return <Minus className="w-6 h-6 text-gray-600" />;
    }
  };

  const getPriceChangeColor = () => {
    if (result.price_change > 0) return 'text-success-600';
    if (result.price_change < 0) return 'text-danger-600';
    return 'text-gray-600';
  };

  const getConfidenceBadge = () => {
    switch (result.confidence.toLowerCase()) {
      case 'high':
        return <span className="badge badge-success">High Confidence</span>;
      case 'medium':
        return <span className="badge badge-warning">Medium Confidence</span>;
      case 'low':
        return <span className="badge badge-danger">Low Confidence</span>;
      default:
        return <span className="badge badge-warning">Unknown</span>;
    }
  };

  return (
    <div className="card animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Prediction Result</h2>
        <Target className="w-6 h-6 text-primary-600" />
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Price Information */}
        <div className="space-y-4">
          <div className="bg-gradient-to-br from-primary-50 to-primary-100 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-primary-700">Current Price</span>
              <Clock className="w-4 h-4 text-primary-600" />
            </div>
            <div className="text-2xl font-bold text-primary-900">
              ₹{result.current_price.toLocaleString()}
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-success-50 to-success-100 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-success-700">Predicted Price</span>
              <Target className="w-4 h-4 text-success-600" />
            </div>
            <div className="text-2xl font-bold text-success-900">
              ₹{result.predicted_price.toLocaleString()}
            </div>
            <div className="text-sm text-success-700 mt-1">
              For {result.prediction_date}
            </div>
          </div>
        </div>
        
        {/* Change Information */}
        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-gray-700">Price Change</span>
              {getPriceChangeIcon()}
            </div>
            <div className={`text-2xl font-bold ${getPriceChangeColor()}`}>
              {result.price_change > 0 ? '+' : ''}₹{result.price_change.toFixed(2)}
            </div>
            <div className={`text-lg font-medium ${getPriceChangeColor()}`}>
              {result.price_change > 0 ? '+' : ''}{result.price_change_pct.toFixed(2)}%
            </div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-gray-700">Confidence</span>
              <BarChart3 className="w-4 h-4 text-gray-600" />
            </div>
            <div className="flex items-center space-x-2">
              {getConfidenceBadge()}
            </div>
          </div>
          
          {result.sentiment_score !== undefined && (
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-gray-700">Sentiment Score</span>
                <span className="text-sm text-gray-500">📰</span>
              </div>
              <div className={`text-lg font-medium ${result.sentiment_score > 0 ? 'text-success-600' : result.sentiment_score < 0 ? 'text-danger-600' : 'text-gray-600'}`}>
                {result.sentiment_score > 0 ? '+' : ''}{result.sentiment_score.toFixed(3)}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {result.sentiment_score > 0 ? 'Positive' : result.sentiment_score < 0 ? 'Negative' : 'Neutral'}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Model Information */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-start space-x-3">
          <BarChart3 className="w-5 h-5 text-blue-600 mt-0.5" />
          <div className="text-sm text-blue-800">
            <p className="font-medium">Model Performance</p>
            <div className="grid grid-cols-2 gap-4 mt-2">
              <div>
                <span className="font-medium">RMSE:</span> {result.model_info.rmse || 'N/A'}
              </div>
              <div>
                <span className="font-medium">R²:</span> {result.model_info.r2 || 'N/A'}
              </div>
            </div>
            {result.model_info.note && (
              <div className="mt-2 text-xs text-blue-600">
                {result.model_info.note}
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Data Source and Timestamp */}
      <div className="mt-4 text-xs text-gray-500 text-center space-y-1">
        {result.data_source && (
          <div className="text-blue-600 font-medium">
            📊 Data Source: {result.data_source}
          </div>
        )}
        <div>
          Prediction made at {new Date(result.timestamp).toLocaleString()}
        </div>
      </div>
      
      {/* Warning for old data */}
      {result.data_source && result.data_source.includes('CSV') && (
        <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center space-x-2 text-yellow-800">
            <span className="text-sm">⚠️</span>
            <span className="text-sm font-medium">Historical Data Warning</span>
          </div>
          <p className="text-xs text-yellow-700 mt-1">
            This prediction is based on historical CSV data, not live market data. 
            For accurate predictions, ensure your internet connection allows access to live stock data.
          </p>
        </div>
      )}
    </div>
  );
};

export default PredictionResult;

