import React from 'react';
import { Zap, Target, TrendingUp } from 'lucide-react';
import { useStock } from '../context/StockContext';
import { makePrediction } from '../services/api';

const PredictionForm: React.FC = () => {
  const { state, dispatch } = useStock();

  const handlePrediction = async () => {
    if (!state.selectedStock) return;

    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'CLEAR_ERROR' });

    try {
      const result = await makePrediction(state.selectedStock);
      dispatch({ type: 'SET_PREDICTION_RESULT', payload: result });
    } catch (error) {
      dispatch({
        type: 'SET_ERROR',
        payload: error instanceof Error ? error.message : 'Prediction failed'
      });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Make Prediction</h2>
        <Target className="w-6 h-6 text-primary-600 dark:text-primary-400" />
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Selected Stock
          </label>
          <div className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600">
            <TrendingUp className="w-5 h-5 text-primary-600 dark:text-primary-400" />
            <span className="font-medium text-gray-900 dark:text-white">{state.selectedStock}</span>
          </div>
        </div>



        <button
          onClick={handlePrediction}
          disabled={state.loading || !state.selectedStock}
          className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {state.loading ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Analyzing...</span>
            </div>
          ) : (
            <div className="flex items-center justify-center space-x-2">
              <Zap className="w-4 h-4" />
              <span>Predict Tomorrow's Price</span>
            </div>
          )}
        </button>
      </div>

      {state.error && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-sm text-red-800 dark:text-red-200">{state.error}</p>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
