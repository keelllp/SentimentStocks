import React, { useEffect } from 'react';
import { ChevronDown, TrendingUp } from 'lucide-react';
import { useStock } from '../context/StockContext';
import { fetchAvailableStocks } from '../services/api';

const StockSelector: React.FC = () => {
  const { state, dispatch } = useStock();

  useEffect(() => {
    const loadStocks = async () => {
      try {
        const stocks = await fetchAvailableStocks();
        dispatch({ type: 'SET_AVAILABLE_STOCKS', payload: stocks });
      } catch (error) {
        console.error('Failed to load stocks:', error);
      }
    };

    loadStocks();
  }, [dispatch]);

  const handleStockChange = (stockSymbol: string) => {
    dispatch({ type: 'SET_SELECTED_STOCK', payload: stockSymbol });
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">Select Stock</h2>
        <TrendingUp className="w-5 h-5 text-primary-600" />
      </div>
      
      <div className="relative">
        <select
          value={state.selectedStock}
          onChange={(e) => handleStockChange(e.target.value)}
          className="input-field appearance-none pr-10 cursor-pointer"
        >
          {state.availableStocks.map((stock) => (
            <option key={stock} value={stock}>
              {stock}
            </option>
          ))}
        </select>
        
        <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
          <ChevronDown className="w-5 h-5 text-gray-400" />
        </div>
      </div>
      
      <div className="mt-3 text-sm text-gray-600">
        <span className="font-medium">Available:</span> {state.availableStocks.length} stocks
      </div>
    </div>
  );
};

export default StockSelector;
