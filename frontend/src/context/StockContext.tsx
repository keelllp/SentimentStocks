import React, { createContext, useContext, useReducer, ReactNode } from 'react';

export interface StockData {
  stock_symbol: string;
  dates: string[];
  prices: number[];
  volumes: number[];
  timestamp: string;
  data_source?: string;
  days_requested?: number;
  days_received?: number;
}

export interface PredictionResult {
  stock_symbol: string;
  current_price: number;
  predicted_price: number;
  price_change: number;
  price_change_pct: number;
  prediction_date: string;
  confidence: string;
  sentiment_score?: number;
  data_source?: string;
  data_range?: {
    start_date: string;
    end_date: string;
    days_analyzed: number;
  };
  model_info: {
    name: string;
    rmse?: number | string;
    r2?: number | string;
    note?: string;
  };
  timestamp: string;
}

export interface StockState {
  selectedStock: string;
  availableStocks: string[];
  stockData: StockData | null;
  predictionResult: PredictionResult | null;
  loading: boolean;
  error: string | null;
}

type StockAction =
  | { type: 'SET_SELECTED_STOCK'; payload: string }
  | { type: 'SET_AVAILABLE_STOCKS'; payload: string[] }
  | { type: 'SET_STOCK_DATA'; payload: StockData | null }
  | { type: 'SET_PREDICTION_RESULT'; payload: PredictionResult | null }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'CLEAR_ERROR' };

const initialState: StockState = {
  selectedStock: 'INFY',
  availableStocks: [],
  stockData: null,
  predictionResult: null,
  loading: false,
  error: null,
};

function stockReducer(state: StockState, action: StockAction): StockState {
  switch (action.type) {
    case 'SET_SELECTED_STOCK':
      return { ...state, selectedStock: action.payload };
    case 'SET_AVAILABLE_STOCKS':
      return { ...state, availableStocks: action.payload };
    case 'SET_STOCK_DATA':
      return { ...state, stockData: action.payload };
    case 'SET_PREDICTION_RESULT':
      return { ...state, predictionResult: action.payload };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    default:
      return state;
  }
}

interface StockContextType {
  state: StockState;
  dispatch: React.Dispatch<StockAction>;
}

const StockContext = createContext<StockContextType | undefined>(undefined);

export function StockProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(stockReducer, initialState);

  return (
    <StockContext.Provider value={{ state, dispatch }}>
      {children}
    </StockContext.Provider>
  );
}

export function useStock() {
  const context = useContext(StockContext);
  if (context === undefined) {
    throw new Error('useStock must be used within a StockProvider');
  }
  return context;
}
