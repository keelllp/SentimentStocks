import axios from 'axios';
import { StockData, PredictionResult } from '../context/StockContext';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      console.error(`Server Error ${status}:`, data);
      
      if (status === 500) {
        throw new Error('Internal server error. Please try again later.');
      } else if (status === 404) {
        throw new Error('Resource not found. Please check your request.');
      } else if (status === 400) {
        throw new Error(data.error || 'Invalid request. Please check your input.');
      } else {
        throw new Error(data.error || `Server error (${status}). Please try again.`);
      }
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Something else happened
      throw new Error('Request failed. Please try again.');
    }
  }
);

// API functions
export const apiService = {
  // Health check
  async checkHealth(): Promise<any> {
    const response = await api.get('/health');
    return response.data;
  },

  // Get available stocks
  async fetchAvailableStocks(): Promise<string[]> {
    const response = await api.get('/stocks');
    return response.data.stocks || [];
  },

  // Get stock data for charting
  async fetchStockData(stockSymbol: string): Promise<StockData> {
    const response = await api.get(`/stock_data/${stockSymbol}`);
    return response.data;
  },

  // Make prediction
  async makePrediction(stockSymbol: string): Promise<PredictionResult> {
    const response = await api.post('/predict', {
      stock_symbol: stockSymbol,
    });
    return response.data;
  },

  // Get model performance
  async getModelPerformance(): Promise<any> {
    const response = await api.get('/performance');
    return response.data;
  },
};

// Export individual functions for backward compatibility
export const checkHealth = apiService.checkHealth;
export const fetchAvailableStocks = apiService.fetchAvailableStocks;
export const fetchStockData = apiService.fetchStockData;
export const makePrediction = apiService.makePrediction;
export const getModelPerformance = apiService.getModelPerformance;

export default apiService;
