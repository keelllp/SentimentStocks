import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { BarChart3, TrendingUp } from 'lucide-react';
import { useStock } from '../context/StockContext';
import { fetchStockData } from '../services/api';
import { StockData } from '../context/StockContext';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const StockChart: React.FC = () => {
  const { state, dispatch } = useStock();
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadStockData = async () => {
      if (!state.selectedStock) return;

      setLoading(true);
      dispatch({ type: 'SET_LOADING', payload: true });
      try {
        const data = await fetchStockData(state.selectedStock);
        setStockData(data);
      } catch (error) {
        console.error('Failed to load stock data:', error);
      } finally {
        setLoading(false);
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    };

    loadStockData();
  }, [state.selectedStock, dispatch]);

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center h-64">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 border-2 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-gray-600">Loading chart data...</span>
          </div>
        </div>
      </div>
    );
  }

  if (!stockData) {
    return (
      <div className="card">
        <div className="flex items-center justify-center h-64 text-gray-500">
          <div className="text-center">
            <BarChart3 className="w-12 h-12 mx-auto mb-2 text-gray-400" />
            <p>No chart data available</p>
          </div>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const chartData = {
    labels: stockData.dates,
    datasets: [
      {
        label: `${state.selectedStock} Price`,
        data: stockData.prices,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: 'rgb(59, 130, 246)',
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
      },
    ],
  };

  // Add prediction point if available
  if (state.predictionResult) {
    const predictionDate = state.predictionResult.prediction_date;
    const predictedPrice = state.predictionResult.predicted_price;
    
    // Add prediction to chart data
    chartData.labels.push(predictionDate);
    chartData.datasets[0].data.push(predictedPrice);
    
         // Add prediction dataset
           chartData.datasets.push({
        label: 'Prediction',
        data: [...Array(stockData.dates.length).fill(null), predictedPrice],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderWidth: 3,
        fill: false,
        tension: 0,
        pointRadius: 6,
        pointHoverRadius: 6,
        pointHoverBackgroundColor: 'rgb(34, 197, 94)',
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
      });
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 20,
        },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(59, 130, 246, 0.5)',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: function(context: any) {
            if (context.dataset.label === 'Stock Price') {
              return `Price: ₹${context.parsed.y.toLocaleString()}`;
            } else if (context.dataset.label === 'Prediction') {
              return `Prediction: ₹${context.parsed.y.toLocaleString()}`;
            }
            return context.dataset.label + ': ' + context.parsed.y;
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Date',
          color: '#6b7280',
        },
        grid: {
          display: false,
        },
        ticks: {
          maxTicksLimit: 8,
          color: '#6b7280',
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Price (₹)',
          color: '#6b7280',
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        ticks: {
          color: '#6b7280',
          callback: function(value: any) {
            return '₹' + value.toLocaleString();
          },
        },
      },
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
    elements: {
      point: {
        hoverRadius: 8,
      },
    },
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Price Chart</h2>
        <TrendingUp className="w-6 h-6 text-primary-600 dark:text-primary-400" />
      </div>
      
      <div className="h-80">
        <Line data={chartData} options={options} />
      </div>
      
      <div className="mt-4 text-sm text-gray-600 dark:text-gray-400 text-center space-y-2">
        <div>
          <span className="font-medium">{state.selectedStock}</span> - Last 100 trading days
          {state.predictionResult && (
            <span className="ml-2 text-success-600 dark:text-success-400">
              • Prediction included for {state.predictionResult.prediction_date}
            </span>
          )}
        </div>
        
        {/* Data Source Information */}
        {stockData && stockData.data_source && (
          <div className={`text-xs px-2 py-1 rounded-full ${
            stockData.data_source.includes('yfinance') 
              ? 'bg-green-100 text-green-800 border border-green-200' 
              : 'bg-yellow-100 text-yellow-800 border border-yellow-200'
          }`}>
            📊 {stockData.data_source}
          </div>
        )}
        
        {/* Warning for old data */}
        {stockData && stockData.data_source && stockData.data_source.includes('CSV') && (
          <div className="p-2 bg-yellow-50 border border-yellow-200 rounded text-yellow-800 text-xs">
            ⚠️ Using historical CSV data. For live data, check your internet connection.
          </div>
        )}
      </div>
    </div>
  );
};

export default StockChart;
