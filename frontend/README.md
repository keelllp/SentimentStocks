# SentimentStocks Frontend

A modern, responsive TypeScript React frontend for the Stock Price Prediction System.

## 🚀 Features

- **Modern UI/UX**: Beautiful, responsive design with Tailwind CSS
- **Real-time Predictions**: Get instant stock price predictions
- **Interactive Charts**: Visualize stock data and predictions with Chart.js
- **Stock Selection**: Choose from multiple available stocks
- **Model Performance**: View detailed model metrics and accuracy
- **Mobile Responsive**: Works perfectly on all devices
- **TypeScript**: Full type safety and better development experience

## 🛠️ Tech Stack

- **React 18**: Latest React with modern features
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **Chart.js**: Interactive data visualization
- **Framer Motion**: Smooth animations and transitions
- **Axios**: HTTP client for API communication
- **Lucide React**: Beautiful, consistent icons

## 📦 Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🔧 Development

### Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

### Project Structure

```
frontend/
├── public/                 # Static files
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── Header.tsx     # Navigation header
│   │   ├── StockSelector.tsx    # Stock dropdown
│   │   ├── PredictionForm.tsx   # Prediction form
│   │   ├── PredictionResult.tsx # Results display
│   │   └── StockChart.tsx      # Chart component
│   ├── context/            # React context
│   │   └── StockContext.tsx    # State management
│   ├── pages/              # Page components
│   │   └── Dashboard.tsx       # Main dashboard
│   ├── services/           # API services
│   │   └── api.ts              # HTTP client
│   ├── App.tsx             # Main app component
│   ├── index.tsx           # Entry point
│   └── index.css           # Global styles
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript config
├── tailwind.config.js      # Tailwind config
└── postcss.config.js       # PostCSS config
```

## 🌐 API Integration

The frontend connects to the Flask backend API running on `http://localhost:5000`.

### API Endpoints

- `GET /health` - Health check
- `GET /stocks` - Available stocks
- `GET /stock_data/:symbol` - Stock data for charts
- `POST /predict` - Make prediction
- `GET /performance` - Model performance metrics

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:5000
```

## 🎨 Customization

### Colors

Modify colors in `tailwind.config.js`:

```javascript
colors: {
  primary: {
    50: '#eff6ff',
    500: '#3b82f6',
    600: '#2563eb',
    // ... more shades
  }
}
```

### Components

All components are modular and can be easily customized:

- Modify component props and styling
- Add new features and functionality
- Extend the design system

## 📱 Responsive Design

The application is fully responsive with:

- **Mobile First**: Designed for mobile devices first
- **Breakpoints**: Tailwind CSS responsive breakpoints
- **Flexible Layout**: Grid and flexbox layouts
- **Touch Friendly**: Optimized for touch interactions

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

### Deploy to Static Hosting

The `build` folder contains static files that can be deployed to:

- Netlify
- Vercel
- GitHub Pages
- AWS S3
- Any static hosting service

### Environment Configuration

Ensure your production environment has the correct API URL:

```env
REACT_APP_API_URL=https://your-api-domain.com
```

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure Flask backend is running on port 5000
   - Check CORS configuration
   - Verify network connectivity

2. **Build Errors**
   - Clear `node_modules` and reinstall
   - Check TypeScript configuration
   - Verify all dependencies are installed

3. **Styling Issues**
   - Ensure Tailwind CSS is properly configured
   - Check PostCSS configuration
   - Verify CSS imports

### Development Tips

- Use React Developer Tools for debugging
- Check browser console for errors
- Monitor network requests in DevTools
- Use TypeScript for better error catching

## 📚 Additional Resources

- [React Documentation](https://reactjs.org/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Chart.js Documentation](https://www.chartjs.org/docs/)
- [Framer Motion Documentation](https://www.framer.com/motion/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the SentimentStocks system.

---

**Happy Coding! 🎉**
