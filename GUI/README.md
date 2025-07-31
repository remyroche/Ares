# Ares Trading Bot GUI

A comprehensive web-based dashboard for monitoring and controlling the Ares Trading Bot system.

## Features

### 🎯 Core Features
- **Real-time Dashboard**: Live performance monitoring with portfolio tracking
- **Kill Switch Control**: Emergency stop functionality with reason logging
- **System Management**: CPU, memory, and uptime monitoring
- **Bot Management**: Launch, stop, and monitor trading bot instances

### 📊 Analysis & Testing
- **Backtesting**: Comprehensive strategy testing with detailed metrics
- **A/B Testing**: Model comparison and performance analysis
- **Trade Analysis**: In-depth trade review with performance metrics
- **Model Management**: Deploy and monitor different ML models

### 🔧 Technical Features
- **WebSocket Support**: Real-time updates and notifications
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Theme**: Professional dark interface optimized for trading
- **Interactive Charts**: Advanced visualizations using Recharts

## Quick Start

### Prerequisites
- Node.js 18+ 
- Python 3.8+ (for the API server)
- npm or yarn

### Installation

1. **Install Frontend Dependencies**
   ```bash
   cd GUI
   npm install
   ```

2. **Start the API Server**
   ```bash
   # From the project root
   cd GUI
   python api_server.py
   ```
   The API server will start on `http://localhost:8000`

3. **Start the Frontend**
   ```bash
   cd GUI
   npm run dev
   ```
   The frontend will start on `http://localhost:3000`

4. **Access the Dashboard**
   Open your browser and navigate to `http://localhost:3000`

## API Endpoints

### Dashboard
- `GET /api/dashboard-data` - Get comprehensive dashboard data
- `GET /api/system/status` - Get system health metrics

### Kill Switch
- `GET /api/kill-switch/status` - Get current kill switch status
- `POST /api/kill-switch/activate` - Activate kill switch
- `POST /api/kill-switch/deactivate` - Deactivate kill switch

### Backtesting
- `POST /api/run-backtest` - Run a new backtest
- `GET /api/backtest/comparison` - Get backtest comparison data

### Bot Management
- `GET /api/bots` - Get all configured bots
- `POST /api/bots` - Add a new bot
- `DELETE /api/bots/{id}` - Remove a bot
- `POST /api/bots/{id}/toggle` - Toggle bot status

### Trade Analysis
- `GET /api/trades/analysis` - Get comprehensive trade analysis
- `GET /api/trades/{id}/detailed` - Get detailed trade information

### Model Management
- `GET /api/models` - Get all available models
- `POST /api/models/{id}/deploy` - Deploy a specific model

## Architecture

### Frontend (React + Vite)
- **React 18**: Modern React with hooks and functional components
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Interactive chart library
- **Lucide React**: Icon library

### Backend (FastAPI)
- **FastAPI**: Modern Python web framework
- **WebSocket Support**: Real-time communication
- **Pydantic**: Data validation and serialization
- **SQLite**: Local database for data persistence

## Development

### Project Structure
```
GUI/
├── src/
│   ├── App.jsx          # Main application component
│   ├── main.jsx         # React entry point
│   └── index.css        # Global styles
├── api_server.py        # FastAPI backend server
├── package.json         # Frontend dependencies
├── vite.config.js       # Vite configuration
├── tailwind.config.js   # Tailwind CSS configuration
└── README.md           # This file
```

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Environment Variables
The frontend automatically connects to the API server at `http://localhost:8000`. 
To change this, modify the `API_BASE_URL` constant in `src/App.jsx`.

## Features in Detail

### Kill Switch
The kill switch provides emergency control over the trading system:
- **Visual Indicators**: Red warning indicators when active
- **Reason Logging**: Required reason for activation
- **Real-time Updates**: WebSocket notifications
- **System Integration**: Connects to the actual trading system

### Dashboard
Comprehensive overview of trading performance:
- **Portfolio Performance**: Real-time equity curve
- **Open Positions**: Current position monitoring
- **Recent Trades**: Latest trade history
- **System Status**: Health indicators

### System Management
Advanced system monitoring and control:
- **Resource Monitoring**: CPU and memory usage
- **Uptime Tracking**: System availability
- **Health Checks**: Automated system diagnostics
- **Restart Capability**: System restart functionality

## Security Considerations

- **Local Development**: Designed for local deployment
- **No Authentication**: Assumes secure local network
- **Kill Switch**: Emergency stop functionality
- **Data Validation**: Input validation on all endpoints

## Troubleshooting

### Common Issues

1. **API Server Not Starting**
   - Check Python dependencies: `pip install fastapi uvicorn`
   - Verify port 8000 is available
   - Check console for error messages

2. **Frontend Not Loading**
   - Ensure Node.js 18+ is installed
   - Run `npm install` to install dependencies
   - Check for port conflicts on 3000

3. **Charts Not Rendering**
   - Verify Recharts is installed: `npm install recharts`
   - Check browser console for errors
   - Ensure data format matches expected schema

### Debug Mode
Enable debug logging by setting environment variables:
```bash
export DEBUG=true
export LOG_LEVEL=debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Ares Trading Bot system. See the main project license for details. 