# Ares Trading Bot GUI

A comprehensive web-based graphical user interface for the Ares trading bot, providing real-time monitoring, control, and analysis capabilities.

## Features

### 🎛️ Launcher Control
- Start/stop different launcher modes (paper, live, backtest, training, etc.)
- Real-time process monitoring
- Training mode selection with configuration
- Process log viewing

### 📊 Dashboard
- Real-time portfolio performance
- Open positions and trade history
- System status monitoring
- Performance metrics and charts

### 🔧 System Management
- Kill switch control
- System health monitoring
- Process management
- Resource usage tracking

### 📈 Trading & Analysis
- Backtesting interface
- Model management and comparison
- Token configuration
- A/B testing tools

### 📊 Monitoring
- ML model performance tracking
- Drift detection alerts
- Feature importance analysis
- Online learning metrics

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install fastapi uvicorn psutil prometheus-client
   ```

2. **Install Node.js dependencies:**
   ```bash
   cd GUI
   npm install
   ```

### Running the GUI

#### Option 1: Unified Startup Script (Recommended)
```bash
bash GUI/start.sh
```

This will start both the API server and frontend automatically.

#### Option 2: Manual Startup

1. **Start the API server:**
   ```bash
   python GUI/api_server.py
   ```

2. **Start the frontend (in a new terminal):**
   ```bash
   cd GUI
   npm run dev
   ```

3. **Access the GUI:**
   - Frontend: http://localhost:3000
   - API Documentation: http://localhost:8000/docs

## Usage

### Launcher Control

The Launcher Control page allows you to:

1. **Start Launcher Modes:**
   - **Paper Trading**: Test strategies with simulated money
   - **Live Trading**: Execute real trades
   - **Backtesting**: Run historical strategy tests
   - **Data Loading**: Download and process market data
   - **Precompute**: Generate wavelet features for faster backtesting

2. **Start Training:**
   - **Light Training**: Quick testing (30 days, ~15 minutes)
   - **Blank Training**: Standard training (180 days, ~60 minutes)
   - **Full Training**: Production training (730 days, ~240 minutes)

3. **Monitor Processes:**
   - View running processes
   - Check process logs
   - Stop processes individually or all at once

### Configuration

Before starting any mode, configure:

- **Symbol**: Trading pair (e.g., ETHUSDT, BTCUSDT)
- **Exchange**: Trading exchange (BINANCE, MEXC, GATEIO)
- **Lookback Days**: Override default data period (optional)

### Example Workflows

#### 1. Quick Strategy Testing
1. Go to Launcher Control
2. Set symbol to "ETHUSDT", exchange to "BINANCE"
3. Click "Paper Trading" to test your strategy
4. Monitor results in the Dashboard

#### 2. Model Training
1. Go to Launcher Control
2. Select "Blank Training" mode
3. Set symbol and exchange
4. Click "Start Training"
5. Monitor progress in the Training Status section

#### 3. Data Collection
1. Go to Launcher Control
2. Click "Data Loading"
3. Wait for data collection to complete
4. Check data status in the Data Status section

## API Endpoints

The GUI provides a comprehensive REST API:

### Launcher Control
- `GET /api/launcher/status` - Get launcher status
- `POST /api/launcher/start` - Start launcher mode
- `POST /api/launcher/stop` - Stop all processes

### Training
- `GET /api/training/modes` - Get available training modes
- `POST /api/training/start` - Start training
- `GET /api/training/status` - Get training status

### System
- `GET /api/system/status` - Get system status
- `GET /api/kill-switch/status` - Get kill switch status
- `POST /api/kill-switch/activate` - Activate kill switch
- `POST /api/kill-switch/deactivate` - Deactivate kill switch

### Data & Models
- `GET /api/data/status` - Get data collection status
- `GET /api/tokens` - Get configured tokens
- `GET /api/models/available` - Get available models
- `GET /api/models/performance/{symbol}/{exchange}` - Get model performance

### Monitoring
- `GET /api/monitoring/dashboard` - Get monitoring data
- `GET /api/monitoring/drift-alerts` - Get drift alerts
- `GET /api/monitoring/ml-tracker-stats` - Get ML tracking stats

## Architecture

### Frontend (React + Vite)
- **Components**: Modular React components for each feature
- **Styling**: Tailwind CSS for responsive design
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React for consistent iconography

### Backend (FastAPI)
- **API Server**: RESTful API with automatic documentation
- **WebSocket**: Real-time updates and notifications
- **Integration**: Direct integration with ares_launcher.py
- **Monitoring**: Prometheus metrics and health checks

### Integration Layer
- **Launcher Integration**: Direct process management
- **Process Monitoring**: Real-time process tracking
- **Log Collection**: Automatic log aggregation
- **Error Handling**: Comprehensive error management

## Configuration

### Environment Variables

- `API_PORT`: API server port (default: 8000)
- `FRONTEND_PORT`: Frontend port (default: 3000)
- `VITE_API_BASE_URL`: Custom API base URL for frontend

### Customization

The GUI can be customized by modifying:

- **Components**: `GUI/src/components/` - Add new features
- **API Endpoints**: `GUI/api_server.py` - Add new functionality
- **Styling**: `GUI/src/index.css` - Customize appearance
- **Configuration**: `GUI/vite.config.js` - Build settings

## Troubleshooting

### Common Issues

1. **API Server Won't Start**
   - Check if port 8000 is available
   - Verify Python dependencies are installed
   - Check for import errors in the console

2. **Frontend Won't Start**
   - Run `npm install` in the GUI directory
   - Check if port 3000 is available
   - Verify Node.js version (18+ required)

3. **Launcher Integration Not Working**
   - Ensure `ares_launcher.py` exists in project root
   - Check file permissions
   - Verify Python path configuration

4. **Processes Not Starting**
   - Check system resources (CPU, memory)
   - Verify symbol and exchange are valid
   - Check for missing data files

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export LOG_LEVEL=DEBUG
python GUI/api_server.py
```

### Health Checks

Test the GUI functionality:

```bash
python GUI/test_gui.py
```

## Development

### Adding New Features

1. **Frontend Component:**
   ```jsx
   // GUI/src/components/NewFeature.jsx
   import React from 'react';
   
   const NewFeature = () => {
     return <div>New Feature</div>;
   };
   
   export default NewFeature;
   ```

2. **API Endpoint:**
   ```python
   # GUI/api_server.py
   @app.get("/api/new-feature")
   async def new_feature():
       return {"message": "New feature"}
   ```

3. **Add to Navigation:**
   ```jsx
   // GUI/src/App.js
   const navigationItems = [
     // ... existing items
     { id: 'new-feature', label: 'New Feature', icon: NewIcon },
   ];
   ```

### Testing

Run the test suite:

```bash
python GUI/test_gui.py
```

### Building for Production

```bash
cd GUI
npm run build
```

## Security Considerations

- The GUI runs on localhost by default
- API endpoints are not authenticated (local use only)
- Process management requires appropriate system permissions
- Kill switch provides emergency stop functionality

## Performance

- **Real-time Updates**: 10-second refresh intervals
- **Process Monitoring**: Lightweight process tracking
- **Memory Usage**: Optimized for long-running sessions
- **Scalability**: Supports multiple concurrent processes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This GUI is part of the Ares trading bot project and follows the same license terms.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check the console logs for errors
4. Create an issue in the project repository

---

**Note**: This GUI is designed for local use and development. For production deployments, consider additional security measures and authentication.