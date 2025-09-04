import React, { useState, useEffect } from 'react';
import { 
  Play, Square, Settings, Bot, TestTube2, Database, 
  TrendingUp, Zap, Loader, CheckCircle, XCircle, 
  AlertTriangle, RefreshCw, Monitor, Cpu, HardDrive
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const LauncherControl = () => {
  const [launcherStatus, setLauncherStatus] = useState(null);
  const [trainingModes, setTrainingModes] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [dataStatus, setDataStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState('blank');
  const [selectedSymbol, setSelectedSymbol] = useState('ETHUSDT');
  const [selectedExchange, setSelectedExchange] = useState('BINANCE');
  const [lookbackDays, setLookbackDays] = useState(null);
  const [error, setError] = useState(null);

  const fetchLauncherStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/launcher/status`);
      if (!response.ok) throw new Error('Failed to fetch launcher status');
      const data = await response.json();
      setLauncherStatus(data);
    } catch (err) {
      console.error('Error fetching launcher status:', err);
    }
  };

  const fetchTrainingModes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/training/modes`);
      if (!response.ok) throw new Error('Failed to fetch training modes');
      const data = await response.json();
      setTrainingModes(data);
    } catch (err) {
      console.error('Error fetching training modes:', err);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/training/status`);
      if (!response.ok) throw new Error('Failed to fetch training status');
      const data = await response.json();
      setTrainingStatus(data);
    } catch (err) {
      console.error('Error fetching training status:', err);
    }
  };

  const fetchDataStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/data/status`);
      if (!response.ok) throw new Error('Failed to fetch data status');
      const data = await response.json();
      setDataStatus(data);
    } catch (err) {
      console.error('Error fetching data status:', err);
    }
  };

  useEffect(() => {
    fetchLauncherStatus();
    fetchTrainingModes();
    fetchTrainingStatus();
    fetchDataStatus();
    
    const interval = setInterval(() => {
      fetchLauncherStatus();
      fetchTrainingStatus();
      fetchDataStatus();
    }, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const startLauncherMode = async (mode) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/launcher/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode,
          symbol: selectedSymbol,
          exchange: selectedExchange
        })
      });
      
      if (!response.ok) throw new Error('Failed to start launcher mode');
      
      const result = await response.json();
      alert(`Success: ${result.message}`);
      
      // Refresh status
      await fetchLauncherStatus();
    } catch (err) {
      setError(err.message);
      alert(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const startTraining = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: selectedMode,
          symbol: selectedSymbol,
          exchange: selectedExchange,
          lookback_days: lookbackDays
        })
      });
      
      if (!response.ok) throw new Error('Failed to start training');
      
      const result = await response.json();
      alert(`Success: ${result.message}`);
      
      // Refresh status
      await fetchTrainingStatus();
    } catch (err) {
      setError(err.message);
      alert(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const stopLauncher = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/launcher/stop`, {
        method: 'POST'
      });
      
      if (!response.ok) throw new Error('Failed to stop launcher');
      
      const result = await response.json();
      alert(`Success: ${result.message}`);
      
      // Refresh status
      await fetchLauncherStatus();
    } catch (err) {
      setError(err.message);
      alert(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const getModeIcon = (mode) => {
    switch (mode) {
      case 'paper': return <TestTube2 size={20} />;
      case 'live': return <TrendingUp size={20} />;
      case 'backtest': return <TestTube2 size={20} />;
      case 'blank': return <Bot size={20} />;
      case 'light': return <Zap size={20} />;
      case 'full': return <Settings size={20} />;
      case 'load': return <Database size={20} />;
      case 'precompute': return <Cpu size={20} />;
      default: return <Bot size={20} />;
    }
  };

  const getModeDescription = (mode) => {
    switch (mode) {
      case 'paper': return 'Paper trading for testing strategies';
      case 'live': return 'Live trading with real money';
      case 'backtest': return 'Historical backtesting';
      case 'blank': return 'Standard training mode (180 days)';
      case 'light': return 'Quick training mode (30 days)';
      case 'full': return 'Full training mode (730 days)';
      case 'load': return 'Load and process market data';
      case 'precompute': return 'Precompute wavelet features';
      default: return 'Unknown mode';
    }
  };

  const launcherModes = [
    { id: 'paper', name: 'Paper Trading', color: 'bg-blue-600 hover:bg-blue-700' },
    { id: 'live', name: 'Live Trading', color: 'bg-red-600 hover:bg-red-700' },
    { id: 'backtest', name: 'Backtesting', color: 'bg-purple-600 hover:bg-purple-700' },
    { id: 'load', name: 'Data Loading', color: 'bg-green-600 hover:bg-green-700' },
    { id: 'precompute', name: 'Precompute', color: 'bg-yellow-600 hover:bg-yellow-700' }
  ];

  const trainingModesList = [
    { id: 'light', name: 'Light Training', color: 'bg-green-600 hover:bg-green-700' },
    { id: 'blank', name: 'Blank Training', color: 'bg-blue-600 hover:bg-blue-700' },
    { id: 'full', name: 'Full Training', color: 'bg-purple-600 hover:bg-purple-700' }
  ];

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold text-white">Launcher Control</h1>
        <p className="text-gray-400">Control the Ares trading bot launcher and training processes.</p>
      </header>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-300 p-4 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertTriangle size={20} />
            <span className="font-bold">Error</span>
          </div>
          <p className="text-sm mt-2">{error}</p>
        </div>
      )}

      {/* Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
          <div className="flex items-center gap-3 mb-4">
            <Monitor size={24} className="text-blue-400" />
            <h3 className="text-lg font-semibold text-white">Launcher Status</h3>
          </div>
          {launcherStatus ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                {launcherStatus.launcher_active ? (
                  <CheckCircle size={16} className="text-green-400" />
                ) : (
                  <XCircle size={16} className="text-red-400" />
                )}
                <span className="text-sm">
                  {launcherStatus.launcher_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              <p className="text-xs text-gray-400">
                Processes: {launcherStatus.running_processes?.length || 0}
              </p>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Loader size={16} className="animate-spin text-gray-400" />
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          )}
        </div>

        <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
          <div className="flex items-center gap-3 mb-4">
            <Bot size={24} className="text-purple-400" />
            <h3 className="text-lg font-semibold text-white">Training Status</h3>
          </div>
          {trainingStatus ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                {trainingStatus.training_active ? (
                  <CheckCircle size={16} className="text-green-400" />
                ) : (
                  <XCircle size={16} className="text-red-400" />
                )}
                <span className="text-sm">
                  {trainingStatus.training_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              <p className="text-xs text-gray-400">
                Processes: {trainingStatus.training_processes?.length || 0}
              </p>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Loader size={16} className="animate-spin text-gray-400" />
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          )}
        </div>

        <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
          <div className="flex items-center gap-3 mb-4">
            <HardDrive size={24} className="text-green-400" />
            <h3 className="text-lg font-semibold text-white">Data Status</h3>
          </div>
          {dataStatus ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle size={16} className="text-green-400" />
                <span className="text-sm">Available</span>
              </div>
              <p className="text-xs text-gray-400">
                Files: {dataStatus.data_files?.length || 0}
              </p>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Loader size={16} className="animate-spin text-gray-400" />
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          )}
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
        <h3 className="text-lg font-semibold text-white mb-4">Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Symbol
            </label>
            <input
              type="text"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 focus:ring-purple-500 focus:border-purple-500"
              placeholder="ETHUSDT"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Exchange
            </label>
            <select
              value={selectedExchange}
              onChange={(e) => setSelectedExchange(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 focus:ring-purple-500 focus:border-purple-500"
            >
              <option value="BINANCE">Binance</option>
              <option value="MEXC">MEXC</option>
              <option value="GATEIO">Gate.io</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              Lookback Days (Optional)
            </label>
            <input
              type="number"
              value={lookbackDays || ''}
              onChange={(e) => setLookbackDays(e.target.value ? parseInt(e.target.value) : null)}
              className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 focus:ring-purple-500 focus:border-purple-500"
              placeholder="Auto"
            />
          </div>
        </div>
      </div>

      {/* Launcher Modes */}
      <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
        <h3 className="text-lg font-semibold text-white mb-4">Launcher Modes</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {launcherModes.map((mode) => (
            <button
              key={mode.id}
              onClick={() => startLauncherMode(mode.id)}
              disabled={isLoading}
              className={`${mode.color} text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed flex items-center justify-center gap-2`}
            >
              {isLoading ? <Loader size={18} className="animate-spin" /> : getModeIcon(mode.id)}
              {mode.name}
            </button>
          ))}
        </div>
        <p className="text-sm text-gray-400 mt-4">
          These modes correspond to the ares_launcher.py commands: paper, live, backtest, load, precompute
        </p>
      </div>

      {/* Training Modes */}
      <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
        <h3 className="text-lg font-semibold text-white mb-4">Training Modes</h3>
        
        {trainingModes && (
          <div className="mb-6">
            <h4 className="text-md font-medium text-white mb-3">Available Training Modes</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(trainingModes.modes || {}).map(([modeId, config]) => (
                <div key={modeId} className="bg-gray-700/50 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <input
                      type="radio"
                      id={modeId}
                      name="trainingMode"
                      value={modeId}
                      checked={selectedMode === modeId}
                      onChange={(e) => setSelectedMode(e.target.value)}
                      className="text-purple-600"
                    />
                    <label htmlFor={modeId} className="font-medium text-white capitalize">
                      {modeId} Mode
                    </label>
                  </div>
                  <p className="text-sm text-gray-400 mb-2">{config.description}</p>
                  <div className="text-xs text-gray-500 space-y-1">
                    <div>Lookback: {config.lookback_days} days</div>
                    <div>Duration: ~{config.estimated_duration_minutes} min</div>
                    <div>Intensity: {config.computational_intensity}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex gap-4">
          <button
            onClick={startTraining}
            disabled={isLoading}
            className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? <Loader size={18} className="animate-spin" /> : <Play size={18} />}
            Start Training
          </button>
          
          <button
            onClick={stopLauncher}
            disabled={isLoading}
            className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? <Loader size={18} className="animate-spin" /> : <Square size={18} />}
            Stop All
          </button>
        </div>
      </div>

      {/* Running Processes */}
      {(launcherStatus?.running_processes?.length > 0 || trainingStatus?.training_processes?.length > 0) && (
        <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
          <h3 className="text-lg font-semibold text-white mb-4">Running Processes</h3>
          
          {launcherStatus?.running_processes?.length > 0 && (
            <div className="mb-4">
              <h4 className="text-md font-medium text-white mb-2">Launcher Processes</h4>
              <div className="space-y-2">
                {launcherStatus.running_processes.map((proc, index) => (
                  <div key={index} className="bg-gray-700/50 p-3 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-medium text-white">{proc.name}</span>
                        <span className="text-sm text-gray-400 ml-2">PID: {proc.pid}</span>
                      </div>
                      <span className="text-sm text-green-400">{proc.status}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1 truncate">{proc.cmdline}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {trainingStatus?.training_processes?.length > 0 && (
            <div>
              <h4 className="text-md font-medium text-white mb-2">Training Processes</h4>
              <div className="space-y-2">
                {trainingStatus.training_processes.map((proc, index) => (
                  <div key={index} className="bg-gray-700/50 p-3 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="font-medium text-white">{proc.name}</span>
                        <span className="text-sm text-gray-400 ml-2">PID: {proc.pid}</span>
                      </div>
                      <span className="text-sm text-green-400">{proc.status}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1 truncate">{proc.cmdline}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LauncherControl;