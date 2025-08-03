import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, AreaChart, Area, PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { 
  ChevronDown, ChevronUp, PlusCircle, Trash2, Play, Settings, Bot, 
  AreaChart as AreaChartIcon, TestTube2, GitCompare, LayoutDashboard, 
  SlidersHorizontal, AlertTriangle, Power, PowerOff, Shield, ShieldOff,
  TrendingUp, TrendingDown, Activity, BarChart3, PieChart as PieChartIcon,
  Target, Zap, Clock, DollarSign, Percent, Users, Database, Cpu,
  AlertCircle, CheckCircle, XCircle, Info, Eye, EyeOff, RefreshCw, Home, Brain, Monitor
} from 'lucide-react';

// Import components
import Backtesting from './components/Backtesting.jsx';
import TradeAnalysis from './components/TradeAnalysis.jsx';
import ModelManagement from './components/ModelManagement.jsx';
import BotManagement from './components/BotManagement.jsx';
import ABTesting from './components/ABTesting.jsx';
import TokenManagement from './components/TokenManagement';
import ModelComparison from './components/ModelComparison';

const API_BASE_URL = 'http://localhost:8000';

// Main App Component
export default function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [killSwitchStatus, setKillSwitchStatus] = useState(false);
  const [systemStatus, setSystemStatus] = useState('healthy');

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <Dashboard killSwitchStatus={killSwitchStatus} systemStatus={systemStatus} />;
      case 'backtesting':
        return <Backtesting />;
      case 'ab-testing':
        return <ABTesting />;
      case 'bots':
        return <BotManagement />;
      case 'models':
        return <ModelManagement />;
      case 'analysis':
        return <TradeAnalysis />;
      case 'token-management':
        return <TokenManagement />;
      case 'model-comparison':
        return <ModelComparison />;
      case 'system':
        return <SystemManagement killSwitchStatus={killSwitchStatus} setKillSwitchStatus={setKillSwitchStatus} />;
      default:
        return <Dashboard killSwitchStatus={killSwitchStatus} systemStatus={systemStatus} />;
    }
  };

  const navigationItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'token-management', label: 'Token Management', icon: Settings },
    { id: 'model-comparison', label: 'Model Comparison', icon: GitCompare },
    { id: 'backtesting', label: 'Backtesting', icon: TestTube2 },
    { id: 'ab-testing', label: 'A/B Testing', icon: GitCompare },
    { id: 'bots', label: 'Bot Management', icon: Bot },
    { id: 'models', label: 'Model Management', icon: Target },
    { id: 'analysis', label: 'Trade Analysis', icon: BarChart3 },
    { id: 'system', label: 'System Control', icon: Settings },
  ];

  return (
    <div className="bg-gray-900 text-gray-200 font-sans flex min-h-screen">
      <Sidebar activePage={activePage} setActivePage={setActivePage} killSwitchStatus={killSwitchStatus} />
      <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto">
        {renderPage()}
      </main>
    </div>
  );
}

// Sidebar Component
const Sidebar = ({ activePage, setActivePage, killSwitchStatus }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'token-management', label: 'Token Management', icon: Settings },
    { id: 'model-comparison', label: 'Model Comparison', icon: GitCompare },
    { id: 'backtesting', label: 'Backtesting', icon: TestTube2 },
    { id: 'ab-testing', label: 'A/B Testing', icon: GitCompare },
    { id: 'bots', label: 'Bot Management', icon: Bot },
    { id: 'models', label: 'Model Management', icon: Target },
    { id: 'analysis', label: 'Trade Analysis', icon: BarChart3 },
    { id: 'system', label: 'System Control', icon: Settings },
  ];

  return (
    <aside className="bg-gray-800/50 backdrop-blur-sm w-16 sm:w-64 p-2 sm:p-4 border-r border-gray-700/50 flex flex-col">
      <div className="flex items-center gap-2 mb-10 p-2">
        <div className="relative">
          <img src="https://placehold.co/40x40/7c3aed/ffffff?text=A" alt="Ares Logo" className="rounded-lg"/>
          {killSwitchStatus && (
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
          )}
        </div>
        <h1 className="text-2xl font-bold text-white hidden sm:block">Ares</h1>
      </div>
      
      <nav className="flex flex-col gap-2 flex-1">
        {navItems.map(item => (
          <button
            key={item.id}
            onClick={() => setActivePage(item.id)}
            className={`flex items-center justify-center sm:justify-start gap-3 p-3 rounded-lg transition-colors ${
              activePage === item.id
                ? 'bg-purple-600/20 text-purple-300'
                : 'hover:bg-gray-700/50 text-gray-400 hover:text-white'
            }`}
          >
            <item.icon size={20} />
            <span className="hidden sm:inline">{item.label}</span>
          </button>
        ))}
      </nav>
      
      <div className="mt-auto hidden sm:block">
        <div className="p-4 bg-gray-800/50 rounded-lg text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <div className={`w-2 h-2 rounded-full ${killSwitchStatus ? 'bg-red-400' : 'bg-green-400'}`}></div>
            <span className="text-sm">{killSwitchStatus ? 'Kill Switch Active' : 'System Active'}</span>
          </div>
          <p className="text-xs text-gray-500">Ares Trading Bot v2.0</p>
        </div>
      </div>
    </aside>
  );
};

// Dashboard Component
const Dashboard = ({ killSwitchStatus, systemStatus }) => {
  const [performanceDays, setPerformanceDays] = useState(7);
  const [dashboardData, setDashboardData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard-data?days=${performanceDays}`);
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      setDashboardData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [performanceDays]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const performanceMetrics = useMemo(() => {
    if(!dashboardData?.performanceCurve?.length) return { change: 0, changePercent: 0 };
    const curve = dashboardData.performanceCurve;
    const startValue = curve[0].portfolioValue;
    const endValue = curve[curve.length - 1].portfolioValue;
    const change = endValue - startValue;
    const changePercent = startValue !== 0 ? (change / startValue) * 100 : 0;
    return { change: change.toFixed(2), changePercent: changePercent.toFixed(2) };
  }, [dashboardData]);
  
  if (error) return <ErrorMessage message="Failed to load dashboard. Is the API server running?" details={error} />;
  if (isLoading && !dashboardData) return <LoadingSpinner />;

  return (
    <div className="space-y-8">
      <header className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-400">Live performance overview and trade monitoring.</p>
        </div>
        <KillSwitchIndicator active={killSwitchStatus} />
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card title="Total PnL (Open)">
          <p className={`text-3xl font-bold ${dashboardData.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${dashboardData.totalPnl.toFixed(2)}
          </p>
        </Card>
        <Card title="Open Positions">
          <p className="text-3xl font-bold text-blue-400">{dashboardData.openPositionsCount}</p>
        </Card>
        <Card title="Running Bots">
          <p className="text-3xl font-bold text-purple-400">{dashboardData.runningBotsCount}</p>
        </Card>
        <Card title="Win Rate (Est.)">
          <p className="text-3xl font-bold text-yellow-400">{dashboardData.winRate}%</p>
        </Card>
      </div>

      <Card>
        <div className="flex justify-between items-center mb-4">
          <div>
            <h3 className="text-lg font-semibold text-white">Portfolio Performance</h3>
            <p className="text-sm text-gray-400">Last {performanceDays} days</p>
          </div>
          <div className="text-right">
            <p className={`text-lg font-bold ${performanceMetrics.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {performanceMetrics.changePercent >= 0 ? '+' : ''}{performanceMetrics.changePercent}%
            </p>
            <p className="text-sm text-gray-400">${performanceMetrics.change}</p>
          </div>
        </div>
        
        {/* Performance Chart */}
        <PerformanceChart 
          data={dashboardData?.performanceCurve} 
          title="Portfolio Performance Over Time" 
        />
      </Card>

      {/* Advanced Analytics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Attribution */}
        <Card title="Performance Attribution">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300">Market Timing</h4>
                <p className="text-2xl font-bold text-blue-400">
                  {dashboardData?.attribution?.market_timing?.contribution?.toFixed(2) || '0.00'}%
                </p>
              </div>
              <div className="bg-gray-700/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300">Stock Selection</h4>
                <p className="text-2xl font-bold text-green-400">
                  {dashboardData?.attribution?.stock_selection?.contribution?.toFixed(2) || '0.00'}%
                </p>
              </div>
              <div className="bg-gray-700/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300">Risk Management</h4>
                <p className="text-2xl font-bold text-yellow-400">
                  {dashboardData?.attribution?.risk_management?.contribution?.toFixed(2) || '0.00'}%
                </p>
              </div>
              <div className="bg-gray-700/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300">Leverage Usage</h4>
                <p className="text-2xl font-bold text-purple-400">
                  {dashboardData?.attribution?.leverage_usage?.contribution?.toFixed(2) || '0.00'}%
                </p>
              </div>
            </div>
            
            {/* Attribution Chart */}
            <AttributionChart 
              data={dashboardData?.attribution} 
              title="Performance Attribution Breakdown" 
            />
          </div>
        </Card>

        {/* Risk Metrics */}
        <Card title="Risk Metrics">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300">VaR (95%)</h4>
                <p className="text-2xl font-bold text-red-400">
                  {dashboardData?.risk_metrics?.var_95?.toFixed(2) || '0.00'}%
                </p>
              </div>
              <div className="bg-gray-700/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300">Expected Shortfall</h4>
                <p className="text-2xl font-bold text-orange-400">
                  {dashboardData?.risk_metrics?.expected_shortfall?.toFixed(2) || '0.00'}%
                </p>
              </div>
              <div className="bg-gray-700/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300">Max Drawdown</h4>
                <p className="text-2xl font-bold text-red-400">
                  {dashboardData?.risk_metrics?.max_drawdown?.toFixed(2) || '0.00'}%
                </p>
              </div>
              <div className="bg-gray-700/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-300">Sharpe Ratio</h4>
                <p className="text-2xl font-bold text-green-400">
                  {dashboardData?.risk_metrics?.sharpe_ratio?.toFixed(2) || '0.00'}
                </p>
              </div>
            </div>
            
            {/* Risk Chart */}
            <RiskMetricsChart 
              data={dashboardData?.risk_metrics} 
              title="Risk Metrics Over Time" 
            />
          </div>
        </Card>
      </div>

      {/* Concept Drift Monitoring */}
      <Card title="Model Health & Concept Drift">
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {dashboardData?.model_health?.map((model, index) => (
              <div key={index} className={`rounded-lg p-4 ${
                model.drift_detected ? 'bg-red-500/20 border border-red-500/30' : 'bg-green-500/20 border border-green-500/30'
              }`}>
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-white">{model.name}</h4>
                  <div className={`w-2 h-2 rounded-full ${
                    model.drift_detected ? 'bg-red-400 animate-pulse' : 'bg-green-400'
                  }`}></div>
                </div>
                <p className="text-sm text-gray-400 mt-1">Accuracy: {model.accuracy?.toFixed(2) || '0.00'}%</p>
                {model.drift_detected && (
                  <p className="text-xs text-red-400 mt-1">Drift detected</p>
                )}
              </div>
            )) || (
              <div className="text-gray-400">No model health data available</div>
            )}
          </div>
        </div>
      </Card>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Open Positions">
          <Table
            columns={[
              { header: 'Pair', accessor: 'pair' }, 
              { header: 'Side', accessor: 'side' }, 
              { header: 'Size', accessor: 'size' }, 
              { header: 'Entry', accessor: 'entryPrice' }, 
              { header: 'PnL ($)', accessor: 'pnl' }
            ]}
            data={dashboardData.openPositions}
            renderCell={(item, column) => {
              if (column.accessor === 'pnl') return <span className={item.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>{item.pnl.toFixed(2)}</span>;
              if (column.accessor === 'side') return <span className={`capitalize font-semibold ${item.side === 'long' ? 'text-green-400' : 'text-red-400'}`}>{item.side}</span>;
              return item[column.accessor];
            }}
          />
        </Card>

        <Card title="Last 10 Trades">
          <Table
            columns={[
              { header: 'Pair', accessor: 'pair' }, 
              { header: 'PnL ($)', accessor: 'pnl' }, 
              { header: 'Date', accessor: 'date' }
            ]}
            data={dashboardData.lastTrades}
            renderCell={(item, column) => {
              if (column.accessor === 'pnl') return <span className={item.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>{item.pnl.toFixed(2)}</span>;
              if (column.accessor === 'date') return <span>{new Date(item.date).toLocaleString()}</span>;
              return item[column.accessor];
            }}
          />
        </Card>
      </div>
    </div>
  );
};

// Kill Switch Indicator Component
const KillSwitchIndicator = ({ active }) => (
  <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${
    active 
      ? 'bg-red-500/20 text-red-300 border-red-500/30' 
      : 'bg-green-500/20 text-green-300 border-green-500/30'
  }`}>
    {active ? <Shield size={20} /> : <ShieldOff size={20} />}
    <span className="font-semibold">{active ? 'Kill Switch Active' : 'System Active'}</span>
  </div>
);

// System Management Component
const SystemManagement = ({ killSwitchStatus, setKillSwitchStatus }) => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [killSwitchReason, setKillSwitchReason] = useState('');

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/system/status`);
      if (!response.ok) throw new Error('Failed to fetch system status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Error fetching system status:', error);
    }
  };

  const activateKillSwitch = async () => {
    if (!killSwitchReason.trim()) {
      alert('Please provide a reason for activating the kill switch');
      return;
    }
    
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/kill-switch/activate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: killSwitchReason, emergency: true })
      });
      
      if (!response.ok) throw new Error('Failed to activate kill switch');
      
      setKillSwitchStatus(true);
      setKillSwitchReason('');
      alert('Kill switch activated successfully');
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const deactivateKillSwitch = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/kill-switch/deactivate`, {
        method: 'POST'
      });
      
      if (!response.ok) throw new Error('Failed to deactivate kill switch');
      
      setKillSwitchStatus(false);
      alert('Kill switch deactivated successfully');
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold text-white">System Control</h1>
        <p className="text-gray-400">Monitor and control the trading system.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Kill Switch Panel */}
        <Card title="Kill Switch Control">
          <div className="space-y-4">
            <div className={`p-4 rounded-lg border ${
              killSwitchStatus 
                ? 'bg-red-500/20 border-red-500/30' 
                : 'bg-green-500/20 border-green-500/30'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                {killSwitchStatus ? <Shield size={20} className="text-red-400" /> : <ShieldOff size={20} className="text-green-400" />}
                <span className="font-semibold">
                  {killSwitchStatus ? 'Kill Switch Active' : 'System Active'}
                </span>
              </div>
              <p className="text-sm text-gray-400">
                {killSwitchStatus ? 'Trading is currently halted' : 'Trading is active and running normally'}
              </p>
            </div>

            {!killSwitchStatus && (
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Reason for activation
                  </label>
                  <textarea
                    value={killSwitchReason}
                    onChange={(e) => setKillSwitchReason(e.target.value)}
                    placeholder="Enter reason for activating kill switch..."
                    className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 focus:ring-red-500 focus:border-red-500"
                    rows={3}
                  />
                </div>
                <button
                  onClick={activateKillSwitch}
                  disabled={isLoading || !killSwitchReason.trim()}
                  className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isLoading ? <RefreshCw size={18} className="animate-spin" /> : <Shield size={18} />}
                  Activate Kill Switch
                </button>
              </div>
            )}

            {killSwitchStatus && (
              <button
                onClick={deactivateKillSwitch}
                disabled={isLoading}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? <RefreshCw size={18} className="animate-spin" /> : <ShieldOff size={18} />}
                Deactivate Kill Switch
              </button>
            )}
          </div>
        </Card>

        {/* System Status Panel */}
        <Card title="System Status">
          {systemStatus ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-gray-700/50 rounded-lg">
                  <Cpu size={24} className="mx-auto mb-2 text-blue-400" />
                  <p className="text-sm text-gray-400">CPU Usage</p>
                  <p className="text-lg font-bold text-white">{systemStatus.cpu_usage?.toFixed(1) || 0}%</p>
                </div>
                <div className="text-center p-3 bg-gray-700/50 rounded-lg">
                  <Database size={24} className="mx-auto mb-2 text-green-400" />
                  <p className="text-sm text-gray-400">Memory Usage</p>
                  <p className="text-lg font-bold text-white">{systemStatus.memory_usage?.percent?.toFixed(1) || 0}%</p>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Status:</span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    systemStatus.status === 'running' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                  }`}>
                    {systemStatus.status}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Uptime:</span>
                  <span className="text-white">{systemStatus.uptime}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Last Heartbeat:</span>
                  <span className="text-white">{new Date(systemStatus.last_heartbeat).toLocaleTimeString()}</span>
                </div>
              </div>
            </div>
          ) : (
            <LoadingSpinner />
          )}
        </Card>
      </div>
    </div>
  );
};

// Reusable UI Components
const Card = ({ title, children }) => (
  <div className="bg-gray-800/50 p-4 sm:p-6 rounded-xl border border-gray-700/50 shadow-lg">
    {title && <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>}
    {children}
  </div>
);

const Table = ({ columns, data, renderCell }) => (
  <div className="overflow-x-auto">
    <table className="w-full text-left text-sm">
      <thead className="border-b border-gray-700 text-gray-400">
        <tr>
          {columns.map(c => <th key={c.accessor} className="p-3 font-semibold">{c.header}</th>)}
        </tr>
      </thead>
      <tbody>
        {data.map((item, index) => (
          <tr key={item.id || index} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
            {columns.map(c => (
              <td key={c.accessor} className="p-3">
                {renderCell ? renderCell(item, c) : item[c.accessor]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const InputGroup = ({ label, ...props }) => (
  <div>
    <label htmlFor={props.id || props.name} className="block text-sm font-medium text-gray-300 mb-1">
      {label}
    </label>
    <input 
      {...props} 
      className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 focus:ring-purple-500 focus:border-purple-500 transition" 
    />
  </div>
);

const RadioInput = ({ name, label, ...props }) => (
  <div className="flex items-center">
    <input 
      name={name} 
      type="radio" 
      {...props} 
      className="h-4 w-4 text-purple-600 bg-gray-700 border-gray-600 focus:ring-purple-500" 
    />
    <label htmlFor={props.id} className="ml-2 block text-sm text-gray-300">{label}</label>
  </div>
);

const Modal = ({ title, children, onClose }) => (
  <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex justify-center items-center z-50 p-4">
    <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-2xl w-full max-w-md">
      <div className="flex justify-between items-center p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">&times;</button>
      </div>
      <div className="p-6">{children}</div>
    </div>
  </div>
);

const CollapsibleSection = ({ title, children }) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="border border-gray-700 rounded-lg">
      <button 
        type="button" 
        onClick={() => setIsOpen(!isOpen)} 
        className="w-full flex justify-between items-center p-3 text-left font-semibold text-gray-200 hover:bg-gray-700/50"
      >
        {title}
        {isOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </button>
      {isOpen && <div className="p-4 border-t border-gray-700">{children}</div>}
    </div>
  );
};

const Spinner = () => <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>;

const LoadingSpinner = () => (
  <div className="flex justify-center items-center h-full w-full">
    <div className="w-8 h-8 border-4 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
  </div>
);

const ErrorMessage = ({ message, details }) => (
  <div className="bg-red-500/10 border border-red-500/30 text-red-300 p-4 rounded-lg">
    <div className="flex items-center gap-2">
      <AlertTriangle size={20} />
      <h4 className="font-bold">{message}</h4>
    </div>
    {details && <p className="text-sm mt-2 font-mono bg-red-900/20 p-2 rounded">{details}</p>}
  </div>
);

// Loading Spinner Component


// Advanced Chart Components
const PerformanceChart = ({ data, title }) => {
  if (!data || data.length === 0) return <div className="text-gray-400">No data available</div>;
  
  return (
    <div className="space-y-4">
      <h4 className="text-white font-medium">{title}</h4>
      <div className="h-64 bg-gray-900/50 rounded-lg p-4">
        {/* Placeholder for actual chart implementation */}
        <div className="flex items-center justify-center h-full text-gray-400">
          <LineChart size={48} />
          <span className="ml-2">Performance Chart</span>
        </div>
      </div>
    </div>
  );
};

const AttributionChart = ({ data, title }) => {
  if (!data || Object.keys(data).length === 0) return <div className="text-gray-400">No attribution data available</div>;
  
  return (
    <div className="space-y-4">
      <h4 className="text-white font-medium">{title}</h4>
      <div className="h-64 bg-gray-900/50 rounded-lg p-4">
        {/* Placeholder for actual chart implementation */}
        <div className="flex items-center justify-center h-full text-gray-400">
          <PieChart size={48} />
          <span className="ml-2">Attribution Chart</span>
        </div>
      </div>
    </div>
  );
};

const RiskMetricsChart = ({ data, title }) => {
  if (!data || Object.keys(data).length === 0) return <div className="text-gray-400">No risk data available</div>;
  
  return (
    <div className="space-y-4">
      <h4 className="text-white font-medium">{title}</h4>
      <div className="h-64 bg-gray-900/50 rounded-lg p-4">
        {/* Placeholder for actual chart implementation */}
        <div className="flex items-center justify-center h-full text-gray-400">
          <BarChart3 size={48} />
          <span className="ml-2">Risk Metrics Chart</span>
        </div>
      </div>
    </div>
  );
};
