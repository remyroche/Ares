import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis } from 'recharts';
import { 
  BarChart3, TrendingUp, TrendingDown, DollarSign, Clock, Target, 
  Activity, Percent, Users, Calendar, Filter, Download, Eye, EyeOff,
  AlertTriangle, CheckCircle, XCircle, Info, RefreshCw
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const TradeAnalysis = () => {
  const [trades, setTrades] = useState([]);
  const [summary, setSummary] = useState(null);
  const [selectedTrade, setSelectedTrade] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    days: 30,
    limit: 100,
    pair: 'all',
    side: 'all'
  });

  const fetchTradeAnalysis = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/trades/analysis?days=${filters.days}&limit=${filters.limit}`);
      if (!response.ok) throw new Error('Failed to fetch trade analysis');
      const data = await response.json();
      setTrades(data.trades);
      setSummary(data.summary);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchTradeDetails = async (tradeId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/trades/${tradeId}/detailed`);
      if (!response.ok) throw new Error('Failed to fetch trade details');
      const data = await response.json();
      setSelectedTrade(data);
    } catch (err) {
      console.error('Error fetching trade details:', err);
    }
  };

  useEffect(() => {
    fetchTradeAnalysis();
  }, [filters]);

  const handleTradeClick = (trade) => {
    fetchTradeDetails(trade.trade_id);
  };

  const exportData = () => {
    const dataStr = JSON.stringify({ trades, summary }, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `trade-analysis-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const getStatusIcon = (pnl) => {
    if (pnl > 0) return <CheckCircle size={16} className="text-green-400" />;
    if (pnl < 0) return <XCircle size={16} className="text-red-400" />;
    return <Info size={16} className="text-gray-400" />;
  };

  const getStatusColor = (pnl) => {
    if (pnl > 0) return 'text-green-400';
    if (pnl < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  const renderSummaryCards = () => (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <SummaryCard 
        title="Total Trades" 
        value={summary?.totalTrades || 0} 
        icon={<BarChart3 className="text-blue-400" />}
        color="text-blue-400"
      />
      <SummaryCard 
        title="Win Rate" 
        value={`${summary?.winRate || 0}%`} 
        icon={<Percent className="text-green-400" />}
        color="text-green-400"
      />
      <SummaryCard 
        title="Total PnL" 
        value={`$${summary?.totalPnl?.toFixed(2) || 0}`} 
        icon={<DollarSign className="text-yellow-400" />}
        color="text-yellow-400"
      />
      <SummaryCard 
        title="Avg Duration" 
        value={`${summary?.avgTradeDuration?.toFixed(1) || 0}s`} 
        icon={<Clock className="text-purple-400" />}
        color="text-purple-400"
      />
    </div>
  );

  const renderTradeChart = () => {
    const chartData = trades.map(trade => ({
      date: new Date(trade.entry_time).toLocaleDateString(),
      pnl: trade.pnl,
      volume: trade.volume,
      confidence: trade.confidence * 100
    }));

    return (
      <div className="h-80 w-full">
        <ResponsiveContainer>
          <ScatterChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
            <XAxis dataKey="date" stroke="#9ca3af" fontSize={12} />
            <YAxis stroke="#9ca3af" fontSize={12} />
            <ZAxis dataKey="volume" range={[50, 200]} />
            <Tooltip contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', borderColor: '#4b5563', borderRadius: '0.5rem' }} />
            <Scatter dataKey="pnl" fill="#8b5cf6" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    );
  };

  if (error) return <ErrorMessage message="Failed to load trade analysis." details={error} />;
  if (isLoading) return <LoadingSpinner />;

  return (
    <div className="space-y-8">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Trade Analysis</h1>
          <p className="text-gray-400">Comprehensive analysis of trading performance and patterns.</p>
        </div>
        <button onClick={exportData} className="btn-secondary flex items-center gap-2">
          <Download size={18} />
          Export Data
        </button>
      </header>

      <Card title="Performance Summary">
        {renderSummaryCards()}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card title="Trade Distribution">
            {renderTradeChart()}
          </Card>
        </div>

        <div>
          <Card title="Quick Stats">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Best Trade:</span>
                <span className="text-green-400 font-bold">${summary?.bestTrade?.toFixed(2) || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Worst Trade:</span>
                <span className="text-red-400 font-bold">${summary?.worstTrade?.toFixed(2) || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Avg Win:</span>
                <span className="text-green-400 font-bold">${summary?.avgWin?.toFixed(2) || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Avg Loss:</span>
                <span className="text-red-400 font-bold">${summary?.avgLoss?.toFixed(2) || 0}</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <Card title="Recent Trades">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-gray-700 text-gray-400">
              <tr>
                <th className="p-3 font-semibold">Trade ID</th>
                <th className="p-3 font-semibold">Pair</th>
                <th className="p-3 font-semibold">Side</th>
                <th className="p-3 font-semibold">PnL</th>
                <th className="p-3 font-semibold">Duration</th>
                <th className="p-3 font-semibold">Confidence</th>
                <th className="p-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {trades.slice(0, 10).map((trade) => (
                <tr key={trade.trade_id} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                  <td className="p-3 font-mono text-xs">{trade.trade_id}</td>
                  <td className="p-3">{trade.pair}</td>
                  <td className="p-3">
                    <span className={`capitalize font-semibold ${trade.side === 'long' ? 'text-green-400' : 'text-red-400'}`}>
                      {trade.side}
                    </span>
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(trade.pnl)}
                      <span className={getStatusColor(trade.pnl)}>
                        ${trade.pnl.toFixed(2)}
                      </span>
                    </div>
                  </td>
                  <td className="p-3">{Math.round(trade.duration / 60)}m</td>
                  <td className="p-3">{Math.round(trade.confidence * 100)}%</td>
                  <td className="p-3">
                    <button 
                      onClick={() => handleTradeClick(trade)}
                      className="text-blue-400 hover:text-blue-300 transition-colors"
                    >
                      <Eye size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {selectedTrade && (
        <Modal title={`Trade Details - ${selectedTrade.trade_id}`} onClose={() => setSelectedTrade(null)}>
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-white mb-2">Basic Info</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Pair:</span>
                    <span className="text-white">{selectedTrade.pair}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Side:</span>
                    <span className={`capitalize ${selectedTrade.side === 'long' ? 'text-green-400' : 'text-red-400'}`}>
                      {selectedTrade.side}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Entry Price:</span>
                    <span className="text-white">${selectedTrade.entry_price}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Exit Price:</span>
                    <span className="text-white">${selectedTrade.exit_price}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold text-white mb-2">Performance</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">PnL:</span>
                    <span className={getStatusColor(selectedTrade.pnl)}>
                      ${selectedTrade.pnl.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Duration:</span>
                    <span className="text-white">{Math.round(selectedTrade.duration / 60)}m</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Fees:</span>
                    <span className="text-white">${selectedTrade.fees?.toFixed(2) || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Slippage:</span>
                    <span className="text-white">{selectedTrade.slippage?.toFixed(2) || 0}%</span>
                  </div>
                </div>
              </div>
            </div>

            {selectedTrade.technical_indicators && (
              <div>
                <h4 className="font-semibold text-white mb-2">Technical Indicators</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">RSI:</span>
                    <span className="text-white">{selectedTrade.technical_indicators.rsi?.toFixed(1) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">MACD:</span>
                    <span className="text-white">{selectedTrade.technical_indicators.macd?.toFixed(3) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">BB Position:</span>
                    <span className="text-white">{selectedTrade.technical_indicators.bollinger_position?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Volume SMA:</span>
                    <span className="text-white">{selectedTrade.technical_indicators.volume_sma_ratio?.toFixed(2) || 'N/A'}</span>
                  </div>
                </div>
              </div>
            )}

            {selectedTrade.risk_metrics && (
              <div>
                <h4 className="font-semibold text-white mb-2">Risk Metrics</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Position Size:</span>
                    <span className="text-white">{selectedTrade.risk_metrics.position_size}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Leverage:</span>
                    <span className="text-white">{selectedTrade.risk_metrics.leverage}x</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Risk/Reward:</span>
                    <span className="text-white">{selectedTrade.risk_metrics.risk_reward_ratio?.toFixed(2) || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Max DD:</span>
                    <span className="text-white">${selectedTrade.risk_metrics.max_drawdown?.toFixed(2) || 0}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
};

// Helper Components
const SummaryCard = ({ title, value, icon, color }) => (
  <div className="text-center p-4 bg-gray-700/50 rounded-lg">
    <div className="flex justify-center mb-2">{icon}</div>
    <p className="text-sm text-gray-400">{title}</p>
    <p className={`text-2xl font-bold ${color}`}>{value}</p>
  </div>
);

const Card = ({ title, children }) => (
  <div className="bg-gray-800/50 p-4 sm:p-6 rounded-xl border border-gray-700/50 shadow-lg">
    {title && <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>}
    {children}
  </div>
);

const Modal = ({ title, children, onClose }) => (
  <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex justify-center items-center z-50 p-4">
    <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
      <div className="flex justify-between items-center p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">&times;</button>
      </div>
      <div className="p-6">{children}</div>
    </div>
  </div>
);

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

export default TradeAnalysis; 