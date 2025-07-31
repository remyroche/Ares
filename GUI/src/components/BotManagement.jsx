import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { 
  Bot, Power, PowerOff, PlusCircle, Trash2, Settings, 
  Activity, Percent, Users, Database, Cpu, AlertCircle, 
  CheckCircle, XCircle, Info, Eye, EyeOff, RefreshCw,
  TrendingUp, TrendingDown, Clock, DollarSign, Target,
  AlertTriangle
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const BotManagement = () => {
  const [bots, setBots] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedBot, setSelectedBot] = useState(null);

  const fetchBots = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/bots`);
      if (!response.ok) throw new Error('Failed to fetch bots');
      const data = await response.json();
      setBots(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBots();
    const interval = setInterval(fetchBots, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [fetchBots]);

  const handleAddBot = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const newBotData = Object.fromEntries(formData.entries());
    try {
      const response = await fetch(`${API_BASE_URL}/api/bots`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newBotData),
      });
      if (!response.ok) throw new Error('Failed to add bot');
      const addedBot = await response.json();
      setBots([...bots, addedBot]);
      setIsModalOpen(false);
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  const handleRemoveBot = async (botId) => {
    if (window.confirm("Are you sure you want to remove this bot?")) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/bots/${botId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to remove bot');
        setBots(bots.filter(bot => bot.id !== botId));
      } catch (err) {
        alert(`Error: ${err.message}`);
      }
    }
  };

  const handleToggleBot = async (botId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/bots/${botId}/toggle`, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to toggle bot status');
      // Optimistically update UI, then refetch
      setBots(bots.map(b => b.id === botId ? {...b, status: b.status === 'running' ? 'stopped' : 'running'} : b));
      fetchBots(); // Re-sync with backend
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  const getStatusPill = (status) => {
    const colors = {
      running: 'bg-green-500/20 text-green-300 border-green-500/30',
      stopped: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
      error: 'bg-red-500/20 text-red-300 border-red-500/30',
    };
    return (
      <div className={`flex items-center gap-2 capitalize text-xs font-medium px-2 py-1 rounded-full border ${colors[status] || 'bg-gray-500/20 text-gray-300'}`}>
        <div className={`w-2 h-2 rounded-full ${status === 'running' ? 'bg-green-400 animate-pulse' : status === 'stopped' ? 'bg-yellow-400' : 'bg-red-400'}`}></div>
        {status}
      </div>
    );
  };

  const renderBotStats = () => {
    const totalBots = bots.length;
    const runningBots = bots.filter(b => b.status === 'running').length;
    const stoppedBots = bots.filter(b => b.status === 'stopped').length;
    const errorBots = bots.filter(b => b.status === 'error').length;
    const totalPnl = bots.reduce((sum, bot) => sum + (bot.pnl || 0), 0);
    const avgWinRate = bots.length > 0 ? bots.reduce((sum, bot) => sum + (bot.winRate || 0), 0) / bots.length : 0;

    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard 
          title="Total Bots" 
          value={totalBots} 
          icon={<Bot className="text-blue-400" />}
          color="text-blue-400"
        />
        <StatCard 
          title="Running" 
          value={runningBots} 
          icon={<CheckCircle className="text-green-400" />}
          color="text-green-400"
        />
        <StatCard 
          title="Stopped" 
          value={stoppedBots} 
          icon={<PowerOff className="text-yellow-400" />}
          color="text-yellow-400"
        />
        <StatCard 
          title="Errors" 
          value={errorBots} 
          icon={<XCircle className="text-red-400" />}
          color="text-red-400"
        />
      </div>
    );
  };

  const renderPerformanceChart = () => {
    const chartData = bots.map(bot => ({
      name: bot.pair,
      pnl: bot.pnl || 0,
      winRate: bot.winRate || 0,
      status: bot.status
    }));

    return (
      <div className="h-80 w-full">
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
            <XAxis dataKey="name" stroke="#9ca3af" fontSize={12} />
            <YAxis stroke="#9ca3af" fontSize={12} />
            <Tooltip contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', borderColor: '#4b5563', borderRadius: '0.5rem' }} />
            <Legend />
            <Bar dataKey="pnl" name="PnL ($)" fill="#8b5cf6" />
            <Bar dataKey="winRate" name="Win Rate (%)" fill="#06b6d4" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  if (error) return <ErrorMessage message="Failed to load bot data." details={error} />;
  if (isLoading) return <LoadingSpinner />;

  return (
    <div className="space-y-8">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Bot Management</h1>
          <p className="text-gray-400">Launch, monitor, and manage your trading bot instances.</p>
        </div>
        <button onClick={() => setIsModalOpen(true)} className="btn-primary flex items-center gap-2">
          <PlusCircle size={18} /> 
          New Bot
        </button>
      </header>

      <Card title="Bot Overview">
        {renderBotStats()}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Bot Performance">
          {renderPerformanceChart()}
        </Card>

        <Card title="Status Distribution">
          <div className="h-80 w-full">
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={[
                    { name: 'Running', value: bots.filter(b => b.status === 'running').length, fill: '#10b981' },
                    { name: 'Stopped', value: bots.filter(b => b.status === 'stopped').length, fill: '#f59e0b' },
                    { name: 'Error', value: bots.filter(b => b.status === 'error').length, fill: '#ef4444' },
                  ]}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                />
                <Tooltip contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', borderColor: '#4b5563', borderRadius: '0.5rem' }} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Bot Instances">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-gray-700 text-gray-400">
              <tr>
                <th className="p-3 font-semibold">Status</th>
                <th className="p-3 font-semibold">Pair</th>
                <th className="p-3 font-semibold">Exchange</th>
                <th className="p-3 font-semibold">Model</th>
                <th className="p-3 font-semibold">PnL</th>
                <th className="p-3 font-semibold">Win Rate</th>
                <th className="p-3 font-semibold">Uptime</th>
                <th className="p-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {bots.map((bot) => (
                <tr key={bot.id} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                  <td className="p-3">{getStatusPill(bot.status)}</td>
                  <td className="p-3 font-mono">{bot.pair}</td>
                  <td className="p-3">{bot.exchange}</td>
                  <td className="p-3">{bot.model}</td>
                  <td className="p-3">
                    <span className={bot.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                      ${bot.pnl?.toFixed(2) || '0.00'}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className="text-blue-400">{bot.winRate?.toFixed(1) || '0.0'}%</span>
                  </td>
                  <td className="p-3 text-gray-400">{bot.uptime}</td>
                  <td className="p-3">
                    <div className="flex gap-2">
                      <button 
                        onClick={() => handleToggleBot(bot.id)} 
                        className={`p-1 rounded-md transition-colors ${
                          bot.status === 'running' 
                            ? 'text-yellow-400 hover:bg-yellow-500/20' 
                            : 'text-green-400 hover:bg-green-500/20'
                        }`}
                        title={bot.status === 'running' ? 'Stop Bot' : 'Start Bot'}
                      >
                        {bot.status === 'running' ? <PowerOff size={16} /> : <Power size={16} />}
                      </button>
                      <button 
                        onClick={() => setSelectedBot(bot)}
                        className="text-blue-400 hover:bg-blue-500/20 p-1 rounded-md"
                        title="View Details"
                      >
                        <Eye size={16} />
                      </button>
                      <button 
                        onClick={() => handleRemoveBot(bot.id)} 
                        className="text-red-400 hover:bg-red-500/20 p-1 rounded-md"
                        title="Remove Bot"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {isModalOpen && (
        <Modal title="Launch New Bot" onClose={() => setIsModalOpen(false)}>
          <form onSubmit={handleAddBot} className="space-y-4">
            <InputGroup label="Token Pair" name="pair" placeholder="e.g., BTC/USDT" required />
            <InputGroup label="Exchange" name="exchange" placeholder="e.g., Binance" required />
            <InputGroup label="Model" name="model" placeholder="e.g., Performer v1.2" required />
            <InputGroup label="Initial Capital" name="capital" type="number" placeholder="10000" step="100" />
            <div className="flex justify-end gap-4 pt-4">
              <button type="button" onClick={() => setIsModalOpen(false)} className="btn-secondary">
                Cancel
              </button>
              <button type="submit" className="btn-primary">
                Launch Bot
              </button>
            </div>
          </form>
        </Modal>
      )}

      {selectedBot && (
        <Modal title={`Bot Details - ${selectedBot.pair}`} onClose={() => setSelectedBot(null)}>
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-white mb-3">Bot Information</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Pair:</span>
                    <span className="text-white font-mono">{selectedBot.pair}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Exchange:</span>
                    <span className="text-white">{selectedBot.exchange}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Model:</span>
                    <span className="text-white">{selectedBot.model}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Status:</span>
                    <span className={`capitalize ${getStatusColor(selectedBot.status)}`}>
                      {selectedBot.status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Uptime:</span>
                    <span className="text-white">{selectedBot.uptime}</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-white mb-3">Performance Metrics</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">PnL:</span>
                    <span className={selectedBot.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                      ${selectedBot.pnl?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Win Rate:</span>
                    <span className="text-blue-400">{selectedBot.winRate?.toFixed(1) || '0.0'}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Trades:</span>
                    <span className="text-white">N/A</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Avg Trade Duration:</span>
                    <span className="text-white">N/A</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex gap-4">
              <button 
                onClick={() => handleToggleBot(selectedBot.id)}
                className={`flex-1 flex items-center justify-center gap-2 ${
                  selectedBot.status === 'running' ? 'btn-danger' : 'btn-success'
                }`}
              >
                {selectedBot.status === 'running' ? (
                  <>
                    <PowerOff size={18} />
                    Stop Bot
                  </>
                ) : (
                  <>
                    <Power size={18} />
                    Start Bot
                  </>
                )}
              </button>
              <button className="btn-secondary flex items-center gap-2 flex-1">
                <Settings size={18} />
                Configure
              </button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};

// Helper Components
const StatCard = ({ title, value, icon, color }) => (
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

const getStatusColor = (status) => {
  switch (status) {
    case 'running': return 'text-green-400';
    case 'stopped': return 'text-yellow-400';
    case 'error': return 'text-red-400';
    default: return 'text-gray-400';
  }
};

export default BotManagement; 