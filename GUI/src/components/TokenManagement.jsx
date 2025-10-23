import React, { useState, useEffect } from 'react';
import { 
  Plus, 
  Trash2, 
  Settings, 
  BarChart3, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Eye,
  EyeOff,
  RefreshCw,
  Download,
  Filter,
  Search
} from 'lucide-react';

const TokenManagement = () => {
  const [tokens, setTokens] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedToken, setSelectedToken] = useState(null);
  const [modelPerformances, setModelPerformances] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAddToken, setShowAddToken] = useState(false);
  const [showModelAnalysis, setShowModelAnalysis] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelAnalysis, setModelAnalysis] = useState(null);
  const [filterEnabled, setFilterEnabled] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  // Form states
  const [newToken, setNewToken] = useState({
    symbol: '',
    exchange: 'BINANCE',
    enabled: true,
    model_version: ''
  });

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

  useEffect(() => {
    fetchTokens();
    fetchAvailableModels();
  }, []);

  const fetchTokens = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/tokens`);
      const data = await response.json();
      setTokens(data);
    } catch (err) {
      setError('Failed to fetch tokens');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/models/available`);
      const data = await response.json();
      setAvailableModels(data);
    } catch (err) {
      console.error('Failed to fetch available models:', err);
    }
  };

  const fetchModelPerformances = async (symbol, exchange) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/models/performance/${symbol}/${exchange}`);
      const data = await response.json();
      setModelPerformances(data);
    } catch (err) {
      setError('Failed to fetch model performances');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddToken = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/tokens`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newToken)
      });
      
      if (response.ok) {
        setShowAddToken(false);
        setNewToken({ symbol: '', exchange: 'BINANCE', enabled: true, model_version: '' });
        fetchTokens();
      }
    } catch (err) {
      setError('Failed to add token');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveToken = async (symbol, exchange) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/tokens/${symbol}/${exchange}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        fetchTokens();
      }
    } catch (err) {
      setError('Failed to remove token');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectModel = async (symbol, exchange, modelVersion) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/models/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, exchange, model_version: modelVersion })
      });
      
      if (response.ok) {
        fetchTokens();
      }
    } catch (err) {
      setError('Failed to select model');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleViewModelAnalysis = async (symbol, exchange, modelId) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/models/analysis/${symbol}/${exchange}/${modelId}`);
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setModelAnalysis(data);
        setSelectedModel({ symbol, exchange, modelId });
        setShowModelAnalysis(true);
      }
    } catch (err) {
      setError('Failed to fetch model analysis');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const filteredTokens = tokens.filter(token => {
    const matchesSearch = token.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         token.exchange.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = !filterEnabled || token.enabled;
    return matchesSearch && matchesFilter;
  });

  const MetricCard = ({ title, value, subtitle, icon: Icon, color = "blue" }) => (
    <div className={`bg-white rounded-lg p-4 border-l-4 border-${color}-500 shadow-sm`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
        </div>
        <Icon className={`w-8 h-8 text-${color}-500`} />
      </div>
    </div>
  );

  const PerformanceChart = ({ data, title }) => (
    <div className="bg-white rounded-lg p-4 shadow-sm">
      <h3 className="text-lg font-semibold mb-3">{title}</h3>
      <div className="h-32 flex items-end space-x-1">
        {data.map((value, index) => (
          <div
            key={index}
            className={`flex-1 bg-blue-500 rounded-t`}
            style={{ height: `${Math.abs(value) * 10}%` }}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Token Management</h1>
          <p className="text-gray-600">Manage trading tokens and model assignments</p>
        </div>
        <button
          onClick={() => setShowAddToken(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center space-x-2 hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" />
          <span>Add Token</span>
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="Total Tokens"
          value={tokens.length}
          subtitle="Configured tokens"
          icon={BarChart3}
          color="blue"
        />
        <MetricCard
          title="Active Tokens"
          value={tokens.filter(t => t.enabled).length}
          subtitle="Currently trading"
          icon={CheckCircle}
          color="green"
        />
        <MetricCard
          title="Inactive Tokens"
          value={tokens.filter(t => !t.enabled).length}
          subtitle="Disabled tokens"
          icon={XCircle}
          color="red"
        />
        <MetricCard
          title="Models Available"
          value={availableModels.length}
          subtitle="Trained models"
          icon={Settings}
          color="purple"
        />
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg p-4 shadow-sm">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search tokens..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setFilterEnabled(!filterEnabled)}
              className={`px-4 py-2 rounded-lg flex items-center space-x-2 ${
                filterEnabled 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Filter className="w-4 h-4" />
              <span>Active Only</span>
            </button>
            <button
              onClick={fetchTokens}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 flex items-center space-x-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
          </div>
        </div>
      </div>

      {/* Tokens Table */}
      <div className="bg-white rounded-lg shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Configured Tokens</h2>
        </div>
        
        {loading ? (
          <div className="p-8 text-center">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto text-blue-600" />
            <p className="mt-2 text-gray-600">Loading tokens...</p>
          </div>
        ) : error ? (
          <div className="p-8 text-center">
            <AlertTriangle className="w-8 h-8 mx-auto text-red-600" />
            <p className="mt-2 text-red-600">{error}</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Token
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Exchange
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Model
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Last Updated
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredTokens.map((token) => (
                  <tr key={`${token.symbol}_${token.exchange}`}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{token.symbol}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{token.exchange}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        token.enabled 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {token.enabled ? (
                          <>
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Active
                          </>
                        ) : (
                          <>
                            <XCircle className="w-3 h-3 mr-1" />
                            Inactive
                          </>
                        )}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        {token.model_version || 'No model assigned'}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-500">
                        {token.last_updated ? new Date(token.last_updated).toLocaleDateString() : 'N/A'}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => {
                            setSelectedToken(token);
                            fetchModelPerformances(token.symbol, token.exchange);
                          }}
                          className="text-blue-600 hover:text-blue-900 flex items-center space-x-1"
                        >
                          <Settings className="w-4 h-4" />
                          <span>Configure</span>
                        </button>
                        <button
                          onClick={() => handleRemoveToken(token.symbol, token.exchange)}
                          className="text-red-600 hover:text-red-900 flex items-center space-x-1"
                        >
                          <Trash2 className="w-4 h-4" />
                          <span>Remove</span>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Model Performance Modal */}
      {selectedToken && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">
                Model Performance - {selectedToken.symbol} on {selectedToken.exchange}
              </h2>
              <button
                onClick={() => setSelectedToken(null)}
                className="text-gray-500 hover:text-gray-700"
              >
                <XCircle className="w-6 h-6" />
              </button>
            </div>

            {loading ? (
              <div className="text-center py-8">
                <RefreshCw className="w-8 h-8 animate-spin mx-auto text-blue-600" />
                <p className="mt-2 text-gray-600">Loading model performances...</p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Model Selection */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3">Select Model</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {availableModels.map((model) => (
                      <div
                        key={model.model_id}
                        className="bg-white rounded-lg p-4 border-2 border-gray-200 hover:border-blue-500 cursor-pointer"
                        onClick={() => handleSelectModel(selectedToken.symbol, selectedToken.exchange, model.model_id)}
                      >
                        <div className="flex justify-between items-start mb-2">
                          <h4 className="font-semibold">{model.model_name}</h4>
                          <span className="text-sm text-gray-500">
                            {Math.round(model.performance_score * 100)}%
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">{model.description}</p>
                        <div className="text-xs text-gray-500">
                          Last trained: {new Date(model.last_trained).toLocaleDateString()}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Performance Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {modelPerformances.map((perf) => (
                    <div key={perf.model_id} className="bg-white rounded-lg p-4 border border-gray-200">
                      <div className="flex justify-between items-start mb-3">
                        <h4 className="font-semibold text-lg">{perf.model_id.toUpperCase()}</h4>
                        <button
                          onClick={() => handleViewModelAnalysis(selectedToken.symbol, selectedToken.exchange, perf.model_id)}
                          className="text-blue-600 hover:text-blue-800 text-sm flex items-center space-x-1"
                        >
                          <Eye className="w-4 h-4" />
                          <span>Details</span>
                        </button>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Win Rate:</span>
                          <span className="text-sm font-medium">{(perf.win_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Net PnL:</span>
                          <span className={`text-sm font-medium ${perf.net_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            ${perf.net_pnl.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Sharpe Ratio:</span>
                          <span className="text-sm font-medium">{perf.sharpe_ratio.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Max Drawdown:</span>
                          <span className="text-sm font-medium text-red-600">
                            {perf.max_drawdown.toFixed(2)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Total Trades:</span>
                          <span className="text-sm font-medium">{perf.total_trades}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Add Token Modal */}
      {showAddToken && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Add New Token</h2>
              <button
                onClick={() => setShowAddToken(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <XCircle className="w-6 h-6" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Symbol
                </label>
                <input
                  type="text"
                  value={newToken.symbol}
                  onChange={(e) => setNewToken({...newToken, symbol: e.target.value})}
                  placeholder="e.g., BTCUSDT"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Exchange
                </label>
                <select
                  value={newToken.exchange}
                  onChange={(e) => setNewToken({...newToken, exchange: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="BINANCE">BINANCE</option>
                  <option value="BYBIT">BYBIT</option>
                  <option value="OKX">OKX</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model Version
                </label>
                <select
                  value={newToken.model_version}
                  onChange={(e) => setNewToken({...newToken, model_version: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select a model</option>
                  {availableModels.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.model_name} ({Math.round(model.performance_score * 100)}%)
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="enabled"
                  checked={newToken.enabled}
                  onChange={(e) => setNewToken({...newToken, enabled: e.target.checked})}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="enabled" className="ml-2 block text-sm text-gray-900">
                  Enable trading for this token
                </label>
              </div>

              <div className="flex space-x-3 pt-4">
                <button
                  onClick={handleAddToken}
                  disabled={loading || !newToken.symbol}
                  className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Adding...' : 'Add Token'}
                </button>
                <button
                  onClick={() => setShowAddToken(false)}
                  className="flex-1 bg-gray-300 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Analysis Modal */}
      {showModelAnalysis && modelAnalysis && selectedModel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-6xl max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">
                Model Analysis - {selectedModel.modelId.toUpperCase()} on {selectedModel.symbol}
              </h2>
              <button
                onClick={() => setShowModelAnalysis(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <XCircle className="w-6 h-6" />
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Basic Metrics */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Basic Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  <MetricCard
                    title="Total Trades"
                    value={modelAnalysis.basic_metrics.total_trades}
                    icon={BarChart3}
                    color="blue"
                  />
                  <MetricCard
                    title="Win Rate"
                    value={`${(modelAnalysis.basic_metrics.win_rate * 100).toFixed(1)}%`}
                    icon={TrendingUp}
                    color="green"
                  />
                  <MetricCard
                    title="Net PnL"
                    value={`$${modelAnalysis.basic_metrics.net_pnl.toFixed(2)}`}
                    icon={BarChart3}
                    color={modelAnalysis.basic_metrics.net_pnl >= 0 ? "green" : "red"}
                  />
                  <MetricCard
                    title="Sharpe Ratio"
                    value={modelAnalysis.basic_metrics.sharpe_ratio.toFixed(2)}
                    icon={TrendingUp}
                    color="purple"
                  />
                </div>
              </div>

              {/* Trade Analysis */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Trade Analysis</h3>
                <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Best Trade:</span>
                    <span className="text-sm font-medium text-green-600">
                      ${modelAnalysis.trade_analysis.best_trade.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Worst Trade:</span>
                    <span className="text-sm font-medium text-red-600">
                      ${modelAnalysis.trade_analysis.worst_trade.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Avg Win:</span>
                    <span className="text-sm font-medium text-green-600">
                      ${modelAnalysis.trade_analysis.avg_win.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Avg Loss:</span>
                    <span className="text-sm font-medium text-red-600">
                      ${modelAnalysis.trade_analysis.avg_loss.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Consecutive Wins:</span>
                    <span className="text-sm font-medium">{modelAnalysis.trade_analysis.consecutive_wins}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Consecutive Losses:</span>
                    <span className="text-sm font-medium">{modelAnalysis.trade_analysis.consecutive_losses}</span>
                  </div>
                </div>
              </div>

              {/* Risk Metrics */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Risk Metrics</h3>
                <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">VaR (95%):</span>
                    <span className="text-sm font-medium text-red-600">
                      ${modelAnalysis.risk_metrics.var_95.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Max Consecutive Losses:</span>
                    <span className="text-sm font-medium">{modelAnalysis.risk_metrics.max_consecutive_losses}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Recovery Factor:</span>
                    <span className="text-sm font-medium">{modelAnalysis.risk_metrics.recovery_factor.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Calmar Ratio:</span>
                    <span className="text-sm font-medium">{modelAnalysis.risk_metrics.calmar_ratio.toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* Performance Trends */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Performance Trends</h3>
                <PerformanceChart
                  data={modelAnalysis.performance_trends.monthly_returns}
                  title="Monthly Returns (%)"
                />
                <PerformanceChart
                  data={modelAnalysis.performance_trends.rolling_sharpe}
                  title="Rolling Sharpe Ratio"
                />
              </div>
            </div>

            {/* Model Info */}
            <div className="mt-6 bg-gray-50 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3">Model Information</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Model ID:</span>
                  <p className="font-medium">{modelAnalysis.model_info.model_id}</p>
                </div>
                <div>
                  <span className="text-gray-600">Version:</span>
                  <p className="font-medium">{modelAnalysis.model_info.model_version}</p>
                </div>
                <div>
                  <span className="text-gray-600">Training Samples:</span>
                  <p className="font-medium">{modelAnalysis.model_info.training_samples.toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-gray-600">Features:</span>
                  <p className="font-medium">{modelAnalysis.model_info.feature_count}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TokenManagement; 