import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Eye,
  RefreshCw,
  Download,
  Filter,
  Search,
  GitCompare,
  Award,
  Activity
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line
} from 'recharts';

const ModelComparison = () => {
  const [tokens, setTokens] = useState([]);
  const [selectedToken, setSelectedToken] = useState(null);
  const [modelPerformances, setModelPerformances] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showComparison, setShowComparison] = useState(false);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

  useEffect(() => {
    fetchTokens();
  }, []);

  const fetchTokens = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/tokens`);
      const data = await response.json();
      setTokens(data.filter(token => token.enabled));
    } catch (err) {
      setError('Failed to fetch tokens');
      console.error(err);
    } finally {
      setLoading(false);
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

  const handleTokenSelect = (token) => {
    setSelectedToken(token);
    fetchModelPerformances(token.symbol, token.exchange);
    setSelectedModels([]);
    setComparisonData(null);
  };

  const handleModelSelect = (modelId) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        return prev.filter(id => id !== modelId);
      } else if (prev.length < 3) {
        return [...prev, modelId];
      }
      return prev;
    });
  };

  const compareModels = async () => {
    if (selectedModels.length < 2) {
      setError('Please select at least 2 models to compare');
      return;
    }

    try {
      setLoading(true);
      const modelA = selectedModels[0];
      const modelB = selectedModels[1];
      
      const response = await fetch(
        `${API_BASE_URL}/api/models/compare/${selectedToken.symbol}/${selectedToken.exchange}?model_a=${modelA}&model_b=${modelB}`
      );
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setComparisonData(data);
        setShowComparison(true);
      }
    } catch (err) {
      setError('Failed to compare models');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const MetricCard = ({ title, value, subtitle, icon: Icon, color = "blue", trend = null }) => (
    <div className={`bg-white rounded-lg p-4 border-l-4 border-${color}-500 shadow-sm`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
          {trend && (
            <div className="flex items-center mt-1">
              {trend > 0 ? (
                <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
              )}
              <span className={`text-xs ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {Math.abs(trend).toFixed(2)}%
              </span>
            </div>
          )}
        </div>
        <Icon className={`w-8 h-8 text-${color}-500`} />
      </div>
    </div>
  );

  const ComparisonChart = ({ data, title, dataKey, color = "#3b82f6" }) => (
    <div className="bg-white rounded-lg p-4 shadow-sm">
      <h3 className="text-lg font-semibold mb-3">{title}</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="model" stroke="#6b7280" />
            <YAxis stroke="#6b7280" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                border: '1px solid #e5e7eb',
                borderRadius: '0.5rem'
              }} 
            />
            <Legend />
            <Bar dataKey={dataKey} fill={color} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const RadarChartComponent = ({ data, title }) => (
    <div className="bg-white rounded-lg p-4 shadow-sm">
      <h3 className="text-lg font-semibold mb-3">{title}</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data}>
            <PolarGrid stroke="#e5e7eb" />
            <PolarAngleAxis dataKey="metric" stroke="#6b7280" />
            <PolarRadiusAxis stroke="#6b7280" />
            <Radar 
              name="Model A" 
              dataKey="modelA" 
              stroke="#3b82f6" 
              fill="#3b82f6" 
              fillOpacity={0.3} 
            />
            <Radar 
              name="Model B" 
              dataKey="modelB" 
              stroke="#ef4444" 
              fill="#ef4444" 
              fillOpacity={0.3} 
            />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Comparison</h1>
          <p className="text-gray-600">Compare model performance and select the best one</p>
        </div>
        <button
          onClick={fetchTokens}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center space-x-2 hover:bg-blue-700"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Token Selection */}
      <div className="bg-white rounded-lg p-4 shadow-sm">
        <h2 className="text-lg font-semibold mb-3">Select Token</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {tokens.map((token) => (
            <button
              key={`${token.symbol}_${token.exchange}`}
              onClick={() => handleTokenSelect(token)}
              className={`p-3 rounded-lg border-2 text-left transition-colors ${
                selectedToken?.symbol === token.symbol && selectedToken?.exchange === token.exchange
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="font-medium">{token.symbol}</div>
              <div className="text-sm text-gray-500">{token.exchange}</div>
              {token.model_version && (
                <div className="text-xs text-blue-600 mt-1">
                  Model: {token.model_version}
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Model Selection */}
      {selectedToken && modelPerformances.length > 0 && (
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <h2 className="text-lg font-semibold mb-3">
            Select Models to Compare ({selectedModels.length}/3)
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {modelPerformances.map((model) => (
              <div
                key={model.model_id}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                  selectedModels.includes(model.model_id)
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleModelSelect(model.model_id)}
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-semibold">{model.model_id.toUpperCase()}</h3>
                  {selectedModels.includes(model.model_id) && (
                    <CheckCircle className="w-5 h-5 text-blue-600" />
                  )}
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Win Rate:</span>
                    <span className="font-medium">{(model.win_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Net PnL:</span>
                    <span className={`font-medium ${model.net_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      ${model.net_pnl.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Sharpe:</span>
                    <span className="font-medium">{model.sharpe_ratio.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Trades:</span>
                    <span className="font-medium">{model.total_trades}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {selectedModels.length >= 2 && (
            <div className="mt-4 flex justify-center">
              <button
                onClick={compareModels}
                disabled={loading}
                className="bg-green-600 text-white px-6 py-2 rounded-lg flex items-center space-x-2 hover:bg-green-700 disabled:opacity-50"
              >
                <GitCompare className="w-4 h-4" />
                <span>{loading ? 'Comparing...' : 'Compare Models'}</span>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Loading and Error States */}
      {loading && (
        <div className="text-center py-8">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto text-blue-600" />
          <p className="mt-2 text-gray-600">Loading...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
            <span className="text-red-800">{error}</span>
          </div>
        </div>
      )}

      {/* Comparison Results Modal */}
      {showComparison && comparisonData && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-6xl max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">
                Model Comparison - {selectedToken.symbol} on {selectedToken.exchange}
              </h2>
              <button
                onClick={() => setShowComparison(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <XCircle className="w-6 h-6" />
              </button>
            </div>

            {/* Winner Announcement */}
            <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-center">
                <Award className="w-8 h-8 text-yellow-600 mr-3" />
                <div className="text-center">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Winner: {comparisonData.winner?.toUpperCase()}
                  </h3>
                  <p className="text-sm text-gray-600">
                    Confidence: {comparisonData.confidence.toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>

            {/* Comparison Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Basic Metrics Comparison */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Performance Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  <MetricCard
                    title="Win Rate Diff"
                    value={`${(comparisonData.comparison_metrics.win_rate_diff * 100).toFixed(1)}%`}
                    subtitle="Model A vs Model B"
                    icon={TrendingUp}
                    color={comparisonData.comparison_metrics.win_rate_diff > 0 ? "green" : "red"}
                    trend={comparisonData.comparison_metrics.win_rate_diff * 100}
                  />
                  <MetricCard
                    title="PnL Difference"
                    value={`$${comparisonData.comparison_metrics.pnl_diff.toFixed(2)}`}
                    subtitle="Model A vs Model B"
                    icon={BarChart3}
                    color={comparisonData.comparison_metrics.pnl_diff > 0 ? "green" : "red"}
                    trend={comparisonData.comparison_metrics.pnl_diff / 10}
                  />
                  <MetricCard
                    title="Sharpe Diff"
                    value={comparisonData.comparison_metrics.sharpe_diff.toFixed(2)}
                    subtitle="Model A vs Model B"
                    icon={Target}
                    color={comparisonData.comparison_metrics.sharpe_diff > 0 ? "green" : "red"}
                    trend={comparisonData.comparison_metrics.sharpe_diff * 10}
                  />
                  <MetricCard
                    title="Profit Factor Diff"
                    value={comparisonData.comparison_metrics.profit_factor_diff.toFixed(2)}
                    subtitle="Model A vs Model B"
                    icon={Activity}
                    color={comparisonData.comparison_metrics.profit_factor_diff > 0 ? "green" : "red"}
                    trend={comparisonData.comparison_metrics.profit_factor_diff * 10}
                  />
                </div>
              </div>

              {/* Risk Metrics */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Risk Metrics</h3>
                <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Max Drawdown Diff:</span>
                    <span className={`text-sm font-medium ${
                      comparisonData.comparison_metrics.max_drawdown_diff > 0 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {comparisonData.comparison_metrics.max_drawdown_diff.toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Avg Trade Duration Diff:</span>
                    <span className="text-sm font-medium">
                      {comparisonData.comparison_metrics.avg_trade_duration_diff.toFixed(2)} hours
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
              <ComparisonChart
                data={[
                  { model: 'Model A', winRate: 65.5, pnl: 1200, sharpe: 1.2 },
                  { model: 'Model B', winRate: 62.3, pnl: 980, sharpe: 1.1 }
                ]}
                title="Win Rate Comparison"
                dataKey="winRate"
                color="#3b82f6"
              />
              
              <ComparisonChart
                data={[
                  { model: 'Model A', winRate: 65.5, pnl: 1200, sharpe: 1.2 },
                  { model: 'Model B', winRate: 62.3, pnl: 980, sharpe: 1.1 }
                ]}
                title="Net PnL Comparison"
                dataKey="pnl"
                color="#10b981"
              />
            </div>

            {/* Radar Chart */}
            <div className="mt-6">
              <RadarChartComponent
                data={[
                  { metric: 'Win Rate', modelA: 65.5, modelB: 62.3 },
                  { metric: 'Sharpe Ratio', modelA: 1.2, modelB: 1.1 },
                  { metric: 'Profit Factor', modelA: 1.5, modelB: 1.3 },
                  { metric: 'Max Drawdown', modelA: -8.5, modelB: -12.3 },
                  { metric: 'Total Trades', modelA: 150, modelB: 120 }
                ]}
                title="Performance Radar Chart"
              />
            </div>

            {/* Action Buttons */}
            <div className="mt-6 flex justify-center space-x-4">
              <button
                onClick={() => {
                  // Deploy winner model
                  console.log('Deploying winner model:', comparisonData.winner);
                }}
                className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 flex items-center space-x-2"
              >
                <CheckCircle className="w-4 h-4" />
                <span>Deploy Winner</span>
              </button>
              <button
                onClick={() => {
                  // Export comparison report
                  console.log('Exporting comparison report');
                }}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 flex items-center space-x-2"
              >
                <Download className="w-4 h-4" />
                <span>Export Report</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelComparison; 