import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { 
  GitCompare, TrendingUp, TrendingDown, Activity, Percent, Users, 
  Database, Cpu, AlertCircle, CheckCircle, XCircle, Info, Eye, 
  EyeOff, RefreshCw, Download, Target, Clock, DollarSign, Trophy,
  AlertTriangle, BarChart3, PieChart as PieChartIcon
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const ABTesting = () => {
  const [testResults, setTestResults] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedWinner, setSelectedWinner] = useState(null);
  const [testHistory, setTestHistory] = useState([]);

  useEffect(() => {
    // Mock data for demonstration
    const mockResults = {
      test_id: "ab_test_001",
      start_date: "2024-01-01",
      end_date: "2024-01-31",
      status: "completed",
      models: {
        model_a: {
          name: "Performer v1.2",
          total_return: 15.2,
          sharpe_ratio: 2.1,
          max_drawdown: 5.5,
          win_rate: 68.5,
          total_trades: 145,
          avg_trade_duration: 4.2,
          profit_factor: 1.85,
          calmar_ratio: 2.76,
          color: "#8b5cf6"
        },
        model_b: {
          name: "Current v3.1",
          total_return: 12.8,
          sharpe_ratio: 1.9,
          max_drawdown: 6.8,
          win_rate: 65.2,
          total_trades: 138,
          avg_trade_duration: 3.8,
          profit_factor: 1.72,
          calmar_ratio: 1.88,
          color: "#06b6d4"
        }
      },
      winner: "model_a",
      confidence_level: 0.95,
      statistical_significance: true
    };

    setTestResults(mockResults);
    setSelectedWinner(mockResults.winner);
    setIsLoading(false);
  }, []);

  const handleSelectWinner = (modelKey) => {
    setSelectedWinner(modelKey);
    // In a real implementation, this would call an API to update the winner
    console.log(`Selected winner: ${modelKey}`);
  };

  const renderComparisonChart = () => {
    if (!testResults) return null;

    const data = [
      { metric: 'Total Return (%)', model_a: testResults.models.model_a.total_return, model_b: testResults.models.model_b.total_return },
      { metric: 'Sharpe Ratio', model_a: testResults.models.model_a.sharpe_ratio, model_b: testResults.models.model_b.sharpe_ratio },
      { metric: 'Max Drawdown (%)', model_a: testResults.models.model_a.max_drawdown, model_b: testResults.models.model_b.max_drawdown },
      { metric: 'Win Rate (%)', model_a: testResults.models.model_a.win_rate, model_b: testResults.models.model_b.win_rate },
      { metric: 'Profit Factor', model_a: testResults.models.model_a.profit_factor, model_b: testResults.models.model_b.profit_factor },
      { metric: 'Calmar Ratio', model_a: testResults.models.model_a.calmar_ratio, model_b: testResults.models.model_b.calmar_ratio },
    ];

    return (
      <div className="h-80 w-full">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
            <XAxis type="number" stroke="#9ca3af" />
            <YAxis dataKey="metric" type="category" stroke="#9ca3af" width={120} />
            <Tooltip contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', borderColor: '#4b5563', borderRadius: '0.5rem' }} />
            <Legend />
            <Bar dataKey="model_a" name={testResults.models.model_a.name} fill={testResults.models.model_a.color} />
            <Bar dataKey="model_b" name={testResults.models.model_b.name} fill={testResults.models.model_b.color} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderPerformanceRadar = (modelData) => {
    const data = [
      { metric: 'Total Return', value: modelData.total_return },
      { metric: 'Sharpe Ratio', value: modelData.sharpe_ratio * 10 }, // Scale for radar chart
      { metric: 'Win Rate', value: modelData.win_rate },
      { metric: 'Profit Factor', value: modelData.profit_factor * 20 }, // Scale for radar chart
      { metric: 'Calmar Ratio', value: modelData.calmar_ratio * 10 }, // Scale for radar chart
    ];

    return (
      <div className="h-64 w-full">
        <ResponsiveContainer>
          <RadarChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
            <PolarGrid stroke="rgba(255, 255, 255, 0.1)" />
            <PolarAngleAxis dataKey="metric" stroke="#9ca3af" fontSize={12} />
            <PolarRadiusAxis stroke="#9ca3af" fontSize={12} domain={[0, 100]} />
            <Radar dataKey="value" stroke={modelData.color} fill={modelData.color} fillOpacity={0.3} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  if (error) return <ErrorMessage message="Failed to load A/B test results." details={error} />;
  if (isLoading) return <LoadingSpinner />;

  return (
    <div className="space-y-8">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">A/B Testing</h1>
          <p className="text-gray-400">Compare model performance and select the winner.</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`px-3 py-1 rounded-full text-xs font-medium ${
            testResults?.status === 'completed' 
              ? 'bg-green-500/20 text-green-300 border border-green-500/30'
              : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
          }`}>
            {testResults?.status === 'completed' ? 'Completed' : 'Running'}
          </div>
          {testResults?.statistical_significance && (
            <div className="px-3 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-300 border border-blue-500/30">
              Statistically Significant
            </div>
          )}
        </div>
      </header>

      <Card title="Test Overview">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-gray-700/50 rounded-lg">
            <Trophy size={24} className="mx-auto mb-2 text-yellow-400" />
            <p className="text-sm text-gray-400">Winner</p>
            <p className="text-lg font-bold text-yellow-400">
              {testResults?.models[testResults?.winner]?.name || 'N/A'}
            </p>
          </div>
          <div className="text-center p-4 bg-gray-700/50 rounded-lg">
            <Percent size={24} className="mx-auto mb-2 text-blue-400" />
            <p className="text-sm text-gray-400">Confidence</p>
            <p className="text-lg font-bold text-blue-400">
              {(testResults?.confidence_level * 100).toFixed(1)}%
            </p>
          </div>
          <div className="text-center p-4 bg-gray-700/50 rounded-lg">
            <Calendar size={24} className="mx-auto mb-2 text-green-400" />
            <p className="text-sm text-gray-400">Duration</p>
            <p className="text-lg font-bold text-green-400">30 days</p>
          </div>
          <div className="text-center p-4 bg-gray-700/50 rounded-lg">
            <BarChart3 size={24} className="mx-auto mb-2 text-purple-400" />
            <p className="text-sm text-gray-400">Total Tests</p>
            <p className="text-lg font-bold text-purple-400">12</p>
          </div>
        </div>
      </Card>

      <Card title="Performance Comparison">
        {renderComparisonChart()}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title={`${testResults?.models.model_a.name} Performance`}>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Total Return:</span>
                <span className="text-green-400">{testResults?.models.model_a.total_return}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Sharpe Ratio:</span>
                <span className="text-blue-400">{testResults?.models.model_a.sharpe_ratio}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Max Drawdown:</span>
                <span className="text-red-400">{testResults?.models.model_a.max_drawdown}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Win Rate:</span>
                <span className="text-yellow-400">{testResults?.models.model_a.win_rate}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Total Trades:</span>
                <span className="text-white">{testResults?.models.model_a.total_trades}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Avg Duration:</span>
                <span className="text-white">{testResults?.models.model_a.avg_trade_duration}h</span>
              </div>
            </div>
            {renderPerformanceRadar(testResults?.models.model_a)}
          </div>
        </Card>

        <Card title={`${testResults?.models.model_b.name} Performance`}>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Total Return:</span>
                <span className="text-green-400">{testResults?.models.model_b.total_return}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Sharpe Ratio:</span>
                <span className="text-blue-400">{testResults?.models.model_b.sharpe_ratio}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Max Drawdown:</span>
                <span className="text-red-400">{testResults?.models.model_b.max_drawdown}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Win Rate:</span>
                <span className="text-yellow-400">{testResults?.models.model_b.win_rate}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Total Trades:</span>
                <span className="text-white">{testResults?.models.model_b.total_trades}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Avg Duration:</span>
                <span className="text-white">{testResults?.models.model_b.avg_trade_duration}h</span>
              </div>
            </div>
            {renderPerformanceRadar(testResults?.models.model_b)}
          </div>
        </Card>
      </div>

      <Card title="Winner Selection">
        <div className="space-y-6">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-white mb-2">Select the Winning Model</h3>
            <p className="text-gray-400 mb-4">
              The winning model will be deployed to production. Choose carefully based on the performance metrics above.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div 
              className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                selectedWinner === 'model_a'
                  ? 'border-purple-500 bg-purple-500/10'
                  : 'border-gray-700 hover:border-gray-600'
              }`}
              onClick={() => handleSelectWinner('model_a')}
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-white">{testResults?.models.model_a.name}</h4>
                {selectedWinner === 'model_a' && <CheckCircle size={20} className="text-green-400" />}
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Return:</span>
                  <span className="text-green-400">{testResults?.models.model_a.total_return}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Sharpe Ratio:</span>
                  <span className="text-blue-400">{testResults?.models.model_a.sharpe_ratio}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Win Rate:</span>
                  <span className="text-yellow-400">{testResults?.models.model_a.win_rate}%</span>
                </div>
              </div>
            </div>

            <div 
              className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                selectedWinner === 'model_b'
                  ? 'border-purple-500 bg-purple-500/10'
                  : 'border-gray-700 hover:border-gray-600'
              }`}
              onClick={() => handleSelectWinner('model_b')}
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-white">{testResults?.models.model_b.name}</h4>
                {selectedWinner === 'model_b' && <CheckCircle size={20} className="text-green-400" />}
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Return:</span>
                  <span className="text-green-400">{testResults?.models.model_b.total_return}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Sharpe Ratio:</span>
                  <span className="text-blue-400">{testResults?.models.model_b.sharpe_ratio}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Win Rate:</span>
                  <span className="text-yellow-400">{testResults?.models.model_b.win_rate}%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-center gap-4">
            <button 
              onClick={() => handleSelectWinner(selectedWinner)}
              disabled={!selectedWinner}
              className="btn-primary flex items-center gap-2"
            >
              <Trophy size={18} />
              Deploy Winner
            </button>
            <button className="btn-secondary flex items-center gap-2">
              <Download size={18} />
              Export Results
            </button>
          </div>
        </div>
      </Card>
    </div>
  );
};

// Helper Components
const Card = ({ title, children }) => (
  <div className="bg-gray-800/50 p-4 sm:p-6 rounded-xl border border-gray-700/50 shadow-lg">
    {title && <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>}
    {children}
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

export default ABTesting; 