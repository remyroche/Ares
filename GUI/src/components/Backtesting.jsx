import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, AreaChart, Area, PieChart, Pie, Cell } from 'recharts';
import { Play, Settings, Download, RefreshCw, TrendingUp, TrendingDown, BarChart3, PieChart as PieChartIcon, Target, Clock, DollarSign, Percent } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const Backtesting = () => {
  const [activeTab, setActiveTab] = useState('new');
  const [results, setResults] = useState(null);
  const [comparisons, setComparisons] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedComparison, setSelectedComparison] = useState(null);

  const fetchComparisons = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/backtest/comparison`);
      if (!response.ok) throw new Error('Failed to fetch comparisons');
      const data = await response.json();
      setComparisons(data.comparisons);
    } catch (err) {
      console.error('Error fetching comparisons:', err);
    }
  };

  useEffect(() => {
    fetchComparisons();
  }, []);

  const handleRunTest = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    const formData = new FormData(e.target);
    const params = Object.fromEntries(formData.entries());

    try {
      const response = await fetch(`${API_BASE_URL}/api/run-backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (!response.ok) throw new Error('Backtest failed to run');
      const data = await response.json();
      setResults(data.results);
      setActiveTab('results');
      fetchComparisons(); // Refresh comparisons
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleComparisonSelect = (comparison) => {
    setSelectedComparison(comparison);
  };

  const exportResults = () => {
    if (!results) return;
    
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `backtest-results-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const renderMetrics = (metrics) => (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard 
        title="Total Return" 
        value={metrics.totalReturn} 
        icon={<TrendingUp className="text-green-400" />}
        color="text-green-400"
      />
      <MetricCard 
        title="Sharpe Ratio" 
        value={metrics.sharpeRatio} 
        icon={<BarChart3 className="text-blue-400" />}
        color="text-blue-400"
      />
      <MetricCard 
        title="Max Drawdown" 
        value={metrics.maxDrawdown} 
        icon={<TrendingDown className="text-red-400" />}
        color="text-red-400"
      />
      <MetricCard 
        title="Win Rate" 
        value={metrics.winRate} 
        icon={<Percent className="text-yellow-400" />}
        color="text-yellow-400"
      />
    </div>
  );

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold text-white">Backtesting & Optimization</h1>
        <p className="text-gray-400">Test strategies, optimize parameters, and analyze results.</p>
      </header>

      <div className="flex border-b border-gray-700">
        <button 
          onClick={() => setActiveTab('new')} 
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'new' 
              ? 'border-b-2 border-purple-500 text-white' 
              : 'text-gray-400 hover:text-white'
          }`}
        >
          New Test
        </button>
        <button 
          onClick={() => setActiveTab('results')} 
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'results' 
              ? 'border-b-2 border-purple-500 text-white' 
              : 'text-gray-400 hover:text-white'
          }`}
          disabled={!results}
        >
          Results
        </button>
        <button 
          onClick={() => setActiveTab('comparison')} 
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'comparison' 
              ? 'border-b-2 border-purple-500 text-white' 
              : 'text-gray-400 hover:text-white'
          }`}
        >
          Comparison
        </button>
      </div>
      
      {error && <ErrorMessage message="An error occurred during the backtest." details={error} />}

      {activeTab === 'new' && (
        <Card>
          <form onSubmit={handleRunTest} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <InputGroup label="Token Pair" name="token_pair" placeholder="e.g., BTC/USDT" required />
              <InputGroup label="Exchange" name="exchange" placeholder="e.g., Binance" required />
            </div>
            
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">Test Type</label>
              <div className="flex gap-4">
                <RadioInput name="test_type" value="blank_run" label="Blank Run" defaultChecked />
                <RadioInput name="test_type" value="param_test" label="Parameter Testing" />
                <RadioInput name="test_type" value="optimization" label="Optimization" />
              </div>
            </div>

            <CollapsibleSection title="Advanced Parameters">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
                <InputGroup label="Start Date" name="start_date" type="date" />
                <InputGroup label="End Date" name="end_date" type="date" />
                <InputGroup label="Initial Capital" name="capital" type="number" placeholder="10000" />
                <InputGroup label="Commission (%)" name="commission" type="number" placeholder="0.1" step="0.01" />
                <InputGroup label="Model Version" name="model_version" placeholder="v1.2" />
              </div>
            </CollapsibleSection>

            <div className="flex justify-end gap-4">
              <button 
                type="button" 
                onClick={() => setActiveTab('comparison')}
                className="btn-secondary"
              >
                View Comparisons
              </button>
              <button 
                type="submit" 
                className="btn-primary flex items-center gap-2" 
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    Run Test
                  </>
                )}
              </button>
            </div>
          </form>
        </Card>
      )}

      {activeTab === 'results' && results && (
        <div className="space-y-8">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold text-white">Backtest Results</h2>
            <button onClick={exportResults} className="btn-secondary flex items-center gap-2">
              <Download size={18} />
              Export Results
            </button>
          </div>

          <Card title="Performance Summary">
            {renderMetrics(results.summary)}
          </Card>

          <Card title="Equity Curve">
            <div className="h-96 w-full">
              <ResponsiveContainer>
                <AreaChart data={results.equityCurve} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                  <defs>
                    <linearGradient id="colorUv" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                  <XAxis dataKey="date" stroke="#9ca3af" fontSize={12} />
                  <YAxis stroke="#9ca3af" fontSize={12} domain={['dataMin', 'dataMax']}/>
                  <Tooltip contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', borderColor: '#4b5563', borderRadius: '0.5rem' }} />
                  <Area type="monotone" dataKey="portfolioValue" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorUv)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {results.tradeAnalysis && (
            <Card title="Trade Analysis">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                  <DollarSign size={24} className="mx-auto mb-2 text-green-400" />
                  <p className="text-sm text-gray-400">Best Trade</p>
                  <p className="text-lg font-bold text-green-400">${results.tradeAnalysis.bestTrade?.toFixed(2)}</p>
                </div>
                <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                  <DollarSign size={24} className="mx-auto mb-2 text-red-400" />
                  <p className="text-sm text-gray-400">Worst Trade</p>
                  <p className="text-lg font-bold text-red-400">${results.tradeAnalysis.worstTrade?.toFixed(2)}</p>
                </div>
                <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                  <Clock size={24} className="mx-auto mb-2 text-blue-400" />
                  <p className="text-sm text-gray-400">Avg Win</p>
                  <p className="text-lg font-bold text-blue-400">${results.tradeAnalysis.avgWin?.toFixed(2)}</p>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {activeTab === 'comparison' && (
        <div className="space-y-8">
          <h2 className="text-2xl font-bold text-white">Strategy Comparison</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card title="Available Strategies">
              <div className="space-y-4">
                {comparisons.map((comparison) => (
                  <div 
                    key={comparison.id}
                    onClick={() => handleComparisonSelect(comparison)}
                    className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                      selectedComparison?.id === comparison.id
                        ? 'border-purple-500 bg-purple-500/10'
                        : 'border-gray-700 hover:border-gray-600'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <h3 className="font-semibold text-white">{comparison.name}</h3>
                        <p className="text-sm text-gray-400">Total Return: {comparison.totalReturn}%</p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-gray-400">Sharpe: {comparison.sharpeRatio}</p>
                        <p className="text-sm text-gray-400">DD: {comparison.maxDrawdown}%</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {selectedComparison && (
              <Card title="Strategy Details">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-gray-700/50 rounded-lg">
                      <Target size={20} className="mx-auto mb-2 text-green-400" />
                      <p className="text-sm text-gray-400">Total Return</p>
                      <p className="text-lg font-bold text-green-400">{selectedComparison.totalReturn}%</p>
                    </div>
                    <div className="text-center p-3 bg-gray-700/50 rounded-lg">
                      <BarChart3 size={20} className="mx-auto mb-2 text-blue-400" />
                      <p className="text-sm text-gray-400">Sharpe Ratio</p>
                      <p className="text-lg font-bold text-blue-400">{selectedComparison.sharpeRatio}</p>
                    </div>
                    <div className="text-center p-3 bg-gray-700/50 rounded-lg">
                      <TrendingDown size={20} className="mx-auto mb-2 text-red-400" />
                      <p className="text-sm text-gray-400">Max Drawdown</p>
                      <p className="text-lg font-bold text-red-400">{selectedComparison.maxDrawdown}%</p>
                    </div>
                    <div className="text-center p-3 bg-gray-700/50 rounded-lg">
                      <Percent size={20} className="mx-auto mb-2 text-yellow-400" />
                      <p className="text-sm text-gray-400">Win Rate</p>
                      <p className="text-lg font-bold text-yellow-400">{selectedComparison.winRate}%</p>
                    </div>
                  </div>
                  
                  <div className="flex gap-4">
                    <button className="btn-primary flex-1">Deploy Strategy</button>
                    <button className="btn-secondary flex-1">View Details</button>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Helper Components
const MetricCard = ({ title, value, icon, color }) => (
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

const ErrorMessage = ({ message, details }) => (
  <div className="bg-red-500/10 border border-red-500/30 text-red-300 p-4 rounded-lg">
    <div className="flex items-center gap-2">
      <AlertTriangle size={20} />
      <h4 className="font-bold">{message}</h4>
    </div>
    {details && <p className="text-sm mt-2 font-mono bg-red-900/20 p-2 rounded">{details}</p>}
  </div>
);

export default Backtesting; 