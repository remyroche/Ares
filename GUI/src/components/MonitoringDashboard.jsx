import React, { useState, useEffect, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  BarChart, Bar, AreaChart, Area, PieChart, Pie, Cell, RadarChart, PolarGrid, 
  PolarAngleAxis, PolarRadiusAxis, Radar, ScatterChart, Scatter, ComposedChart
} from 'recharts';
import { 
  Download, Settings, TrendingUp, TrendingDown, AlertTriangle, Activity, 
  Brain, Shield, Database, Cpu, Memory, Clock, Target, Zap, BarChart3,
  PieChart as PieChartIcon, LineChart as LineChartIcon, AreaChart as AreaChartIcon,
  Calendar, Filter, RefreshCw, Eye, EyeOff, FileText, AlertCircle, CheckCircle
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const MonitoringDashboard = () => {
  const [monitoringData, setMonitoringData] = useState({
    metrics_dashboard: {},
    performance_dashboard: {},
    ml_tracker: {},
    ml_monitor: {},
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [chartConfigs, setChartConfigs] = useState({
    performance: { type: 'line', timeRange: '24h' },
    anomalies: { type: 'bar', timeRange: '7d' },
    predictions: { type: 'area', timeRange: '24h' },
    correlations: { type: 'radar', timeRange: '7d' },
    riskMetrics: { type: 'composed', timeRange: '24h' },
    systemHealth: { type: 'line', timeRange: '1h' }
  });
  const [exportModal, setExportModal] = useState(false);
  const [exportConfig, setExportConfig] = useState({
    dataType: 'performance',
    timeRange: '7d',
    format: 'csv'
  });
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelDetails, setModelDetails] = useState(null);
  const [modelDetailsLoading, setModelDetailsLoading] = useState(false);

  // Chart type options
  const chartTypes = [
    { value: 'line', label: 'Line Chart', icon: LineChartIcon },
    { value: 'bar', label: 'Bar Chart', icon: BarChart3 },
    { value: 'area', label: 'Area Chart', icon: AreaChartIcon },
    { value: 'pie', label: 'Pie Chart', icon: PieChartIcon },
    { value: 'radar', label: 'Radar Chart', icon: Target },
    { value: 'scatter', label: 'Scatter Plot', icon: Activity },
    { value: 'composed', label: 'Composed Chart', icon: Zap }
  ];

  // Time range options
  const timeRanges = [
    { value: '1h', label: '1 Hour' },
    { value: '6h', label: '6 Hours' },
    { value: '24h', label: '24 Hours' },
    { value: '7d', label: '7 Days' },
    { value: '30d', label: '30 Days' },
    { value: '90d', label: '90 Days' }
  ];

  useEffect(() => {
    fetchMonitoringData();
    const interval = setInterval(fetchMonitoringData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchMonitoringData = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/monitoring/dashboard`);
      if (!response.ok) throw new Error('Failed to fetch monitoring data');
      const data = await response.json();
      setMonitoringData(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const updateChartConfig = (chartKey, config) => {
    setChartConfigs(prev => ({
      ...prev,
      [chartKey]: { ...prev[chartKey], ...config }
    }));
  };

  const exportData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/monitoring/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exportConfig)
      });
      
      if (!response.ok) throw new Error('Export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `monitoring_data_${exportConfig.dataType}_${exportConfig.timeRange}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      setExportModal(false);
    } catch (err) {
      setError(`Export failed: ${err.message}`);
    }
  };

  const renderChart = (data, config, title, onBarClick) => {
    const { type, timeRange } = config;
    
    switch (type) {
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Legend />
              {Object.keys(data[0] || {}).filter(key => key !== 'timestamp').map((key, index) => (
                <Line key={key} type="monotone" dataKey={key} stroke={`hsl(${index * 60}, 70%, 50%)`} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );
      
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Legend />
              {Object.keys(data[0] || {}).filter(key => key !== 'timestamp').map((key, index) => (
                <Bar key={key} dataKey={key} fill={`hsl(${index * 60}, 70%, 50%)`} onClick={onBarClick} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );
      
      case 'area':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Legend />
              {Object.keys(data[0] || {}).filter(key => key !== 'timestamp').map((key, index) => (
                <Area key={key} type="monotone" dataKey={key} fill={`hsl(${index * 60}, 70%, 30%)`} stroke={`hsl(${index * 60}, 70%, 50%)`} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        );
      
      case 'radar':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={data}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis />
              <Radar name="Value" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} onClick={onBarClick} />
            </RadarChart>
          </ResponsiveContainer>
        );
      
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                onClick={onBarClick}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={`hsl(${index * 60}, 70%, 50%)`} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        );
      
      case 'scatter':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={data}>
              <CartesianGrid />
              <XAxis dataKey="x" />
              <YAxis dataKey="y" />
              <Tooltip />
              <Scatter fill="#8884d8" onClick={onBarClick} />
            </ScatterChart>
          </ResponsiveContainer>
        );
      
      case 'composed':
        return (
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="volume" fill="#8884d8" onClick={onBarClick} />
              <Line type="monotone" dataKey="value" stroke="#82ca9d" />
            </ComposedChart>
          </ResponsiveContainer>
        );
      
      default:
        return <div>Unsupported chart type</div>;
    }
  };

  const ChartCard = ({ title, data, config, onConfigChange, onBarClick }) => (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <div className="flex items-center gap-2">
          <select
            value={config.type}
            onChange={(e) => onConfigChange({ type: e.target.value })}
            className="bg-gray-700 text-white px-2 py-1 rounded text-sm"
          >
            {chartTypes.map(type => (
              <option key={type.value} value={type.value}>{type.label}</option>
            ))}
          </select>
          <select
            value={config.timeRange}
            onChange={(e) => onConfigChange({ timeRange: e.target.value })}
            className="bg-gray-700 text-white px-2 py-1 rounded text-sm"
          >
            {timeRanges.map(range => (
              <option key={range.value} value={range.value}>{range.label}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="h-80">
        {data && data.length > 0 ? renderChart(data, config, title, onBarClick) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            No data available
          </div>
        )}
      </div>
    </div>
  );

  const MetricCard = ({ title, value, change, icon: Icon, color = "blue" }) => (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {change && (
            <div className={`flex items-center text-sm ${change > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {change > 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              {Math.abs(change)}%
            </div>
          )}
        </div>
        <Icon className={`w-8 h-8 text-${color}-400`} />
      </div>
    </div>
  );

  const AlertCard = ({ alerts, onAlertClick }) => (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle className="w-5 h-5 text-yellow-400" />
        <h3 className="text-lg font-semibold text-white">Active Alerts</h3>
      </div>
      <div className="space-y-2">
        {alerts && alerts.length > 0 ? alerts.map((alert, index) => (
          <div key={index} className="flex items-center gap-2 p-2 bg-gray-700 rounded cursor-pointer" onClick={() => onAlertClick && onAlertClick(alert)}>
            <div className={`w-2 h-2 rounded-full ${
              alert.severity === 'critical' ? 'bg-red-400' :
              alert.severity === 'high' ? 'bg-orange-400' :
              alert.severity === 'medium' ? 'bg-yellow-400' : 'bg-blue-400'
            }`} />
            <span className="text-sm text-gray-300">{alert.message}</span>
          </div>
        )) : (
          <p className="text-gray-400 text-sm">No active alerts</p>
        )}
      </div>
    </div>
  );

  // Helper functions to extract new metrics
  const getDriftAnalytics = () => {
    return monitoringData.ml_monitor?.average_performance || {};
  };
  const getFeatureImportanceStability = () => {
    // Placeholder: extract from metrics_dashboard or ml_monitor if available
    return monitoringData.metrics_dashboard?.model_behavior_metrics || {};
  };
  const getOnlineLearningMetrics = () => {
    return monitoringData.ml_monitor?.online_learning_enabled ? monitoringData.ml_monitor : {};
  };
  const getRetrainingRecommendations = () => {
    return monitoringData.ml_tracker?.retraining_recommendations || [];
  };
  const getRegimePerformance = () => {
    return monitoringData.ml_tracker?.regime_performance || {};
  };

  // Drill-down: fetch model details when selectedModel changes
  useEffect(() => {
    if (selectedModel) {
      setModelDetailsLoading(true);
      Promise.all([
        fetch(`${API_BASE_URL}/api/monitoring/feature-importance/${selectedModel}`).then(r => r.json()),
        fetch(`${API_BASE_URL}/api/monitoring/online-learning/${selectedModel}`).then(r => r.json()),
        fetch(`${API_BASE_URL}/api/monitoring/retraining-recommendations`).then(r => r.json()),
      ]).then(([featureImportance, onlineLearning, retrainingRecs]) => {
        setModelDetails({ featureImportance, onlineLearning, retrainingRecs });
        setModelDetailsLoading(false);
      });
    } else {
      setModelDetails(null);
    }
  }, [selectedModel]);

  // Drill-down: fetch drift alert details if needed (could be expanded for more info)

  // Chart click handlers
  const handleDriftChartClick = (data) => {
    if (data && data.model) setSelectedModel(data.model);
  };
  const handleFeatureImportanceChartClick = (data) => {
    if (data && data.model) setSelectedModel(data.model);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-lg p-4">
        <p className="text-red-400">Error: {error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Monitoring Dashboard</h1>
          <p className="text-gray-400">Real-time system monitoring and analytics</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setExportModal(true)}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
          >
            <Download className="w-4 h-4" />
            Export Data
          </button>
          <button
            onClick={fetchMonitoringData}
            className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Model Accuracy"
          value={`${(monitoringData.performance_dashboard?.[0]?.model_accuracy * 100 || 0).toFixed(1)}%`}
          change={2.5}
          icon={Brain}
          color="blue"
        />
        <MetricCard
          title="Win Rate"
          value={`${(monitoringData.performance_dashboard?.[0]?.trading_win_rate * 100 || 0).toFixed(1)}%`}
          change={-1.2}
          icon={Target}
          color="green"
        />
        <MetricCard
          title="System Health"
          value={`${(monitoringData.ml_monitor?.[0]?.health_score * 100 || 0).toFixed(1)}%`}
          change={0.8}
          icon={Activity}
          color="green"
        />
        <MetricCard
          title="Active Alerts"
          value={monitoringData.ml_monitor?.total_alerts || 0}
          icon={AlertTriangle}
          color="yellow"
        />
        {/* New: Drift Alerts */}
        <MetricCard
          title="Drift Alerts"
          value={monitoringData.ml_monitor?.total_alerts || 0}
          change={monitoringData.ml_monitor?.critical_alerts || 0}
          icon={AlertTriangle}
          color="red"
        />
        {/* New: Online Learning */}
        <MetricCard
          title="Online Learning"
          value={monitoringData.ml_monitor?.online_learning_enabled ? 'Enabled' : 'Disabled'}
          icon={Brain}
          color="purple"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard
          title="Performance Metrics"
          data={monitoringData.performance_dashboard}
          config={chartConfigs.performance}
          onConfigChange={(config) => updateChartConfig('performance', config)}
        />
        <ChartCard
          title="Anomaly Detection"
          data={monitoringData.ml_monitor}
          config={chartConfigs.anomalies}
          onConfigChange={(config) => updateChartConfig('anomalies', config)}
        />
        <ChartCard
          title="Predictive Analytics"
          data={monitoringData.ml_tracker}
          config={chartConfigs.predictions}
          onConfigChange={(config) => updateChartConfig('predictions', config)}
        />
        <ChartCard
          title="Correlation Analysis"
          data={monitoringData.metrics_dashboard}
          config={chartConfigs.correlations}
          onConfigChange={(config) => updateChartConfig('correlations', config)}
        />
        <ChartCard
          title="Risk Metrics"
          data={monitoringData.ml_tracker}
          config={chartConfigs.riskMetrics}
          onConfigChange={(config) => updateChartConfig('riskMetrics', config)}
        />
        <ChartCard
          title="System Health"
          data={monitoringData.ml_monitor}
          config={chartConfigs.systemHealth}
          onConfigChange={(config) => updateChartConfig('systemHealth', config)}
        />
        {/* New: Drift Analytics Chart */}
        <ChartCard
          title="Drift Analytics"
          data={Object.entries(getDriftAnalytics()).map(([model, perf]) => ({ model, ...perf }))}
          config={{ type: 'bar', timeRange: '24h' }}
          onConfigChange={() => {}}
          onBarClick={handleDriftChartClick}
        />
        {/* New: Feature Importance Stability */}
        <ChartCard
          title="Feature Importance Stability"
          data={Object.entries(getFeatureImportanceStability()).map(([model, metrics]) => ({ model, ...metrics }))}
          config={{ type: 'radar', timeRange: '24h' }}
          onConfigChange={() => {}}
          onBarClick={handleFeatureImportanceChartClick}
        />
        {/* New: Online Learning Metrics */}
        <ChartCard
          title="Online Learning Metrics"
          data={[getOnlineLearningMetrics()]}
          config={{ type: 'line', timeRange: '24h' }}
          onConfigChange={() => {}}
        />
        {/* New: Retraining Recommendations */}
        <ChartCard
          title="Retraining Recommendations"
          data={getRetrainingRecommendations().map((rec, i) => ({ recommendation: rec, idx: i }))}
          config={{ type: 'bar', timeRange: '7d' }}
          onConfigChange={() => {}}
        />
        {/* New: Regime/Ensemble Performance */}
        <ChartCard
          title="Regime Performance"
          data={Object.entries(getRegimePerformance()).map(([regime, perf]) => ({ regime, perf }))}
          config={{ type: 'bar', timeRange: '30d' }}
          onConfigChange={() => {}}
        />
      </div>

      {/* Alerts Section with drill-down */}
      <AlertCard alerts={monitoringData.ml_monitor?.current_metrics?.alerts || []} onAlertClick={setSelectedAlert} />
      {/* Alert Drill-down Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-96">
            <h3 className="text-lg font-semibold text-white mb-4">Alert Details</h3>
            <div className="space-y-2">
              <div><b>Severity:</b> {selectedAlert.severity}</div>
              <div><b>Message:</b> {selectedAlert.message}</div>
              {selectedAlert.drift_type && <div><b>Drift Type:</b> {selectedAlert.drift_type}</div>}
              {selectedAlert.features_affected && <div><b>Affected Features:</b> {selectedAlert.features_affected.join(', ')}</div>}
              <div><b>Timestamp:</b> {selectedAlert.timestamp}</div>
            </div>
            <div className="flex gap-2 mt-4">
              <button onClick={() => setSelectedAlert(null)} className="flex-1 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded">Close</button>
            </div>
          </div>
        </div>
      )}
      {/* Model Drill-down Modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-[32rem] max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-white mb-4">Model Details: {selectedModel}</h3>
            {modelDetailsLoading ? (
              <div className="text-gray-300">Loading...</div>
            ) : modelDetails ? (
              <div className="space-y-4">
                <div>
                  <b>Feature Importance History:</b>
                  <pre className="bg-gray-900 rounded p-2 text-xs text-gray-200 max-h-40 overflow-y-auto">{JSON.stringify(modelDetails.featureImportance, null, 2)}</pre>
                </div>
                <div>
                  <b>Online Learning Metrics:</b>
                  <pre className="bg-gray-900 rounded p-2 text-xs text-gray-200 max-h-40 overflow-y-auto">{JSON.stringify(modelDetails.onlineLearning, null, 2)}</pre>
                </div>
                <div>
                  <b>Retraining Recommendations:</b>
                  <ul className="list-disc ml-6 text-gray-200">
                    {modelDetails.retrainingRecs.map((rec, i) => <li key={i}>{rec}</li>)}
                  </ul>
                </div>
              </div>
            ) : <div className="text-gray-300">No details available.</div>}
            <div className="flex gap-2 mt-4">
              <button onClick={() => setSelectedModel(null)} className="flex-1 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded">Close</button>
            </div>
          </div>
        </div>
      )}
      {/* Export Modal */}
      {exportModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-96">
            <h3 className="text-lg font-semibold text-white mb-4">Export Data</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-300 mb-2">Data Type</label>
                <select
                  value={exportConfig.dataType}
                  onChange={(e) => setExportConfig(prev => ({ ...prev, dataType: e.target.value }))}
                  className="w-full bg-gray-700 text-white px-3 py-2 rounded"
                >
                  <option value="performance">Performance Metrics</option>
                  <option value="anomalies">Anomaly Detection</option>
                  <option value="predictions">Predictive Analytics</option>
                  <option value="correlations">Correlation Analysis</option>
                  <option value="riskMetrics">Risk Metrics</option>
                  <option value="systemHealth">System Health</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-2">Time Range</label>
                <select
                  value={exportConfig.timeRange}
                  onChange={(e) => setExportConfig(prev => ({ ...prev, timeRange: e.target.value }))}
                  className="w-full bg-gray-700 text-white px-3 py-2 rounded"
                >
                  {timeRanges.map(range => (
                    <option key={range.value} value={range.value}>{range.label}</option>
                  ))}
                </select>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={exportData}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
                >
                  Export CSV
                </button>
                <button
                  onClick={() => setExportModal(false)}
                  className="flex-1 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MonitoringDashboard;