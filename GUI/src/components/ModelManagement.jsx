import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { 
  Target, TrendingUp, TrendingDown, Activity, Percent, Users, 
  Database, Cpu, AlertCircle, CheckCircle, XCircle, Info, Eye, 
  EyeOff, RefreshCw, Download, Upload, Settings, Zap, Clock,
  BarChart3, PieChart as PieChartIcon, GitBranch, GitCommit, AlertTriangle
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const ModelManagement = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deployStatus, setDeployStatus] = useState({});

  const fetchModels = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/models`);
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      setModels(data.models);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const deployModel = async (modelId) => {
    setDeployStatus(prev => ({ ...prev, [modelId]: 'deploying' }));
    try {
      const response = await fetch(`${API_BASE_URL}/api/models/${modelId}/deploy`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to deploy model');
      
      setDeployStatus(prev => ({ ...prev, [modelId]: 'deployed' }));
      setTimeout(() => {
        setDeployStatus(prev => ({ ...prev, [modelId]: null }));
      }, 3000);
    } catch (err) {
      setDeployStatus(prev => ({ ...prev, [modelId]: 'error' }));
      console.error('Error deploying model:', err);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active': return <CheckCircle size={16} className="text-green-400" />;
      case 'testing': return <AlertCircle size={16} className="text-yellow-400" />;
      case 'inactive': return <XCircle size={16} className="text-red-400" />;
      default: return <Info size={16} className="text-gray-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'testing': return 'text-yellow-400';
      case 'inactive': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const renderPerformanceRadar = (performance) => {
    const data = [
      { metric: 'Accuracy', value: performance.accuracy },
      { metric: 'Precision', value: performance.precision },
      { metric: 'Recall', value: performance.recall },
      { metric: 'F1 Score', value: performance.f1_score },
    ];

    return (
      <div className="h-64 w-full">
        <ResponsiveContainer>
          <RadarChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
            <PolarGrid stroke="rgba(255, 255, 255, 0.1)" />
            <PolarAngleAxis dataKey="metric" stroke="#9ca3af" fontSize={12} />
            <PolarRadiusAxis stroke="#9ca3af" fontSize={12} domain={[0, 100]} />
            <Radar dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderModelComparison = () => {
    const comparisonData = models.map(model => ({
      name: model.name,
      accuracy: model.performance.accuracy,
      precision: model.performance.precision,
      recall: model.performance.recall,
      f1_score: model.performance.f1_score,
    }));

    return (
      <div className="h-80 w-full">
        <ResponsiveContainer>
          <BarChart data={comparisonData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
            <XAxis dataKey="name" stroke="#9ca3af" fontSize={12} />
            <YAxis stroke="#9ca3af" fontSize={12} domain={[0, 100]} />
            <Tooltip contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', borderColor: '#4b5563', borderRadius: '0.5rem' }} />
            <Legend />
            <Bar dataKey="accuracy" name="Accuracy" fill="#8b5cf6" />
            <Bar dataKey="precision" name="Precision" fill="#06b6d4" />
            <Bar dataKey="recall" name="Recall" fill="#10b981" />
            <Bar dataKey="f1_score" name="F1 Score" fill="#f59e0b" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  if (error) return <ErrorMessage message="Failed to load models." details={error} />;
  if (isLoading) return <LoadingSpinner />;

  return (
    <div className="space-y-8">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Model Management</h1>
          <p className="text-gray-400">Deploy, monitor, and manage ML models.</p>
        </div>
        <button onClick={fetchModels} className="btn-secondary flex items-center gap-2">
          <RefreshCw size={18} />
          Refresh
        </button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card title="Model Performance Comparison">
            {renderModelComparison()}
          </Card>
        </div>

        <div>
          <Card title="Model Summary">
            <div className="space-y-4">
              <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                <Target size={24} className="mx-auto mb-2 text-purple-400" />
                <p className="text-sm text-gray-400">Total Models</p>
                <p className="text-2xl font-bold text-purple-400">{models.length}</p>
              </div>
              <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                <CheckCircle size={24} className="mx-auto mb-2 text-green-400" />
                <p className="text-sm text-gray-400">Active Models</p>
                <p className="text-2xl font-bold text-green-400">
                  {models.filter(m => m.status === 'active').length}
                </p>
              </div>
              <div className="text-center p-4 bg-gray-700/50 rounded-lg">
                <AlertCircle size={24} className="mx-auto mb-2 text-yellow-400" />
                <p className="text-sm text-gray-400">Testing Models</p>
                <p className="text-2xl font-bold text-yellow-400">
                  {models.filter(m => m.status === 'testing').length}
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <Card title="Available Models">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => (
            <div 
              key={model.id}
              className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                selectedModel?.id === model.id
                  ? 'border-purple-500 bg-purple-500/10'
                  : 'border-gray-700 hover:border-gray-600'
              }`}
              onClick={() => setSelectedModel(model)}
            >
              <div className="flex justify-between items-start mb-3">
                <div>
                  <h3 className="font-semibold text-white">{model.name}</h3>
                  <p className="text-sm text-gray-400">v{model.version}</p>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusIcon(model.status)}
                  <span className={`text-xs font-medium ${getStatusColor(model.status)}`}>
                    {model.status}
                  </span>
                </div>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Type:</span>
                  <span className="text-white capitalize">{model.type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Accuracy:</span>
                  <span className="text-green-400">{model.performance.accuracy}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">F1 Score:</span>
                  <span className="text-blue-400">{model.performance.f1_score}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Updated:</span>
                  <span className="text-white text-xs">
                    {new Date(model.last_updated).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <div className="mt-4 flex gap-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deployModel(model.id);
                  }}
                  disabled={deployStatus[model.id] === 'deploying'}
                  className={`flex-1 text-xs py-2 px-3 rounded transition-colors ${
                    deployStatus[model.id] === 'deploying'
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                      : 'bg-purple-600 hover:bg-purple-700 text-white'
                  }`}
                >
                  {deployStatus[model.id] === 'deploying' ? (
                    <>
                      <RefreshCw size={12} className="animate-spin mr-1" />
                      Deploying...
                    </>
                  ) : (
                    <>
                      <Upload size={12} className="mr-1" />
                      Deploy
                    </>
                  )}
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedModel(model);
                  }}
                  className="flex-1 text-xs py-2 px-3 rounded bg-gray-600 hover:bg-gray-700 text-white transition-colors"
                >
                  <Eye size={12} className="mr-1" />
                  Details
                </button>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {selectedModel && (
        <Modal title={`Model Details - ${selectedModel.name}`} onClose={() => setSelectedModel(null)}>
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-white mb-3">Model Information</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Name:</span>
                    <span className="text-white">{selectedModel.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Version:</span>
                    <span className="text-white">{selectedModel.version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Type:</span>
                    <span className="text-white capitalize">{selectedModel.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Status:</span>
                    <span className={`capitalize ${getStatusColor(selectedModel.status)}`}>
                      {selectedModel.status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Last Updated:</span>
                    <span className="text-white">
                      {new Date(selectedModel.last_updated).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-semibold text-white mb-3">Performance Metrics</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Accuracy:</span>
                    <span className="text-green-400">{selectedModel.performance.accuracy}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Precision:</span>
                    <span className="text-blue-400">{selectedModel.performance.precision}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Recall:</span>
                    <span className="text-yellow-400">{selectedModel.performance.recall}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">F1 Score:</span>
                    <span className="text-purple-400">{selectedModel.performance.f1_score}%</span>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold text-white mb-3">Performance Radar</h4>
              {renderPerformanceRadar(selectedModel.performance)}
            </div>

            <div className="flex gap-4">
              <button 
                onClick={() => deployModel(selectedModel.id)}
                disabled={deployStatus[selectedModel.id] === 'deploying'}
                className="btn-primary flex items-center gap-2 flex-1"
              >
                {deployStatus[selectedModel.id] === 'deploying' ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" />
                    Deploying...
                  </>
                ) : (
                  <>
                    <Upload size={18} />
                    Deploy Model
                  </>
                )}
              </button>
              <button className="btn-secondary flex items-center gap-2 flex-1">
                <Download size={18} />
                Export Model
              </button>
            </div>
          </div>
        </Modal>
      )}
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

export default ModelManagement; 