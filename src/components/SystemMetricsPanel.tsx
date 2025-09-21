import React, { useState, useEffect } from 'react';
import { Cpu, HardDrive, Wifi, Users, Shield, TrendingUp } from 'lucide-react';
import { SystemMetrics } from '../types';
import { generateSystemMetrics } from '../utils/mockData';

interface SystemMetricsPanelProps {
  isDarkMode: boolean;
}

export const SystemMetricsPanel: React.FC<SystemMetricsPanelProps> = ({ isDarkMode }) => {
  const [metrics, setMetrics] = useState<SystemMetrics>(generateSystemMetrics());
  const [history, setHistory] = useState<SystemMetrics[]>([]);

  useEffect(() => {
    const interval = setInterval(() => {
      const newMetrics = generateSystemMetrics();
      setMetrics(newMetrics);
      setHistory(prev => [...prev.slice(-19), newMetrics]);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getMetricColor = (value: number, threshold: number) => {
    if (value > threshold * 0.8) return isDarkMode ? 'text-red-400' : 'text-red-600';
    if (value > threshold * 0.6) return isDarkMode ? 'text-yellow-400' : 'text-yellow-600';
    return isDarkMode ? 'text-green-400' : 'text-green-600';
  };

  const renderMiniChart = (data: number[], color: string) => {
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;

    return (
      <svg className="w-16 h-8" viewBox="0 0 64 32">
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="2"
          points={data.map((value, index) => 
            `${(index / (data.length - 1)) * 64},${32 - ((value - min) / range) * 32}`
          ).join(' ')}
        />
      </svg>
    );
  };

  return (
    <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-2xl transition-all duration-300 border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
          <TrendingUp className={`h-6 w-6 ${isDarkMode ? 'text-cyan-400' : 'text-blue-600'}`} />
          <span>System Metrics</span>
        </h2>
        <div className={`px-3 py-1 rounded-full text-xs font-semibold ${isDarkMode ? 'bg-blue-500/20 text-blue-400' : 'bg-blue-100 text-blue-600'} animate-pulse`}>
          REAL-TIME
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {/* CPU Usage */}
        <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} border ${isDarkMode ? 'border-gray-600' : 'border-gray-200'} transition-all duration-300 hover:shadow-lg`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Cpu className={`h-5 w-5 ${getMetricColor(metrics.cpuUsage, 100)}`} />
              <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>CPU</span>
            </div>
            <div className="flex items-center space-x-2">
              {history.length > 0 && renderMiniChart(
                history.map(h => h.cpuUsage), 
                getMetricColor(metrics.cpuUsage, 100).includes('red') ? '#ef4444' : 
                getMetricColor(metrics.cpuUsage, 100).includes('yellow') ? '#f59e0b' : '#10b981'
              )}
              <span className={`text-lg font-bold ${getMetricColor(metrics.cpuUsage, 100)}`}>
                {metrics.cpuUsage}%
              </span>
            </div>
          </div>
          <div className={`w-full h-2 rounded-full ${isDarkMode ? 'bg-gray-600' : 'bg-gray-300'} overflow-hidden`}>
            <div
              className={`h-full transition-all duration-1000 ease-out ${
                metrics.cpuUsage > 80 ? 'bg-red-500' : 
                metrics.cpuUsage > 60 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
              style={{ width: `${metrics.cpuUsage}%` }}
            />
          </div>
        </div>

        {/* Memory Usage */}
        <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} border ${isDarkMode ? 'border-gray-600' : 'border-gray-200'} transition-all duration-300 hover:shadow-lg`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <HardDrive className={`h-5 w-5 ${getMetricColor(metrics.memoryUsage, 100)}`} />
              <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Memory</span>
            </div>
            <div className="flex items-center space-x-2">
              {history.length > 0 && renderMiniChart(
                history.map(h => h.memoryUsage), 
                getMetricColor(metrics.memoryUsage, 100).includes('red') ? '#ef4444' : 
                getMetricColor(metrics.memoryUsage, 100).includes('yellow') ? '#f59e0b' : '#10b981'
              )}
              <span className={`text-lg font-bold ${getMetricColor(metrics.memoryUsage, 100)}`}>
                {metrics.memoryUsage}%
              </span>
            </div>
          </div>
          <div className={`w-full h-2 rounded-full ${isDarkMode ? 'bg-gray-600' : 'bg-gray-300'} overflow-hidden`}>
            <div
              className={`h-full transition-all duration-1000 ease-out ${
                metrics.memoryUsage > 80 ? 'bg-red-500' : 
                metrics.memoryUsage > 60 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
              style={{ width: `${metrics.memoryUsage}%` }}
            />
          </div>
        </div>

        {/* Network Throughput */}
        <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} border ${isDarkMode ? 'border-gray-600' : 'border-gray-200'} transition-all duration-300 hover:shadow-lg`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Wifi className={`h-5 w-5 ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`} />
              <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Network</span>
            </div>
            <span className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {metrics.networkThroughput} Mbps
            </span>
          </div>
        </div>

        {/* Active Connections */}
        <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} border ${isDarkMode ? 'border-gray-600' : 'border-gray-200'} transition-all duration-300 hover:shadow-lg`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Users className={`h-5 w-5 ${isDarkMode ? 'text-green-400' : 'text-green-600'}`} />
              <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Connections</span>
            </div>
            <span className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {metrics.activeConnections}
            </span>
          </div>
        </div>

        {/* Blocked Attacks */}
        <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} border ${isDarkMode ? 'border-gray-600' : 'border-gray-200'} transition-all duration-300 hover:shadow-lg`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Shield className={`h-5 w-5 ${isDarkMode ? 'text-red-400' : 'text-red-600'} animate-pulse`} />
              <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Blocked</span>
            </div>
            <span className={`text-lg font-bold ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
              {metrics.blockedAttacks}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};