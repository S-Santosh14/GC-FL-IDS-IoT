import React, { useState, useEffect } from 'react';
import { AlertTriangle, Info, Shield } from 'lucide-react';
import { Alert } from '../types';
import { generateMockAlert } from '../utils/mockData';

interface LiveAlertsPanelProps {
  isDarkMode: boolean;
}

export const LiveAlertsPanel: React.FC<LiveAlertsPanelProps> = ({ isDarkMode }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  useEffect(() => {
    // Initialize with some alerts
    const initialAlerts = Array.from({ length: 5 }, () => generateMockAlert());
    setAlerts(initialAlerts);

    // Add new alert every 3 seconds
    const interval = setInterval(() => {
      const newAlert = generateMockAlert();
      setAlerts(prev => [newAlert, ...prev.slice(0, 9)]); // Keep only last 10 alerts
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getRiskColor = (risk: Alert['riskFactor']) => {
    switch (risk) {
      case 'Normal': return isDarkMode ? 'text-green-400' : 'text-green-600';
      case 'Suspicious': return isDarkMode ? 'text-yellow-400' : 'text-yellow-600';
      case 'Intrusion': return isDarkMode ? 'text-red-400' : 'text-red-600';
    }
  };

  const getRiskBg = (risk: Alert['riskFactor']) => {
    switch (risk) {
      case 'Normal': return 'bg-green-500/20 border-green-500/30';
      case 'Suspicious': return 'bg-yellow-500/20 border-yellow-500/30';
      case 'Intrusion': return 'bg-red-500/20 border-red-500/30';
    }
  };

  const getRiskIcon = (risk: Alert['riskFactor']) => {
    switch (risk) {
      case 'Normal': return <Shield className="h-4 w-4" />;
      case 'Suspicious': return <Info className="h-4 w-4" />;
      case 'Intrusion': return <AlertTriangle className="h-4 w-4" />;
    }
  };

  return (
    <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-2xl transition-all duration-300 border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
          <AlertTriangle className={`h-6 w-6 ${isDarkMode ? 'text-cyan-400' : 'text-blue-600'}`} />
          <span>Live Security Alerts</span>
        </h2>
        <div className={`px-3 py-1 rounded-full text-xs font-semibold ${isDarkMode ? 'bg-green-500/20 text-green-400' : 'bg-green-100 text-green-600'} animate-pulse`}>
          LIVE
        </div>
      </div>

      <div className="overflow-hidden rounded-lg">
        <div className={`${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} px-4 py-3 border-b ${isDarkMode ? 'border-gray-600' : 'border-gray-200'}`}>
          <div className="grid grid-cols-6 gap-4 text-sm font-semibold">
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Time</span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Device</span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Attack Type</span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Risk</span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Confidence</span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Action</span>
          </div>
        </div>

        <div className="max-h-96 overflow-y-auto">
          {alerts.map((alert, index) => (
            <div 
              key={alert.id} 
              className={`px-4 py-3 border-b transition-all duration-300 hover:bg-opacity-50 relative ${
                isDarkMode ? 'border-gray-600 hover:bg-gray-700' : 'border-gray-200 hover:bg-gray-50'
              } ${alert.riskFactor === 'Intrusion' && index < 2 ? `${getRiskBg(alert.riskFactor)} animate-pulse border-2` : ''}
              ${alert.riskFactor === 'Intrusion' && index === 0 ? 'shadow-lg shadow-red-500/25' : ''}`}
            >
              {/* Glowing border for critical alerts */}
              {alert.riskFactor === 'Intrusion' && index < 2 && (
                <div className="absolute inset-0 rounded border-2 border-red-500 animate-pulse opacity-50"></div>
              )}
              
              <div className="grid grid-cols-6 gap-4 items-center text-sm">
                <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>{alert.time}</span>
                <span className={`${isDarkMode ? 'text-white' : 'text-gray-900'} font-medium`}>{alert.device}</span>
                <span className={isDarkMode ? 'text-white' : 'text-gray-900'}>{alert.attackType}</span>
                <span className={`flex items-center space-x-1 font-semibold ${getRiskColor(alert.riskFactor)}`}>
                  {getRiskIcon(alert.riskFactor)}
                  <span>{alert.riskFactor}</span>
                </span>
                <span className={isDarkMode ? 'text-white' : 'text-gray-900'}>{alert.confidence}%</span>
                <button
                  onClick={() => setSelectedAlert(alert)}
                  className={`px-3 py-1 rounded-md text-xs font-medium transition-all duration-200 transform hover:scale-105 ${
                    isDarkMode 
                      ? 'bg-cyan-600 hover:bg-cyan-500 text-white hover:shadow-cyan-500/25' 
                      : 'bg-blue-600 hover:bg-blue-500 text-white hover:shadow-blue-500/25'
                  } shadow-md hover:shadow-lg`}
                >
                  Explain
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Alert Explanation Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 max-w-md mx-4 shadow-2xl border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
                {getRiskIcon(selectedAlert.riskFactor)}
                <span>Alert Explanation</span>
              </h3>
              <button
                onClick={() => setSelectedAlert(null)}
                className={`${isDarkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'} transition-colors`}
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-3">
              <div>
                <span className={`text-sm font-semibold ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Device: </span>
                <span className={isDarkMode ? 'text-white' : 'text-gray-900'}>{selectedAlert.device}</span>
              </div>
              
              <div>
                <span className={`text-sm font-semibold ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Risk Level: </span>
                <span className={`font-semibold ${getRiskColor(selectedAlert.riskFactor)}`}>{selectedAlert.riskFactor}</span>
              </div>
              
              <div>
                <span className={`text-sm font-semibold ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Confidence: </span>
                <span className={isDarkMode ? 'text-white' : 'text-gray-900'}>{selectedAlert.confidence}%</span>
              </div>
              
              <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} border-l-4 ${
                selectedAlert.riskFactor === 'Intrusion' ? 'border-red-500' : 
                selectedAlert.riskFactor === 'Suspicious' ? 'border-yellow-500' : 'border-green-500'
              }`}>
                <p className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  {selectedAlert.explanation}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};