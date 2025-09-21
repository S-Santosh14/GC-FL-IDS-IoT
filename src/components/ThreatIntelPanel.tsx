import React, { useState, useEffect } from 'react';
import { AlertTriangle, Eye, Clock, ExternalLink } from 'lucide-react';
import { ThreatIntelligence } from '../types';
import { generateThreatIntelligence } from '../utils/mockData';

interface ThreatIntelPanelProps {
  isDarkMode: boolean;
}

export const ThreatIntelPanel: React.FC<ThreatIntelPanelProps> = ({ isDarkMode }) => {
  const [threats, setThreats] = useState<ThreatIntelligence[]>([]);

  useEffect(() => {
    setThreats(generateThreatIntelligence());
    
    const interval = setInterval(() => {
      // Occasionally add new threat intelligence
      if (Math.random() > 0.7) {
        const newThreat: ThreatIntelligence = {
          id: Math.random().toString(36).substr(2, 9),
          source: ['MITRE ATT&CK', 'Threat Intel Feed', 'CVE Database', 'OSINT'][Math.floor(Math.random() * 4)],
          threatType: ['APT Campaign', 'Botnet Activity', 'Zero-day Exploit', 'Malware Variant'][Math.floor(Math.random() * 4)],
          severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any,
          description: 'New threat detected in global intelligence feeds',
          timestamp: new Date().toISOString()
        };
        setThreats(prev => [newThreat, ...prev.slice(0, 4)]);
      }
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const getSeverityColor = (severity: ThreatIntelligence['severity']) => {
    switch (severity) {
      case 'low': return isDarkMode ? 'text-blue-400 bg-blue-500/20 border-blue-500/30' : 'text-blue-600 bg-blue-100 border-blue-200';
      case 'medium': return isDarkMode ? 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30' : 'text-yellow-600 bg-yellow-100 border-yellow-200';
      case 'high': return isDarkMode ? 'text-orange-400 bg-orange-500/20 border-orange-500/30' : 'text-orange-600 bg-orange-100 border-orange-200';
      case 'critical': return isDarkMode ? 'text-red-400 bg-red-500/20 border-red-500/30' : 'text-red-600 bg-red-100 border-red-200';
    }
  };

  const getSeverityIcon = (severity: ThreatIntelligence['severity']) => {
    switch (severity) {
      case 'low': return '🔵';
      case 'medium': return '🟡';
      case 'high': return '🟠';
      case 'critical': return '🔴';
    }
  };

  return (
    <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-2xl transition-all duration-300 border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
          <Eye className={`h-6 w-6 ${isDarkMode ? 'text-cyan-400' : 'text-blue-600'}`} />
          <span>Threat Intelligence</span>
        </h2>
        <div className={`px-3 py-1 rounded-full text-xs font-semibold ${isDarkMode ? 'bg-purple-500/20 text-purple-400' : 'bg-purple-100 text-purple-600'} animate-pulse`}>
          INTEL
        </div>
      </div>

      <div className="space-y-4">
        {threats.map((threat, index) => (
          <div 
            key={threat.id}
            className={`p-4 rounded-lg border transition-all duration-300 hover:shadow-lg ${
              threat.severity === 'critical' && index === 0 ? 'animate-pulse' : ''
            } ${getSeverityColor(threat.severity)}`}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center space-x-2">
                <span className="text-lg">{getSeverityIcon(threat.severity)}</span>
                <div>
                  <h3 className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    {threat.threatType}
                  </h3>
                  <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Source: {threat.source}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Clock className={`h-4 w-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`} />
                <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  {new Date(threat.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
            
            <p className={`text-sm mb-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              {threat.description}
            </p>
            
            <div className="flex items-center justify-between">
              <span className={`text-xs font-semibold px-2 py-1 rounded-full uppercase ${getSeverityColor(threat.severity)}`}>
                {threat.severity}
              </span>
              <button className={`flex items-center space-x-1 text-xs font-medium transition-colors ${
                isDarkMode ? 'text-cyan-400 hover:text-cyan-300' : 'text-blue-600 hover:text-blue-500'
              }`}>
                <span>View Details</span>
                <ExternalLink className="h-3 w-3" />
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className={`mt-4 p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'} border ${isDarkMode ? 'border-gray-600' : 'border-gray-200'}`}>
        <div className="flex items-center justify-between text-sm">
          <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Intelligence Sources</span>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className={`text-xs font-medium ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
              4 Active Feeds
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};