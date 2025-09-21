import React, { useState, useEffect } from 'react';
import { Network, Wifi, WifiOff } from 'lucide-react';
import { NetworkNode, NetworkConnection } from '../types';
import { generateNetworkNodes, generateNetworkConnections } from '../utils/mockData';

interface NetworkGraphPanelProps {
  isDarkMode: boolean;
}

export const NetworkGraphPanel: React.FC<NetworkGraphPanelProps> = ({ isDarkMode }) => {
  const [nodes, setNodes] = useState<NetworkNode[]>([]);
  const [connections, setConnections] = useState<NetworkConnection[]>([]);
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);

  useEffect(() => {
    setNodes(generateNetworkNodes());
    setConnections(generateNetworkConnections());

    const interval = setInterval(() => {
      // Randomly update node statuses
      setNodes(prev => prev.map(node => ({
        ...node,
        status: Math.random() > 0.8 ? 
          (Math.random() > 0.5 ? 'suspicious' : 'compromised') : 
          node.status
      })));
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  const getNodeColor = (status: NetworkNode['status']) => {
    switch (status) {
      case 'normal': return '#10b981';
      case 'suspicious': return '#f59e0b';
      case 'compromised': return '#ef4444';
    }
  };

  const getConnectionColor = (status: NetworkConnection['status']) => {
    switch (status) {
      case 'normal': return isDarkMode ? '#6b7280' : '#9ca3af';
      case 'suspicious': return '#ef4444';
    }
  };

  const getNodeIcon = (type: NetworkNode['type']) => {
    switch (type) {
      case 'router': return '🔗';
      case 'device': return '📱';
      case 'sensor': return '🔍';
    }
  };

  return (
    <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-2xl transition-all duration-300 border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} h-full`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
          <Network className={`h-6 w-6 ${isDarkMode ? 'text-cyan-400' : 'text-blue-600'}`} />
          <span>IoT Network Topology</span>
        </h2>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Normal</span>
          </div>
          <div className="flex items-center space-x-2 text-sm">
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Suspicious</span>
          </div>
          <div className="flex items-center space-x-2 text-sm">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>Compromised</span>
          </div>
        </div>
      </div>

      <div className={`relative ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'} rounded-lg p-4 h-80 overflow-hidden border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        <svg className="w-full h-full" viewBox="0 0 400 250">
          {/* Render connections */}
          {connections.map((conn, index) => {
            const fromNode = nodes.find(n => n.id === conn.from);
            const toNode = nodes.find(n => n.id === conn.to);
            
            if (!fromNode || !toNode) return null;
            
            return (
              <g key={index}>
                <line
                  x1={fromNode.x}
                  y1={fromNode.y}
                  x2={toNode.x}
                  y2={toNode.y}
                  stroke={getConnectionColor(conn.status)}
                  strokeWidth={conn.status === 'suspicious' ? 3 : 2}
                  strokeDasharray={conn.status === 'suspicious' ? '5,5' : 'none'}
                  className={conn.status === 'suspicious' ? 'animate-pulse' : ''}
                />
                {/* Traffic flow animation */}
                <circle
                  r="2"
                  fill={conn.status === 'suspicious' ? '#ef4444' : '#10b981'}
                  className="opacity-75"
                >
                  <animateMotion
                    dur="3s"
                    repeatCount="indefinite"
                    path={`M${fromNode.x},${fromNode.y} L${toNode.x},${toNode.y}`}
                  />
                </circle>
                {/* Connection info on hover */}
                <text
                  x={(fromNode.x + toNode.x) / 2}
                  y={(fromNode.y + toNode.y) / 2 - 10}
                  textAnchor="middle"
                  className={`text-xs font-medium opacity-0 hover:opacity-100 transition-opacity ${isDarkMode ? 'fill-gray-300' : 'fill-gray-700'}`}
                >
                  {Math.round(conn.traffic)}MB/s
                </text>
                <text
                  x={(fromNode.x + toNode.x) / 2}
                  y={(fromNode.y + toNode.y) / 2 + 5}
                  textAnchor="middle"
                  className={`text-xs opacity-0 hover:opacity-100 transition-opacity ${isDarkMode ? 'fill-gray-400' : 'fill-gray-600'}`}
                >
                  {Math.round(conn.latency)}ms
                </text>
                {/* Hover area */}
                <line
                  x1={fromNode.x}
                  y1={fromNode.y}
                  x2={toNode.x}
                  y2={toNode.y}
                  stroke="transparent"
                  strokeWidth="20"
                  className="cursor-pointer"
                />
                {conn.status === 'suspicious' && (
                  <circle
                    cx={(fromNode.x + toNode.x) / 2}
                    cy={(fromNode.y + toNode.y) / 2}
                    r="3"
                    fill="#ef4444"
                    className="animate-ping"
                  />
                )}
              </g>
            );
          })}

          {/* Render nodes */}
          {nodes.map((node) => (
            <g key={node.id}>
              <circle
                cx={node.x}
                cy={node.y}
                r={node.type === 'router' ? 20 : 15}
                fill={getNodeColor(node.status)}
                stroke={isDarkMode ? '#374151' : '#e5e7eb'}
                strokeWidth="2"
                className={`cursor-pointer transition-all duration-300 hover:r-18 ${
                  node.status === 'compromised' ? 'animate-pulse' : ''
                }`}
                onClick={() => setSelectedNode(node)}
              />
              <text
                x={node.x}
                y={node.y + 5}
                textAnchor="middle"
                className={`text-xs font-bold pointer-events-none ${
                  node.status === 'compromised' ? 'fill-white' : isDarkMode ? 'fill-gray-900' : 'fill-white'
                }`}
              >
                {getNodeIcon(node.type)}
              </text>
              <text
                x={node.x}
                y={node.y + 35}
                textAnchor="middle"
                className={`text-xs font-medium pointer-events-none ${isDarkMode ? 'fill-gray-300' : 'fill-gray-700'}`}
              >
                {node.name.split(' ')[0]}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} border ${isDarkMode ? 'border-gray-600' : 'border-gray-200'}`}>
          <div className="flex items-center justify-between">
            <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Connected Devices</span>
            <Wifi className={`h-4 w-4 ${isDarkMode ? 'text-green-400' : 'text-green-600'}`} />
          </div>
          <div className={`text-2xl font-bold mt-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            {nodes.filter(n => n.status !== 'compromised').length}
          </div>
        </div>
        
        <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-100'} border ${isDarkMode ? 'border-gray-600' : 'border-gray-200'}`}>
          <div className="flex items-center justify-between">
            <span className={`text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>Threats Active</span>
            <WifiOff className={`h-4 w-4 ${isDarkMode ? 'text-red-400' : 'text-red-600'}`} />
          </div>
          <div className={`text-2xl font-bold mt-1 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
            {nodes.filter(n => n.status === 'compromised' || n.status === 'suspicious').length}
          </div>
        </div>
      </div>

      {/* Node Details Modal */}
      {selectedNode && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 max-w-md mx-4 shadow-2xl border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
                <span>{getNodeIcon(selectedNode.type)}</span>
                <span>Device Details</span>
              </h3>
              <button
                onClick={() => setSelectedNode(null)}
                className={`${isDarkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-900'} transition-colors`}
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <span className={`text-sm font-semibold ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Name: </span>
                <span className={isDarkMode ? 'text-white' : 'text-gray-900'}>{selectedNode.name}</span>
              </div>
              
              <div>
                <span className={`text-sm font-semibold ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Type: </span>
                <span className={isDarkMode ? 'text-white' : 'text-gray-900'}>{selectedNode.type}</span>
              </div>
              
              <div>
                <span className={`text-sm font-semibold ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Status: </span>
                <span className={`font-semibold capitalize`} style={{ color: getNodeColor(selectedNode.status) }}>
                  {selectedNode.status}
                </span>
              </div>
              
              <div>
                <span className={`text-sm font-semibold ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Position: </span>
                <span className={isDarkMode ? 'text-white' : 'text-gray-900'}>({selectedNode.x}, {selectedNode.y})</span>
              </div>
              
              {selectedNode.status !== 'normal' && (
                <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-red-900/30' : 'bg-red-50'} border-l-4 border-red-500`}>
                  <p className={`text-sm font-medium text-red-${isDarkMode ? '300' : '700'}`}>
                    {selectedNode.status === 'suspicious' 
                      ? 'Unusual network behavior detected'
                      : 'Device compromised - immediate action required'
                    }
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};