import React from 'react';
import { Shield, Wifi, Activity, Globe, Zap } from 'lucide-react';

interface HeaderProps {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

export const Header: React.FC<HeaderProps> = ({ isDarkMode, toggleDarkMode }) => {
  return (
    <header className={`${isDarkMode ? 'bg-gray-900 border-gray-700' : 'bg-white border-gray-200'} border-b-2 transition-all duration-300 relative overflow-hidden`}>
      {/* Animated background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 animate-pulse"></div>
      </div>
      
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Shield className={`h-12 w-12 ${isDarkMode ? 'text-cyan-400' : 'text-blue-600'} drop-shadow-lg animate-pulse`} />
              <Wifi className={`h-6 w-6 ${isDarkMode ? 'text-green-400' : 'text-green-600'} absolute -top-1 -right-1 animate-bounce`} />
              <div className={`absolute -inset-2 rounded-full ${isDarkMode ? 'bg-cyan-400/20' : 'bg-blue-600/20'} animate-ping`}></div>
            </div>
            <div>
              <h1 className={`text-3xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} tracking-tight`}>
                GC-FL-IDS: Real-Time IoT Intrusion Detection
              </h1>
              <p className={`text-lg ${isDarkMode ? 'text-cyan-300' : 'text-blue-600'} mt-1`}>
                Graph + Federated Learning based Smart Security
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Real-time status indicators */}
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-1">
                <Activity className={`h-4 w-4 ${isDarkMode ? 'text-green-400' : 'text-green-600'} animate-pulse`} />
                <span className={`text-xs font-medium ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>LIVE</span>
              </div>
              <div className="flex items-center space-x-1">
                <Globe className={`h-4 w-4 ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                <span className={`text-xs font-medium ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`}>GLOBAL</span>
              </div>
              <div className="flex items-center space-x-1">
                <Zap className={`h-4 w-4 ${isDarkMode ? 'text-yellow-400' : 'text-yellow-600'} animate-pulse`} />
                <span className={`text-xs font-medium ${isDarkMode ? 'text-yellow-400' : 'text-yellow-600'}`}>AI</span>
              </div>
            </div>
            
            <div className={`px-4 py-2 rounded-lg ${isDarkMode ? 'bg-gray-800 border border-gray-600' : 'bg-gray-100 border border-gray-300'} transition-all duration-300`}>
              <span className={`text-sm font-semibold ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
                System Status: ACTIVE
              </span>
            </div>
            
            <button
              onClick={toggleDarkMode}
              className={`p-3 rounded-lg transition-all duration-300 transform hover:scale-110 ${
                isDarkMode 
                  ? 'bg-yellow-400 text-gray-900 hover:bg-yellow-300' 
                  : 'bg-gray-800 text-white hover:bg-gray-700'
              } shadow-lg hover:shadow-xl hover:shadow-yellow-400/25`}
            >
              {isDarkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};