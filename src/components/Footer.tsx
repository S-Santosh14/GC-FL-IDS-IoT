import React, { useState, useEffect } from 'react';
import { Clock, Users, Zap } from 'lucide-react';

interface FooterProps {
  isDarkMode: boolean;
}

export const Footer: React.FC<FooterProps> = ({ isDarkMode }) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [flRound, setFlRound] = useState(12);
  const [trainingProgress, setTrainingProgress] = useState(0);

  useEffect(() => {
    const timeInterval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    const progressInterval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          setFlRound(r => r + 1);
          return 0;
        }
        return prev + Math.random() * 5;
      });
    }, 2000);

    return () => {
      clearInterval(timeInterval);
      clearInterval(progressInterval);
    };
  }, []);

  return (
    <footer className={`${isDarkMode ? 'bg-gray-900 border-gray-700' : 'bg-white border-gray-200'} border-t-2 mt-8 transition-all duration-300 relative overflow-hidden`}>
      {/* Animated background */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500 via-cyan-500 to-blue-500 animate-pulse"></div>
      </div>
      
      <div className="container mx-auto px-6 py-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
          {/* Federated Learning Status */}
          <div className="flex items-center space-x-4 relative">
            <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-purple-500/20' : 'bg-purple-100'} border ${isDarkMode ? 'border-purple-500/30' : 'border-purple-200'}`}>
              <Users className={`h-6 w-6 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'} animate-pulse`} />
            </div>
            <div>
              <div className={`text-sm font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                FL Round #{flRound} in progress
              </div>
              <div className="w-48 h-2 bg-gray-300 rounded-full mt-2 overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-purple-500 to-purple-600 transition-all duration-1000 ease-out relative"
                  style={{ width: `${trainingProgress}%` }}
                >
                  <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                </div>
              </div>
              <div className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'} mt-1`}>
                Training Progress: {Math.round(trainingProgress)}%
              </div>
            </div>
          </div>

          {/* Last Updated */}
          <div className="flex items-center justify-center space-x-3">
            <Clock className={`h-5 w-5 ${isDarkMode ? 'text-cyan-400' : 'text-blue-600'}`} />
            <div className="text-center">
              <div className={`text-sm font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                Last Updated
              </div>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {currentTime.toLocaleString()}
              </div>
            </div>
          </div>

          {/* System Performance */}
          <div className="flex items-center justify-end space-x-4 relative">
            <div>
              <div className={`text-sm font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'} text-right`}>
                System Performance
              </div>
              <div className={`text-sm ${isDarkMode ? 'text-green-400' : 'text-green-600'} text-right flex items-center justify-end space-x-1`}>
                <Zap className="h-4 w-4 animate-pulse" />
                <span>Optimal (99.2%)</span>
              </div>
            </div>
            <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-green-500/20' : 'bg-green-100'} border ${isDarkMode ? 'border-green-500/30' : 'border-green-200'}`}>
              <Zap className={`h-6 w-6 ${isDarkMode ? 'text-green-400' : 'text-green-600'} animate-pulse`} />
            </div>
          </div>
        </div>

        <div className={`mt-6 pt-4 border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} text-center`}>
          <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Developed by <span className={`font-semibold ${isDarkMode ? 'text-cyan-400' : 'text-blue-600'} animate-pulse`}>CyberSec Research Team</span> | 
            GC-FL-IDS v2.1 | Real-time IoT Security Monitoring Platform | 
            <span className={`font-semibold ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>🔒 Secured by AI</span>
          </p>
        </div>
      </div>
    </footer>
  );
};