import React, { useState, useEffect } from 'react';
import { PieChart, BarChart3 } from 'lucide-react';
import { AttackData } from '../types';
import { generateAttackDistribution } from '../utils/mockData';

interface AttackDistributionPanelProps {
  isDarkMode: boolean;
}

export const AttackDistributionPanel: React.FC<AttackDistributionPanelProps> = ({ isDarkMode }) => {
  const [attackData, setAttackData] = useState<AttackData[]>([]);
  const [viewType, setViewType] = useState<'pie' | 'bar'>('pie');

  useEffect(() => {
    setAttackData(generateAttackDistribution());
    
    const interval = setInterval(() => {
      setAttackData(generateAttackDistribution());
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const total = attackData.reduce((sum, item) => sum + item.count, 0);

  const renderPieChart = () => {
    let cumulativePercentage = 0;
    
    return (
      <div className="flex items-center justify-center">
        <div className="relative w-48 h-48">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke={isDarkMode ? '#374151' : '#e5e7eb'}
              strokeWidth="8"
            />
            {attackData.map((item, index) => {
              const percentage = (item.count / total) * 100;
              const strokeDasharray = `${(percentage * 283) / 100} 283`;
              const strokeDashoffset = -((cumulativePercentage * 283) / 100);
              
              cumulativePercentage += percentage;
              
              return (
                <circle
                  key={item.type}
                  cx="50"
                  cy="50"
                  r="45"
                  fill="none"
                  stroke={item.color}
                  strokeWidth="8"
                  strokeDasharray={strokeDasharray}
                  strokeDashoffset={strokeDashoffset}
                  className="transition-all duration-1000 hover:stroke-[10]"
                />
              );
            })}
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{total}</div>
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>Total Events</div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderBarChart = () => {
    const maxCount = Math.max(...attackData.map(item => item.count));
    
    return (
      <div className="space-y-3">
        {attackData.map((item) => (
          <div key={item.type} className="flex items-center space-x-3">
            <div className={`w-16 text-sm font-medium ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              {item.type}
            </div>
            <div className="flex-1 relative">
              <div className={`h-6 rounded-full ${isDarkMode ? 'bg-gray-700' : 'bg-gray-200'} overflow-hidden`}>
                <div
                  className="h-full transition-all duration-1000 ease-out rounded-full"
                  style={{
                    width: `${(item.count / maxCount) * 100}%`,
                    backgroundColor: item.color
                  }}
                />
              </div>
              <div className={`absolute right-2 top-0 h-6 flex items-center text-xs font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                {item.count}
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-xl p-6 shadow-2xl transition-all duration-300 border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} h-full`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-2`}>
          <BarChart3 className={`h-6 w-6 ${isDarkMode ? 'text-cyan-400' : 'text-blue-600'}`} />
          <span>Attack Distribution</span>
        </h2>
        
        <div className="flex space-x-2">
          <button
            onClick={() => setViewType('pie')}
            className={`p-2 rounded-lg transition-all duration-200 ${
              viewType === 'pie' 
                ? isDarkMode ? 'bg-cyan-600 text-white' : 'bg-blue-600 text-white'
                : isDarkMode ? 'bg-gray-700 text-gray-400 hover:text-white' : 'bg-gray-200 text-gray-600 hover:text-gray-900'
            }`}
          >
            <PieChart className="h-4 w-4" />
          </button>
          <button
            onClick={() => setViewType('bar')}
            className={`p-2 rounded-lg transition-all duration-200 ${
              viewType === 'bar' 
                ? isDarkMode ? 'bg-cyan-600 text-white' : 'bg-blue-600 text-white'
                : isDarkMode ? 'bg-gray-700 text-gray-400 hover:text-white' : 'bg-gray-200 text-gray-600 hover:text-gray-900'
            }`}
          >
            <BarChart3 className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="mb-6">
        {viewType === 'pie' ? renderPieChart() : renderBarChart()}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {attackData.map((item) => (
          <div key={item.type} className="flex items-center space-x-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: item.color }}
            />
            <span className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              {item.type}
            </span>
            <span className={`text-sm font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              ({Math.round((item.count / total) * 100)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};