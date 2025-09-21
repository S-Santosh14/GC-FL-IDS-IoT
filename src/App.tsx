import React, { useState } from 'react';
import { Header } from './components/Header';
import { LiveAlertsPanel } from './components/LiveAlertsPanel';
import { AttackDistributionPanel } from './components/AttackDistributionPanel';
import { NetworkGraphPanel } from './components/NetworkGraphPanel';
import { SystemMetricsPanel } from './components/SystemMetricsPanel';
import { ThreatIntelPanel } from './components/ThreatIntelPanel';
import { Footer } from './components/Footer';

function App() {
  const [isDarkMode, setIsDarkMode] = useState(true);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className={`min-h-screen transition-all duration-300 relative overflow-hidden ${
      isDarkMode 
        ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900' 
        : 'bg-gradient-to-br from-gray-50 via-white to-gray-100'
    }`}>
      {/* Animated background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(120,119,198,0.3),rgba(255,255,255,0))]"></div>
        <div className="absolute inset-0 bg-[conic-gradient(from_0deg_at_50%_50%,rgba(120,119,198,0.1),rgba(255,255,255,0),rgba(120,119,198,0.1))] animate-spin" style={{ animationDuration: '20s' }}></div>
      </div>
      
      <Header isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode} />
      
      <main className="container mx-auto px-6 py-8 relative z-10">
        {/* Top Row - System Metrics and Threat Intelligence */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="lg:col-span-1">
            <SystemMetricsPanel isDarkMode={isDarkMode} />
          </div>
          <div className="lg:col-span-3">
            <ThreatIntelPanel isDarkMode={isDarkMode} />
          </div>
        </div>
        
        {/* Main Row - Core Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left Panel - Network Graph */}
          <div className="lg:col-span-1">
            <NetworkGraphPanel isDarkMode={isDarkMode} />
          </div>
          
          {/* Center Panel - Live Alerts */}
          <div className="lg:col-span-2">
            <LiveAlertsPanel isDarkMode={isDarkMode} />
          </div>
          
          {/* Right Panel - Attack Distribution */}
          <div className="lg:col-span-1">
            <AttackDistributionPanel isDarkMode={isDarkMode} />
          </div>
        </div>
      </main>

      <Footer isDarkMode={isDarkMode} />
    </div>
  );
}

export default App;