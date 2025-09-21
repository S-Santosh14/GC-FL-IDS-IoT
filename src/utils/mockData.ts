import { Alert, AttackData, NetworkNode, NetworkConnection, SystemMetrics, ThreatIntelligence } from '../types';

const attackTypes = ['DoS', 'Probe', 'Malware', 'Brute Force', 'Data Exfiltration', 'Normal'];
const deviceNames = ['IoT-Camera-01', 'Smart-Thermostat-02', 'Security-Sensor-03', 'Router-Main', 'Smart-Lock-04', 'Light-Controller-05'];
const riskFactors: Alert['riskFactor'][] = ['Normal', 'Suspicious', 'Intrusion'];

export const generateMockAlert = (): Alert => {
  const attackType = attackTypes[Math.floor(Math.random() * attackTypes.length)];
  const riskFactor = attackType === 'Normal' ? 'Normal' : 
                     Math.random() > 0.6 ? 'Intrusion' : 'Suspicious';
  
  return {
    id: Math.random().toString(36).substr(2, 9),
    time: new Date().toLocaleTimeString(),
    device: deviceNames[Math.floor(Math.random() * deviceNames.length)],
    attackType,
    riskFactor,
    confidence: Math.floor(Math.random() * 40) + 60, // 60-100%
    explanation: generateExplanation(attackType, riskFactor)
  };
};

const generateExplanation = (attackType: string, riskFactor: string): string => {
  const explanations = {
    'DoS': 'Unusual high traffic volume detected, potential denial of service attack',
    'Probe': 'Multiple port scanning attempts from external source',
    'Malware': 'Suspicious executable detected in device communication',
    'Brute Force': 'Multiple failed authentication attempts detected',
    'Data Exfiltration': 'Large data transfer to unknown external endpoint',
    'Normal': 'Standard device communication pattern'
  };
  
  return explanations[attackType as keyof typeof explanations] || 'Unknown threat pattern detected';
};

export const generateAttackDistribution = (): AttackData[] => [
  { type: 'DoS', count: Math.floor(Math.random() * 50) + 10, color: '#ef4444' },
  { type: 'Probe', count: Math.floor(Math.random() * 30) + 5, color: '#f97316' },
  { type: 'Malware', count: Math.floor(Math.random() * 20) + 2, color: '#eab308' },
  { type: 'Brute Force', count: Math.floor(Math.random() * 15) + 3, color: '#8b5cf6' },
  { type: 'Normal', count: Math.floor(Math.random() * 100) + 50, color: '#10b981' }
];

export const generateNetworkNodes = (): NetworkNode[] => [
  { id: 'router', name: 'Main Router', type: 'router', x: 200, y: 100, status: 'normal' },
  { id: 'camera1', name: 'Security Camera', type: 'device', x: 100, y: 50, status: 'suspicious' },
  { id: 'thermostat', name: 'Smart Thermostat', type: 'device', x: 300, y: 50, status: 'normal' },
  { id: 'sensor1', name: 'Door Sensor', type: 'sensor', x: 50, y: 150, status: 'normal' },
  { id: 'sensor2', name: 'Motion Sensor', type: 'sensor', x: 350, y: 150, status: 'compromised' },
  { id: 'lock', name: 'Smart Lock', type: 'device', x: 150, y: 200, status: 'normal' },
  { id: 'lights', name: 'Light Controller', type: 'device', x: 250, y: 200, status: 'normal' }
];

export const generateNetworkConnections = (): NetworkConnection[] => [
  { from: 'router', to: 'camera1', status: 'suspicious', traffic: Math.random() * 100, latency: Math.random() * 50 + 10 },
  { from: 'router', to: 'thermostat', status: 'normal', traffic: Math.random() * 100, latency: Math.random() * 50 + 10 },
  { from: 'router', to: 'sensor1', status: 'normal', traffic: Math.random() * 100, latency: Math.random() * 50 + 10 },
  { from: 'router', to: 'sensor2', status: 'suspicious', traffic: Math.random() * 100, latency: Math.random() * 50 + 10 },
  { from: 'router', to: 'lock', status: 'normal', traffic: Math.random() * 100, latency: Math.random() * 50 + 10 },
  { from: 'router', to: 'lights', status: 'normal', traffic: Math.random() * 100, latency: Math.random() * 50 + 10 }
];

export const generateSystemMetrics = (): SystemMetrics => ({
  cpuUsage: Math.floor(Math.random() * 30) + 20,
  memoryUsage: Math.floor(Math.random() * 40) + 30,
  networkThroughput: Math.floor(Math.random() * 1000) + 500,
  activeConnections: Math.floor(Math.random() * 50) + 100,
  blockedAttacks: Math.floor(Math.random() * 20) + 5
});

export const generateThreatIntelligence = (): ThreatIntelligence[] => [
  {
    id: '1',
    source: 'MITRE ATT&CK',
    threatType: 'APT29 Campaign',
    severity: 'critical',
    description: 'Advanced persistent threat targeting IoT infrastructure',
    timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString()
  },
  {
    id: '2',
    source: 'Threat Intel Feed',
    threatType: 'Botnet Activity',
    severity: 'high',
    description: 'Mirai variant detected in network traffic patterns',
    timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString()
  },
  {
    id: '3',
    source: 'CVE Database',
    threatType: 'Zero-day Exploit',
    severity: 'medium',
    description: 'Potential vulnerability in IoT device firmware',
    timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString()
  }
];