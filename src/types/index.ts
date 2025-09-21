export interface Alert {
  id: string;
  time: string;
  device: string;
  attackType: string;
  riskFactor: 'Normal' | 'Suspicious' | 'Intrusion';
  confidence: number;
  explanation: string;
}

export interface AttackData {
  type: string;
  count: number;
  color: string;
}

export interface NetworkNode {
  id: string;
  name: string;
  type: 'router' | 'device' | 'sensor';
  x: number;
  y: number;
  status: 'normal' | 'suspicious' | 'compromised';
}

export interface NetworkConnection {
  from: string;
  to: string;
  status: 'normal' | 'suspicious';
  traffic: number;
  latency: number;
}

export interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkThroughput: number;
  activeConnections: number;
  blockedAttacks: number;
}

export interface ThreatIntelligence {
  id: string;
  source: string;
  threatType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  timestamp: string;
}