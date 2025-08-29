export interface Hazard {
  id: string;
  timestamp: string;
  type: "CME" | "Solar Flare" | "Debris Conjunction" | "Radiation Storm";
  severity: "Low" | "Medium" | "High" | "Critical";
  description: string;
  coordinates?: { lat: number; lon: number };
}

export interface Trajectory {
  id: string;
  name: string;
  deltaV: number; // m/s
  travelTime: number; // hours
  radiationExposure: number; // percentage
  fuelConsumption: number; // kg
  risk: "Low" | "Medium" | "High";
  points: Array<{ x: number; y: number; z: number; time: number }>;
}

export interface MissionLog {
  id: string;
  timestamp: string;
  source: "ODIN-AI" | "Flight Controller" | "Navigation" | "Hazard Detection";
  message: string;
  priority: "Info" | "Warning" | "Critical";
}

export const hazards: Hazard[] = [
  {
    id: "haz-001",
    timestamp: "T+72:14:30",
    type: "CME",
    severity: "High",
    description:
      "Coronal Mass Ejection detected. Velocity: 1,200 km/s. Expected arrival: T+74:00:00",
    coordinates: { lat: 15.2, lon: -45.8 },
  },
  {
    id: "haz-002",
    timestamp: "T+156:45:12",
    type: "Debris Conjunction",
    severity: "Medium",
    description:
      "Space debris cluster detected in lunar approach corridor. Risk of collision: 12%",
  },
  {
    id: "haz-003",
    timestamp: "T+48:30:00",
    type: "Solar Flare",
    severity: "Critical",
    description:
      "X-class solar flare in progress. Radiation levels exceeding safety thresholds",
  },
  {
    id: "haz-004",
    timestamp: "T+24:15:45",
    type: "Radiation Storm",
    severity: "High",
    description:
      "Geomagnetic storm detected. Proton flux levels: 1000 pfu. Crew shelter advised.",
  },
  {
    id: "haz-005",
    timestamp: "T+96:22:33",
    type: "Debris Conjunction",
    severity: "Low",
    description:
      "Micrometeorite cloud detected 500km ahead. Minor course adjustment recommended.",
  },
  {
    id: "haz-006",
    timestamp: "T+12:44:12",
    type: "CME",
    severity: "Medium",
    description:
      "Minor coronal mass ejection. Velocity: 600 km/s. Minimal mission impact expected.",
  },
  {
    id: "haz-007",
    timestamp: "T+144:18:55",
    type: "Solar Flare",
    severity: "High",
    description:
      "M-class solar flare detected. Communication blackout possible for 2-4 hours.",
  },
  {
    id: "haz-008",
    timestamp: "T+36:11:28",
    type: "Radiation Storm",
    severity: "Critical",
    description:
      "Solar energetic particle event in progress. Immediate shelter protocol required.",
  },
];

export const trajectories: Trajectory[] = [
  {
    id: "baseline",
    name: "Baseline Hohmann Transfer",
    deltaV: 3100,
    travelTime: 72,
    radiationExposure: 85,
    fuelConsumption: 2450,
    risk: "Medium",
    points: [
      { x: 0, y: 0, z: 0, time: 0 },
      { x: 50, y: 20, z: 5, time: 24 },
      { x: 120, y: 45, z: 8, time: 48 },
      { x: 200, y: 60, z: 12, time: 72 },
    ],
  },
  {
    id: "safe-route-alpha",
    name: "Safe Route Alpha",
    deltaV: 3280,
    travelTime: 84,
    radiationExposure: 25,
    fuelConsumption: 2680,
    risk: "Low",
    points: [
      { x: 0, y: 0, z: 0, time: 0 },
      { x: 45, y: 35, z: 15, time: 28 },
      { x: 110, y: 65, z: 25, time: 56 },
      { x: 200, y: 60, z: 12, time: 84 },
    ],
  },
  {
    id: "emergency-fast",
    name: "Emergency Fast Transit",
    deltaV: 4200,
    travelTime: 48,
    radiationExposure: 95,
    fuelConsumption: 3200,
    risk: "High",
    points: [
      { x: 0, y: 0, z: 0, time: 0 },
      { x: 80, y: 15, z: 3, time: 16 },
      { x: 160, y: 40, z: 8, time: 32 },
      { x: 200, y: 60, z: 12, time: 48 },
    ],
  },
  {
    id: "fuel-efficient",
    name: "Fuel Efficient Route",
    deltaV: 2850,
    travelTime: 96,
    radiationExposure: 45,
    fuelConsumption: 2200,
    risk: "Low",
    points: [
      { x: 0, y: 0, z: 0, time: 0 },
      { x: 35, y: 25, z: 8, time: 32 },
      { x: 85, y: 50, z: 15, time: 64 },
      { x: 200, y: 60, z: 12, time: 96 },
    ],
  },
  {
    id: "lunar-polar",
    name: "Lunar Polar Approach",
    deltaV: 3350,
    travelTime: 78,
    radiationExposure: 65,
    fuelConsumption: 2750,
    risk: "Medium",
    points: [
      { x: 0, y: 0, z: 0, time: 0 },
      { x: 55, y: 30, z: 20, time: 26 },
      { x: 125, y: 55, z: 35, time: 52 },
      { x: 200, y: 60, z: 45, time: 78 },
    ],
  },
];

export const missionLogs: MissionLog[] = [
  {
    id: "log-001",
    timestamp: "T+72:14:45",
    source: "ODIN-AI",
    message:
      "CME detected. Analyzing trajectory alternatives. Recommending switch to Safe Route Alpha for 90% radiation reduction.",
    priority: "Critical",
  },
  {
    id: "log-002",
    timestamp: "T+72:15:12",
    source: "Navigation",
    message:
      "Trajectory recalculation complete. New delta-V requirement: 3,280 m/s (+180 m/s). Fuel reserves sufficient.",
    priority: "Warning",
  },
  {
    id: "log-003",
    timestamp: "T+72:15:45",
    source: "ODIN-AI",
    message:
      "Route optimization successful. Travel time increased by 12 hours, but crew radiation exposure reduced by 71%.",
    priority: "Info",
  },
  {
    id: "log-004",
    timestamp: "T+72:16:00",
    source: "Flight Controller",
    message:
      "Approved trajectory change to Safe Route Alpha. Initiating course correction burn in T-15 minutes.",
    priority: "Info",
  },
  {
    id: "log-005",
    timestamp: "T+48:30:15",
    source: "Hazard Detection",
    message:
      "X-class solar flare detected. Duration: 6-8 hours. Immediate shelter protocol activated.",
    priority: "Critical",
  },
];

export const currentMissionStats = {
  missionElapsedTime: "T+72:16:23",
  currentPhase: "Trans-Lunar Injection",
  trajectoryActive: "safe-route-alpha",
  crewStatus: "Nominal",
  systemsStatus: "All Green",
  fuelRemaining: 78.5, // percentage
  distanceToMoon: 156420, // km
  currentVelocity: 10.8, // km/s
};
