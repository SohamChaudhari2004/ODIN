/**
 *const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";Dynamic Mission Data Service
 * Fetches real-time mission data from ODIN backend APIs
 */

import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Fallback data for when API is unavailable
const FALLBACK_HAZARDS = [
  {
    id: "fallback-001",
    timestamp: new Date().toISOString(),
    type: "Solar Flare" as const,
    severity: "Medium" as const,
    description: "Simulated solar flare event (API offline)",
  },
  {
    id: "fallback-002",
    timestamp: new Date().toISOString(),
    type: "CME" as const,
    severity: "High" as const,
    description: "Simulated CME event (API offline)",
  },
];

// Types remain the same but we'll fetch data dynamically
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

export interface CurrentMissionStats {
  missionElapsedTime: string;
  currentPhase: string;
  trajectoryActive: string;
  crewStatus: string;
  systemsStatus: string;
  fuelRemaining: number; // percentage
  distanceToMoon: number; // km
  currentVelocity: number; // km/s
}

class DynamicMissionDataService {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  constructor() {
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    try {
      this.ws = new WebSocket(`ws://localhost:8000/ws`);

      this.ws.onopen = () => {
        console.log("🚀 Connected to ODIN mission data stream");
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.notifyListeners(data.type || "update", data);
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      this.ws.onclose = () => {
        console.log("📡 Disconnected from ODIN mission data stream");
        // Attempt to reconnect after 5 seconds
        setTimeout(() => this.initializeWebSocket(), 5000);
      };

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
    } catch (error) {
      console.error("Failed to initialize WebSocket:", error);
    }
  }

  private notifyListeners(type: string, data: any) {
    const typeListeners = this.listeners.get(type);
    if (typeListeners) {
      typeListeners.forEach((callback) => callback(data));
    }
  }

  public subscribe(type: string, callback: (data: any) => void) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(callback);

    // Return unsubscribe function
    return () => {
      const typeListeners = this.listeners.get(type);
      if (typeListeners) {
        typeListeners.delete(callback);
      }
    };
  }

  // Fetch current hazards from space weather service
  async getCurrentHazards(): Promise<Hazard[]> {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/hazards/current`);
      const hazards = response.data.hazards || [];

      // Transform backend data to frontend format
      return hazards.map((hazard: any, index: number) => ({
        id: hazard.hazard_id || `haz-${Date.now()}-${index}`,
        timestamp: hazard.timestamp || new Date().toISOString(),
        type: this.mapHazardType(hazard.type),
        severity: this.mapSeverity(hazard.severity),
        description: hazard.description || "Space weather event detected",
        coordinates: hazard.coordinates,
      }));
    } catch (error) {
      console.error("Error fetching current hazards:", error);
      return this.generateFallbackHazards();
    }
  }

  // Fetch predicted hazards
  async getPredictedHazards(horizonHours: number = 72): Promise<Hazard[]> {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/hazards/predict?horizon_hours=${horizonHours}`
      );
      const predictions = response.data.predicted_hazards || [];

      return predictions.map((hazard: any, index: number) => ({
        id: `pred-${Date.now()}-${index}`,
        timestamp: hazard.time_to_peak
          ? new Date(Date.now() + hazard.time_to_peak * 3600000).toISOString()
          : new Date().toISOString(),
        type: this.mapHazardType(hazard.type),
        severity: this.mapSeverity(hazard.severity),
        description:
          hazard.impact_description ||
          hazard.description ||
          "Predicted space weather event",
        coordinates: hazard.coordinates,
      }));
    } catch (error) {
      console.error("Error fetching predicted hazards:", error);
      // Return simulated predicted hazards when API unavailable
      return [
        {
          id: `pred-${Date.now()}-sim1`,
          timestamp: new Date(Date.now() + 24 * 3600000).toISOString(), // 24h from now
          type: "Solar Flare" as const,
          severity: "Medium" as const,
          description:
            "Predicted M-class solar flare (Simulated - API temporarily unavailable)",
        },
        {
          id: `pred-${Date.now()}-sim2`,
          timestamp: new Date(Date.now() + 48 * 3600000).toISOString(), // 48h from now
          type: "CME" as const,
          severity: "High" as const,
          description:
            "Predicted coronal mass ejection (Simulated - API temporarily unavailable)",
        },
      ];
    }
  }

  // Fetch available trajectories
  async getTrajectoryOptions(
    destination: string = "Moon"
  ): Promise<Trajectory[]> {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/trajectory/options?destination=${destination}`
      );
      const trajectories = response.data.trajectories || [];

      return trajectories.map((traj: any) => ({
        id:
          traj.id || traj.name?.toLowerCase().replace(/\s+/g, "-") || "unknown",
        name: traj.name || "Unknown Trajectory",
        deltaV: traj.delta_v || traj.totalDeltaV || 3000,
        travelTime: traj.travel_time || traj.duration || 72,
        radiationExposure: traj.radiation_exposure || traj.radiationRisk || 50,
        fuelConsumption: traj.fuel_consumption || traj.fuelRequired || 2500,
        risk: this.mapRiskLevel(traj.risk || traj.safety_score),
        points: this.generateTrajectoryPoints(
          traj.points || [],
          traj.travel_time || 72
        ),
      }));
    } catch (error) {
      console.error("Error fetching trajectories:", error);
      return this.generateFallbackTrajectories();
    }
  }

  // Fetch current mission status
  async getCurrentMissionStats(): Promise<CurrentMissionStats> {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/mission/status`);
      const status = response.data;

      return {
        missionElapsedTime: status.mission_time
          ? `T+${status.mission_time.toFixed(2)}:00:00`
          : "T+00:00:00",
        currentPhase: status.current_phase || status.phase || "Pre-Launch",
        trajectoryActive:
          status.active_trajectory?.id ||
          status.trajectory_active ||
          "baseline",
        crewStatus: status.crew_status || "Nominal",
        systemsStatus: status.systems_status || "All Green",
        fuelRemaining: status.fuel_remaining || 100,
        distanceToMoon:
          status.distance_to_target || status.distance_to_moon || 384400,
        currentVelocity: status.current_velocity || status.velocity || 0,
      };
    } catch (error) {
      console.error("Error fetching mission status:", error);
      return this.generateFallbackMissionStats();
    }
  }

  // Fetch mission logs
  async getMissionLogs(): Promise<MissionLog[]> {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/odin/decision-logs`
      );
      const logs = response.data.decision_logs || response.data.logs || [];

      return logs.map((log: any, index: number) => ({
        id: log.id || `log-${Date.now()}-${index}`,
        timestamp: log.timestamp || new Date().toISOString(),
        source: this.mapLogSource(log.source || log.system),
        message: log.message || log.description || "System update",
        priority: this.mapPriority(log.priority || log.level),
      }));
    } catch (error) {
      console.error("Error fetching mission logs:", error);
      return this.generateFallbackLogs();
    }
  }

  // Fetch space weather data and convert to hazards
  async getSpaceWeatherHazards(): Promise<Hazard[]> {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/space-weather/current`
      );
      const spaceWeather = response.data.space_weather;
      const hazards: Hazard[] = [];

      // Convert active space weather events to hazards
      if (spaceWeather?.active_events) {
        spaceWeather.active_events.forEach((event: any, index: number) => {
          hazards.push({
            id: `sw-${Date.now()}-${index}`,
            timestamp: event.start_time || new Date().toISOString(),
            type: this.mapSpaceWeatherType(event.type),
            severity: this.mapSpaceWeatherSeverity(event),
            description:
              event.description || this.generateEventDescription(event),
            coordinates: event.coordinates,
          });
        });
      }

      // Check for high radiation levels
      if (
        spaceWeather?.radiation_environment?.radiation_storm_level !== "none"
      ) {
        hazards.push({
          id: `radiation-${Date.now()}`,
          timestamp: new Date().toISOString(),
          type: "Radiation Storm",
          severity: this.mapRadiationSeverity(
            spaceWeather.radiation_environment.radiation_storm_level
          ),
          description: `Radiation storm detected: ${spaceWeather.radiation_environment.radiation_storm_level} level`,
        });
      }

      return hazards;
    } catch (error) {
      console.error("Error fetching space weather hazards:", error);
      return [];
    }
  }

  // Helper methods for data transformation
  private mapHazardType(type: string): Hazard["type"] {
    const typeMap: Record<string, Hazard["type"]> = {
      solar_flare: "Solar Flare",
      coronal_mass_ejection: "CME",
      cme: "CME",
      debris: "Debris Conjunction",
      debris_conjunction: "Debris Conjunction",
      radiation: "Radiation Storm",
      radiation_storm: "Radiation Storm",
      solar_energetic_particles: "Radiation Storm",
    };
    return typeMap[type] || "Solar Flare";
  }

  private mapSeverity(severity: string): Hazard["severity"] {
    const severityMap: Record<string, Hazard["severity"]> = {
      S1: "Low",
      S2: "Medium",
      S3: "High",
      S4: "Critical",
      S5: "Critical",
      minor: "Low",
      moderate: "Medium",
      major: "High",
      severe: "Critical",
      low: "Low",
      medium: "Medium",
      high: "High",
      critical: "Critical",
    };
    return severityMap[severity?.toLowerCase()] || "Medium";
  }

  private mapRiskLevel(risk: any): Trajectory["risk"] {
    if (typeof risk === "number") {
      if (risk > 0.8) return "Low";
      if (risk > 0.6) return "Medium";
      return "High";
    }
    const riskMap: Record<string, Trajectory["risk"]> = {
      low: "Low",
      medium: "Medium",
      high: "High",
    };
    return riskMap[risk?.toLowerCase()] || "Medium";
  }

  private mapLogSource(source: string): MissionLog["source"] {
    const sourceMap: Record<string, MissionLog["source"]> = {
      odin: "ODIN-AI",
      ai: "ODIN-AI",
      navigation: "Navigation",
      hazard: "Hazard Detection",
      controller: "Flight Controller",
    };
    return sourceMap[source?.toLowerCase()] || "ODIN-AI";
  }

  private mapPriority(priority: string): MissionLog["priority"] {
    const priorityMap: Record<string, MissionLog["priority"]> = {
      info: "Info",
      warning: "Warning",
      critical: "Critical",
      low: "Info",
      medium: "Warning",
      high: "Critical",
    };
    return priorityMap[priority?.toLowerCase()] || "Info";
  }

  private mapSpaceWeatherType(type: string): Hazard["type"] {
    const typeMap: Record<string, Hazard["type"]> = {
      solar_flare: "Solar Flare",
      coronal_mass_ejection: "CME",
      cme_impact: "CME",
      solar_energetic_particles: "Radiation Storm",
    };
    return typeMap[type] || "Solar Flare";
  }

  private mapSpaceWeatherSeverity(event: any): Hazard["severity"] {
    if (event.class) {
      // Solar flare classes: C < M < X
      if (event.class.startsWith("X")) return "Critical";
      if (event.class.startsWith("M")) return "High";
      if (event.class.startsWith("C")) return "Medium";
      return "Low";
    }
    return this.mapSeverity(event.severity || "medium");
  }

  private mapRadiationSeverity(level: string): Hazard["severity"] {
    const severityMap: Record<string, Hazard["severity"]> = {
      S1: "Low",
      S2: "Low",
      S3: "Medium",
      S4: "High",
      S5: "Critical",
    };
    return severityMap[level] || "Medium";
  }

  private generateEventDescription(event: any): string {
    switch (event.type) {
      case "solar_flare":
        return `${event.class || "M-class"} solar flare detected. Duration: ${
          event.duration_minutes || 60
        } minutes.`;
      case "coronal_mass_ejection":
        return `CME detected. Speed: ${
          event.speed_km_s || 500
        } km/s. Estimated arrival: ${event.estimated_arrival || "TBD"}.`;
      case "solar_energetic_particles":
        return `Solar particle event. Flux level: ${
          event.flux_level || "elevated"
        }. Duration: ${event.duration_hours || 12} hours.`;
      default:
        return "Space weather event detected.";
    }
  }

  private generateTrajectoryPoints(
    points: any[],
    duration: number
  ): Array<{ x: number; y: number; z: number; time: number }> {
    if (points && points.length > 0) {
      return points.map((p) => ({
        x: p.x || p.position?.[0] || 0,
        y: p.y || p.position?.[1] || 0,
        z: p.z || p.position?.[2] || 0,
        time: p.time || p.timestamp || 0,
      }));
    }

    // Generate fallback trajectory points
    const numPoints = 4;
    const result = [];
    for (let i = 0; i < numPoints; i++) {
      const progress = i / (numPoints - 1);
      result.push({
        x: progress * 200,
        y: progress * 60 + Math.sin(progress * Math.PI) * 20,
        z: progress * 15,
        time: progress * duration,
      });
    }
    return result;
  }

  // Fallback data methods (in case API fails)
  private generateFallbackHazards(): Hazard[] {
    return [
      {
        id: "fallback-001",
        timestamp: new Date().toISOString(),
        type: "Solar Flare",
        severity: "Medium",
        description:
          "Space weather monitoring active. No current threats detected.",
      },
    ];
  }

  private generateFallbackTrajectories(): Trajectory[] {
    return [
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
    ];
  }

  private generateFallbackMissionStats(): CurrentMissionStats {
    return {
      missionElapsedTime: "T+00:00:00",
      currentPhase: "System Initialization",
      trajectoryActive: "baseline",
      crewStatus: "Nominal",
      systemsStatus: "All Green",
      fuelRemaining: 100,
      distanceToMoon: 384400,
      currentVelocity: 0,
    };
  }

  private generateFallbackLogs(): MissionLog[] {
    return [
      {
        id: "fallback-log-001",
        timestamp: new Date().toISOString(),
        source: "ODIN-AI",
        message: "ODIN system initialized. Ready for mission planning.",
        priority: "Info",
      },
    ];
  }
}

// Create singleton instance
const dynamicMissionDataService = new DynamicMissionDataService();

// Export the service and reactive data hooks
export { dynamicMissionDataService };

// Export reactive data fetching functions
export const useDynamicMissionData = () => {
  return {
    getCurrentHazards: () => dynamicMissionDataService.getCurrentHazards(),
    getPredictedHazards: (hours?: number) =>
      dynamicMissionDataService.getPredictedHazards(hours),
    getTrajectoryOptions: (destination?: string) =>
      dynamicMissionDataService.getTrajectoryOptions(destination),
    getCurrentMissionStats: () =>
      dynamicMissionDataService.getCurrentMissionStats(),
    getMissionLogs: () => dynamicMissionDataService.getMissionLogs(),
    getSpaceWeatherHazards: () =>
      dynamicMissionDataService.getSpaceWeatherHazards(),
    subscribe: (type: string, callback: (data: any) => void) =>
      dynamicMissionDataService.subscribe(type, callback),
  };
};

// Export static types for components that still need them
export type { Hazard, Trajectory, MissionLog, CurrentMissionStats };
