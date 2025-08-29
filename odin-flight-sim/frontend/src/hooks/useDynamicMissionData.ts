/**
 * React Hook for Dynamic Mission Data
 * Provides real-time mission data from ODIN backend
 */

import { useState, useEffect, useCallback } from "react";
import { dynamicMissionDataService } from "../data/dynamicMissionData";
import type {
  Hazard,
  Trajectory,
  MissionLog,
  CurrentMissionStats,
} from "../data/dynamicMissionData";

interface MissionData {
  hazards: Hazard[];
  predictedHazards: Hazard[];
  spaceWeatherHazards: Hazard[];
  trajectories: Trajectory[];
  missionStats: CurrentMissionStats;
  missionLogs: MissionLog[];
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

interface UseDynamicMissionDataOptions {
  autoRefresh?: boolean;
  refreshInterval?: number; // in milliseconds
  enableWebSocket?: boolean;
  predictionHorizon?: number; // hours
}

export const useDynamicMissionData = (
  options: UseDynamicMissionDataOptions = {}
) => {
  const {
    autoRefresh = true,
    refreshInterval = 30000, // 30 seconds
    enableWebSocket = true,
    predictionHorizon = 72,
  } = options;

  const [data, setData] = useState<MissionData>({
    hazards: [],
    predictedHazards: [],
    spaceWeatherHazards: [],
    trajectories: [],
    missionStats: {
      missionElapsedTime: "T+00:00:00",
      currentPhase: "Initialization",
      trajectoryActive: "baseline",
      crewStatus: "Nominal",
      systemsStatus: "All Green",
      fuelRemaining: 100,
      distanceToMoon: 384400,
      currentVelocity: 0,
    },
    missionLogs: [],
    isLoading: true,
    error: null,
    lastUpdated: null,
  });

  // Fetch all mission data
  const fetchMissionData = useCallback(async () => {
    try {
      setData((prev) => ({ ...prev, isLoading: true, error: null }));

      const [
        currentHazards,
        predictedHazards,
        spaceWeatherHazards,
        trajectories,
        missionStats,
        missionLogs,
      ] = await Promise.allSettled([
        dynamicMissionDataService.getCurrentHazards(),
        dynamicMissionDataService.getPredictedHazards(predictionHorizon),
        dynamicMissionDataService.getSpaceWeatherHazards(),
        dynamicMissionDataService.getTrajectoryOptions(),
        dynamicMissionDataService.getCurrentMissionStats(),
        dynamicMissionDataService.getMissionLogs(),
      ]);

      setData((prev) => ({
        ...prev,
        hazards:
          currentHazards.status === "fulfilled"
            ? currentHazards.value
            : prev.hazards,
        predictedHazards:
          predictedHazards.status === "fulfilled"
            ? predictedHazards.value
            : prev.predictedHazards,
        spaceWeatherHazards:
          spaceWeatherHazards.status === "fulfilled"
            ? spaceWeatherHazards.value
            : prev.spaceWeatherHazards,
        trajectories:
          trajectories.status === "fulfilled"
            ? trajectories.value
            : prev.trajectories,
        missionStats:
          missionStats.status === "fulfilled"
            ? missionStats.value
            : prev.missionStats,
        missionLogs:
          missionLogs.status === "fulfilled"
            ? missionLogs.value
            : prev.missionLogs,
        isLoading: false,
        lastUpdated: new Date(),
        error: null,
      }));
    } catch (error) {
      console.error("Error fetching mission data:", error);
      setData((prev) => ({
        ...prev,
        isLoading: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch mission data",
      }));
    }
  }, [predictionHorizon]);

  // Fetch specific data types
  const refreshHazards = useCallback(async () => {
    try {
      const [currentHazards, spaceWeatherHazards] = await Promise.all([
        dynamicMissionDataService.getCurrentHazards(),
        dynamicMissionDataService.getSpaceWeatherHazards(),
      ]);

      setData((prev) => ({
        ...prev,
        hazards: currentHazards,
        spaceWeatherHazards: spaceWeatherHazards,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      console.error("Error refreshing hazards:", error);
    }
  }, []);

  const refreshTrajectories = useCallback(async (destination = "Moon") => {
    try {
      const trajectories = await dynamicMissionDataService.getTrajectoryOptions(
        destination
      );
      setData((prev) => ({
        ...prev,
        trajectories,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      console.error("Error refreshing trajectories:", error);
    }
  }, []);

  const refreshMissionStatus = useCallback(async () => {
    try {
      const missionStats =
        await dynamicMissionDataService.getCurrentMissionStats();
      setData((prev) => ({
        ...prev,
        missionStats,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      console.error("Error refreshing mission status:", error);
    }
  }, []);

  const refreshLogs = useCallback(async () => {
    try {
      const missionLogs = await dynamicMissionDataService.getMissionLogs();
      setData((prev) => ({
        ...prev,
        missionLogs,
        lastUpdated: new Date(),
      }));
    } catch (error) {
      console.error("Error refreshing logs:", error);
    }
  }, []);

  // Initialize data and set up auto-refresh
  useEffect(() => {
    fetchMissionData();

    if (autoRefresh) {
      const interval = setInterval(fetchMissionData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchMissionData, autoRefresh, refreshInterval]);

  // Set up WebSocket subscriptions
  useEffect(() => {
    if (!enableWebSocket) return;

    const unsubscribeFunctions: (() => void)[] = [];

    // Subscribe to real-time mission updates
    unsubscribeFunctions.push(
      dynamicMissionDataService.subscribe("mission_update", (updateData) => {
        setData((prev) => ({
          ...prev,
          missionStats: { ...prev.missionStats, ...updateData },
          lastUpdated: new Date(),
        }));
      })
    );

    // Subscribe to hazard updates
    unsubscribeFunctions.push(
      dynamicMissionDataService.subscribe("hazard_alert", (hazardData) => {
        setData((prev) => ({
          ...prev,
          hazards: [...prev.hazards, hazardData],
          lastUpdated: new Date(),
        }));
      })
    );

    // Subscribe to trajectory updates
    unsubscribeFunctions.push(
      dynamicMissionDataService.subscribe(
        "trajectory_update",
        (trajectoryData) => {
          setData((prev) => ({
            ...prev,
            trajectories: prev.trajectories.map((t) =>
              t.id === trajectoryData.id ? { ...t, ...trajectoryData } : t
            ),
            lastUpdated: new Date(),
          }));
        }
      )
    );

    // Subscribe to log updates
    unsubscribeFunctions.push(
      dynamicMissionDataService.subscribe("log_update", (logData) => {
        setData((prev) => ({
          ...prev,
          missionLogs: [logData, ...prev.missionLogs].slice(0, 100), // Keep last 100 logs
          lastUpdated: new Date(),
        }));
      })
    );

    return () => {
      unsubscribeFunctions.forEach((unsubscribe) => unsubscribe());
    };
  }, [enableWebSocket]);

  // Computed values
  const allHazards = [...data.hazards, ...data.spaceWeatherHazards];
  const criticalHazards = allHazards.filter((h) => h.severity === "Critical");
  const activeTrajectory = data.trajectories.find(
    (t) => t.id === data.missionStats.trajectoryActive
  );

  return {
    // Data
    ...data,
    allHazards,
    criticalHazards,
    activeTrajectory,

    // Actions
    refresh: fetchMissionData,
    refreshHazards,
    refreshTrajectories,
    refreshMissionStatus,
    refreshLogs,

    // Utilities
    isConnected: !data.error,
    hasRecentUpdate: data.lastUpdated
      ? Date.now() - data.lastUpdated.getTime() < 60000
      : false,
  };
};

// Export types for use in components
export type { MissionData, UseDynamicMissionDataOptions };
