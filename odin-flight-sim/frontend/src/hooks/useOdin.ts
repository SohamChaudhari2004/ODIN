/**
 * ODIN React Hook
 * Provides React integration for ODIN backend services
 */

import { useState, useEffect, useCallback, useRef } from "react";
import odinAPI, {
  MissionStatus,
  TrajectoryData,
  HazardData,
  SpaceWeatherData,
  MissionInitRequest,
  MissionConstraints,
} from "@/lib/odinAPI";

interface UseOdinOptions {
  enableRealTime?: boolean;
  autoReconnect?: boolean;
}

interface SystemStatus {
  system_name: string;
  version: string;
  operational: boolean;
  subsystems: Record<string, boolean>;
}

interface OdinState {
  // Connection state
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;

  // Mission state
  missionStatus: MissionStatus | null;
  systemStatus: SystemStatus | null;

  // Data
  currentTrajectory: TrajectoryData | null;
  availableTrajectories: TrajectoryData[];
  activeHazards: HazardData[];
  spaceWeather: SpaceWeatherData | null;

  // AI
  aiRecommendations: string[];
  lastDecisionExplanation: string | null;
}

export const useOdin = (options: UseOdinOptions = {}) => {
  const { enableRealTime = true, autoReconnect = true } = options;

  const [state, setState] = useState<OdinState>({
    isConnected: false,
    isLoading: false,
    error: null,
    missionStatus: null,
    systemStatus: null,
    currentTrajectory: null,
    availableTrajectories: [],
    activeHazards: [],
    spaceWeather: null,
    aiRecommendations: [],
    lastDecisionExplanation: null,
  });

  const wsInitialized = useRef(false);

  // Update state helper
  const updateState = useCallback((updates: Partial<OdinState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  // Error handler
  const handleError = useCallback(
    (error: Error | unknown, context: string) => {
      console.error(`ODIN Error (${context}):`, error);
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error occurred";
      updateState({
        error: errorMessage,
        isLoading: false,
      });
    },
    [updateState]
  );

  // =============================================================================
  // MISSION MANAGEMENT
  // =============================================================================

  const initializeMission = useCallback(
    async (request: MissionInitRequest) => {
      updateState({ isLoading: true, error: null });
      try {
        const missionStatus = await odinAPI.initializeMission(request);
        updateState({
          missionStatus,
          isLoading: false,
          error: null,
        });
        return missionStatus;
      } catch (error) {
        handleError(error, "initializeMission");
        throw error;
      }
    },
    [updateState, handleError]
  );

  const startAutonomousMode = useCallback(
    async (missionId: string) => {
      updateState({ isLoading: true });
      try {
        const result = await odinAPI.startAutonomousMode(missionId);
        updateState({ isLoading: false });
        return result;
      } catch (error) {
        handleError(error, "startAutonomousMode");
        throw error;
      }
    },
    [updateState, handleError]
  );

  const stopAutonomousMode = useCallback(
    async (missionId: string) => {
      updateState({ isLoading: true });
      try {
        const result = await odinAPI.stopAutonomousMode(missionId);
        updateState({ isLoading: false });
        return result;
      } catch (error) {
        handleError(error, "stopAutonomousMode");
        throw error;
      }
    },
    [updateState, handleError]
  );

  // =============================================================================
  // TRAJECTORY MANAGEMENT
  // =============================================================================

  const calculateTrajectory = useCallback(
    async (
      startTime: string,
      destination: string,
      constraints?: MissionConstraints
    ) => {
      updateState({ isLoading: true });
      try {
        const trajectory = await odinAPI.calculateTrajectory(
          startTime,
          destination,
          constraints
        );
        updateState({
          currentTrajectory: trajectory,
          isLoading: false,
        });
        return trajectory;
      } catch (error) {
        handleError(error, "calculateTrajectory");
        throw error;
      }
    },
    [updateState, handleError]
  );

  const getTrajectoryOptions = useCallback(
    async (destination: string) => {
      updateState({ isLoading: true });
      try {
        const trajectories = await odinAPI.getTrajectoryOptions(destination);
        updateState({
          availableTrajectories: trajectories,
          isLoading: false,
        });
        return trajectories;
      } catch (error) {
        handleError(error, "getTrajectoryOptions");
        throw error;
      }
    },
    [updateState, handleError]
  );

  const replanTrajectory = useCallback(
    async (missionId: string, newConstraints?: MissionConstraints) => {
      updateState({ isLoading: true });
      try {
        const trajectory = await odinAPI.replanTrajectory(
          missionId,
          newConstraints
        );
        updateState({
          currentTrajectory: trajectory,
          isLoading: false,
        });
        return trajectory;
      } catch (error) {
        handleError(error, "replanTrajectory");
        throw error;
      }
    },
    [updateState, handleError]
  );

  // =============================================================================
  // HAZARD MONITORING
  // =============================================================================

  const getCurrentHazards = useCallback(async () => {
    try {
      const hazards = await odinAPI.getCurrentHazards();
      updateState({ activeHazards: hazards });
      return hazards;
    } catch (error) {
      handleError(error, "getCurrentHazards");
      throw error;
    }
  }, [updateState, handleError]);

  const predictHazards = useCallback(
    async (timeframe: number) => {
      try {
        const hazards = await odinAPI.predictHazards(timeframe);
        return hazards;
      } catch (error) {
        handleError(error, "predictHazards");
        throw error;
      }
    },
    [handleError]
  );

  const injectHazard = useCallback(
    async (hazardType: string, severity: string) => {
      try {
        const result = await odinAPI.injectHazard(hazardType, severity);
        // Refresh hazards after injection
        await getCurrentHazards();
        return result;
      } catch (error) {
        handleError(error, "injectHazard");
        throw error;
      }
    },
    [getCurrentHazards, handleError]
  );

  // =============================================================================
  // SPACE WEATHER
  // =============================================================================

  const getSpaceWeather = useCallback(async () => {
    try {
      const weather = await odinAPI.getSpaceWeather();
      updateState({ spaceWeather: weather });
      return weather;
    } catch (error) {
      handleError(error, "getSpaceWeather");
      throw error;
    }
  }, [updateState, handleError]);

  // =============================================================================
  // AI SERVICES
  // =============================================================================

  const getAIRecommendations = useCallback(
    async (missionId: string) => {
      try {
        const result = await odinAPI.getAIRecommendations(missionId);
        updateState({ aiRecommendations: result.recommendations });
        return result.recommendations;
      } catch (error) {
        handleError(error, "getAIRecommendations");
        throw error;
      }
    },
    [updateState, handleError]
  );

  const explainDecision = useCallback(
    async (decisionId: string) => {
      try {
        const result = await odinAPI.explainDecision(decisionId);
        updateState({ lastDecisionExplanation: result.explanation });
        return result;
      } catch (error) {
        handleError(error, "explainDecision");
        throw error;
      }
    },
    [updateState, handleError]
  );

  // =============================================================================
  // SYSTEM STATUS
  // =============================================================================

  const checkSystemStatus = useCallback(async () => {
    try {
      const status = await odinAPI.getSystemStatus();
      updateState({ systemStatus: status });
      return status;
    } catch (error) {
      handleError(error, "checkSystemStatus");
      throw error;
    }
  }, [updateState, handleError]);

  const healthCheck = useCallback(async () => {
    try {
      const health = await odinAPI.healthCheck();
      updateState({
        isConnected: health.status === "ok",
        error: health.status !== "ok" ? "Backend not healthy" : null,
      });
      return health;
    } catch (error) {
      updateState({
        isConnected: false,
        error: "Backend not reachable",
      });
      handleError(error, "healthCheck");
      throw error;
    }
  }, [updateState, handleError]);

  // =============================================================================
  // REAL-TIME UPDATES
  // =============================================================================

  const initializeWebSocket = useCallback(() => {
    if (!enableRealTime || wsInitialized.current) return;

    odinAPI.connectWebSocket({
      onMissionUpdate: (data) => {
        updateState({ missionStatus: data });
      },
      onHazardAlert: (data) => {
        setState((prev) => ({
          ...prev,
          activeHazards: [...prev.activeHazards, data],
        }));
      },
      onDecisionUpdate: (data) => {
        // Handle decision updates
        console.log("Decision update:", data);
      },
      onTrajectoryUpdate: (data) => {
        updateState({ currentTrajectory: data });
      },
    });

    wsInitialized.current = true;
    updateState({ isConnected: true });
  }, [enableRealTime, updateState]);

  // =============================================================================
  // EFFECTS
  // =============================================================================

  // Initialize connection on mount
  useEffect(() => {
    const initialize = async () => {
      try {
        await healthCheck();
        await checkSystemStatus();

        if (enableRealTime) {
          initializeWebSocket();
        }
      } catch (error) {
        console.error("Failed to initialize ODIN connection:", error);
      }
    };

    initialize();

    // Cleanup WebSocket on unmount
    return () => {
      if (wsInitialized.current) {
        odinAPI.disconnectWebSocket();
        wsInitialized.current = false;
      }
    };
  }, [enableRealTime, healthCheck, checkSystemStatus, initializeWebSocket]);

  // Auto-reconnect logic
  useEffect(() => {
    if (!autoReconnect || state.isConnected) return;

    const reconnectInterval = setInterval(async () => {
      try {
        await healthCheck();
        if (enableRealTime && !wsInitialized.current) {
          initializeWebSocket();
        }
      } catch (error) {
        // Connection still failed, will try again
      }
    }, 5000);

    return () => clearInterval(reconnectInterval);
  }, [
    autoReconnect,
    state.isConnected,
    enableRealTime,
    healthCheck,
    initializeWebSocket,
  ]);

  return {
    // State
    ...state,

    // Mission Management
    initializeMission,
    startAutonomousMode,
    stopAutonomousMode,

    // Trajectory Management
    calculateTrajectory,
    getTrajectoryOptions,
    replanTrajectory,

    // Hazard Monitoring
    getCurrentHazards,
    predictHazards,
    injectHazard,

    // Space Weather
    getSpaceWeather,

    // AI Services
    getAIRecommendations,
    explainDecision,

    // System
    checkSystemStatus,
    healthCheck,
  };
};
