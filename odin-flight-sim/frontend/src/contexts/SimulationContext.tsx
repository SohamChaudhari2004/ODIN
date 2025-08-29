import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  ReactNode,
} from "react";
import {
  SimulationEngine,
  SimulationState,
  LiveMetrics,
} from "@/lib/simulationEngine";
import {
  trajectories,
  type Trajectory,
  type Hazard,
  type MissionLog,
} from "@/data/missionData";

interface SimulationContextType {
  engine: SimulationEngine | null;
  state: SimulationState | null;
  metrics: LiveMetrics | null;
  isRunning: boolean;
  isPaused: boolean;
  currentTrajectory: string;
  missionLogs: MissionLog[];

  // Actions
  startSimulation: () => void;
  pauseSimulation: () => void;
  resumeSimulation: () => void;
  resetSimulation: () => void;
  setTrajectory: (trajectoryId: string) => void;
  injectHazard: (hazard: Hazard) => void;
  removeHazard: (hazardId: string) => void;
  setTimeScale: (scale: number) => void;
  addMissionLog: (log: Omit<MissionLog, "id" | "timestamp">) => void;
}

const SimulationContext = createContext<SimulationContextType | null>(null);

export { SimulationContext };

interface SimulationProviderProps {
  children: ReactNode;
}

export function SimulationProvider({ children }: SimulationProviderProps) {
  const [engine, setEngine] = useState<SimulationEngine | null>(null);
  const [state, setState] = useState<SimulationState | null>(null);
  const [metrics, setMetrics] = useState<LiveMetrics | null>(null);
  const [currentTrajectory, setCurrentTrajectory] = useState("baseline");
  const [missionLogs, setMissionLogs] = useState<MissionLog[]>([]);

  // Initialize simulation engine
  useEffect(() => {
    const initialTrajectory =
      trajectories.find((t) => t.id === currentTrajectory) || trajectories[0];
    const newEngine = new SimulationEngine(initialTrajectory);

    const unsubscribe = newEngine.subscribe((newState, newMetrics) => {
      setState(newState);
      setMetrics(newMetrics);
    });

    setEngine(newEngine);

    // Initial state update
    setState(newEngine.getState());
    setMetrics(newEngine.getMetrics());

    return () => {
      unsubscribe();
    };
  }, [currentTrajectory]);

  const addMissionLog = useCallback(
    (log: Omit<MissionLog, "id" | "timestamp">) => {
      const newLog: MissionLog = {
        ...log,
        id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toTimeString().split(" ")[0],
      };

      setMissionLogs((prev) => [...prev, newLog]);
    },
    []
  );

  const startSimulation = useCallback(() => {
    if (!engine) return;

    engine.start();
    addMissionLog({
      source: "Flight Controller",
      message: "Mission simulation started. All systems nominal.",
      priority: "Info",
    });
  }, [engine, addMissionLog]);

  const pauseSimulation = useCallback(() => {
    if (!engine) return;

    engine.pause();
    addMissionLog({
      source: "Flight Controller",
      message: "Simulation paused. Maintaining current status.",
      priority: "Info",
    });
  }, [engine, addMissionLog]);

  const resumeSimulation = useCallback(() => {
    if (!engine) return;

    engine.resume();
    addMissionLog({
      source: "Flight Controller",
      message: "Simulation resumed. Continuing mission profile.",
      priority: "Info",
    });
  }, [engine, addMissionLog]);

  const resetSimulation = useCallback(() => {
    if (!engine) return;

    engine.reset();
    setMissionLogs([]);
    addMissionLog({
      source: "Flight Controller",
      message: "Simulation reset to baseline parameters.",
      priority: "Info",
    });
  }, [engine, addMissionLog]);

  const setTrajectory = useCallback(
    (trajectoryId: string) => {
      const trajectory = trajectories.find((t) => t.id === trajectoryId);
      if (!trajectory || !engine) return;

      setCurrentTrajectory(trajectoryId);
      engine.setTrajectory(trajectory);

      addMissionLog({
        source: "ODIN-AI",
        message: `Trajectory switched to ${trajectory.name}. Recalculating mission parameters...`,
        priority: "Info",
      });
    },
    [engine, addMissionLog]
  );

  const injectHazard = useCallback(
    (hazard: Hazard) => {
      if (!engine) return;

      engine.addHazard(hazard);

      const responses = [
        `${hazard.type} detected. Analyzing impact on current trajectory.`,
        `Hazard assessment complete. Severity: ${hazard.severity}. Evaluating alternatives.`,
        `Recommendation: Consider trajectory adjustment for optimal safety margins.`,
        `Monitoring ${hazard.type}. Crew safety protocols activated.`,
      ];

      addMissionLog({
        source: "ODIN-AI",
        message: responses[Math.floor(Math.random() * responses.length)],
        priority:
          hazard.severity === "Critical"
            ? "Critical"
            : hazard.severity === "High"
            ? "Warning"
            : "Info",
      });
    },
    [engine, addMissionLog]
  );

  const removeHazard = useCallback(
    (hazardId: string) => {
      if (!engine) return;

      engine.removeHazard(hazardId);
      addMissionLog({
        source: "Hazard Detection",
        message: `Hazard ${hazardId} cleared. Threat level reduced.`,
        priority: "Info",
      });
    },
    [engine, addMissionLog]
  );

  const setTimeScale = useCallback(
    (scale: number) => {
      if (!engine) return;

      engine.setTimeScale(scale);
      addMissionLog({
        source: "Flight Controller",
        message: `Simulation time scale adjusted to ${scale}x real-time.`,
        priority: "Info",
      });
    },
    [engine, addMissionLog]
  );

  const value: SimulationContextType = {
    engine,
    state,
    metrics,
    isRunning: state?.isRunning || false,
    isPaused: state?.isPaused || false,
    currentTrajectory,
    missionLogs,

    startSimulation,
    pauseSimulation,
    resumeSimulation,
    resetSimulation,
    setTrajectory,
    injectHazard,
    removeHazard,
    setTimeScale,
    addMissionLog,
  };

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
}
