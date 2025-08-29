import { useCallback, useState } from "react";
import cosmicBackground from "@/assets/cosmic-background.jpg";
import TrajectoryView3D from "@/components/TrajectoryView3D";
import HazardPanel from "@/components/HazardPanel";
import MissionLogs from "@/components/MissionLogs";
import ControlPanel from "@/components/ControlPanel";
import MetricsDashboard from "@/components/MetricsDashboard";
import SimulationGraph3D from "@/components/SimulationGraph3D";
import TimeControlPanel from "@/components/TimeControlPanel";
import LiveChartsPanel from "@/components/LiveChartsPanel";
import CrewResourcePanel from "@/components/CrewResourcePanel";
import AIDecisionSupport from "@/components/AIDecisionSupport";
import { Badge } from "@/components/ui/badge";
import {
  Satellite,
  AlertTriangle,
  Activity,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useSimulation } from "@/hooks/useSimulation";
import type { Hazard } from "@/data/missionData";

const Index = () => {
  const {
    state,
    metrics,
    isRunning,
    isPaused,
    currentTrajectory,
    missionLogs,
    startSimulation,
    pauseSimulation,
    resumeSimulation,
    resetSimulation,
    setTrajectory,
    injectHazard,
    addMissionLog,
  } = useSimulation();

  const [showLiveCharts, setShowLiveCharts] = useState(false);

  const handleTrajectorySelect = useCallback(
    (trajectoryId: string) => {
      setTrajectory(trajectoryId);
    },
    [setTrajectory]
  );

  const handleHazardInject = useCallback(
    (hazard: Hazard) => {
      injectHazard(hazard);
    },
    [injectHazard]
  );

  const handlePlaySimulation = useCallback(() => {
    if (isPaused) {
      resumeSimulation();
    } else {
      startSimulation();
    }
  }, [isPaused, resumeSimulation, startSimulation]);

  const handlePauseSimulation = useCallback(() => {
    pauseSimulation();
  }, [pauseSimulation]);

  const handleResetSimulation = useCallback(() => {
    resetSimulation();
  }, [resetSimulation]);

  const handleInjectHazard = useCallback(() => {
    // Trigger hazard injection in HazardPanel
    window.dispatchEvent(new CustomEvent("inject-hazard"));
  }, []);

  const handleRecalculateRoute = useCallback(() => {
    // Switch to safe route if not already
    if (currentTrajectory !== "safe-route-alpha") {
      setTrajectory("safe-route-alpha");
    }

    addMissionLog({
      source: "ODIN-AI",
      message:
        "Route recalculation complete. Optimized for current conditions.",
      priority: "Info",
    });
  }, [currentTrajectory, setTrajectory, addMissionLog]);

  // Determine hazard status from metrics
  const hazardStatus =
    metrics?.hazardLevel === "Critical"
      ? "critical"
      : metrics?.hazardLevel === "Warning"
      ? "warning"
      : "nominal";

  // Format mission time
  const formatTime = (hours: number) => {
    const h = Math.floor(hours);
    const m = Math.floor((hours - h) * 60);
    const s = Math.floor(((hours - h) * 60 - m) * 60);
    return `T+${h.toString().padStart(2, "0")}:${m
      .toString()
      .padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="h-screen w-screen bg-gradient-cosmic relative overflow-y-scroll">
      {/* Cosmic background */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-fixed opacity-30"
        style={{
          backgroundImage: `url(${cosmicBackground})`,
        }}
      />
      <div className="absolute inset-0 bg-background/80" />

      {/* Header */}
      <header className="relative z-10 bg-card/90 backdrop-blur-sm border-b border-border shrink-0">
        <div className="px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Satellite className="w-6 h-6 text-primary glow-primary" />
                <div>
                  <h1 className="text-xl font-bold text-foreground">ODIN</h1>
                  <p className="text-xs text-muted-foreground font-mono-mission">
                    Autonomous Earth–Moon Mission Planner
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-primary" />
                <Badge variant="outline" className="font-mono-mission text-xs">
                  {state ? formatTime(state.currentTime) : "T+00:00:00"}
                </Badge>
              </div>

              <div className="flex items-center space-x-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    hazardStatus === "critical"
                      ? "bg-destructive animate-hazard-pulse"
                      : hazardStatus === "warning"
                      ? "bg-warning animate-pulse"
                      : "bg-accent glow-success"
                  }`}
                />
                <Badge
                  className={`text-xs ${
                    hazardStatus === "critical"
                      ? "bg-destructive text-destructive-foreground"
                      : hazardStatus === "warning"
                      ? "bg-warning text-warning-foreground"
                      : "bg-accent text-accent-foreground"
                  }`}
                >
                  {hazardStatus === "critical"
                    ? "CRITICAL"
                    : hazardStatus === "warning"
                    ? "WARNING"
                    : "NOMINAL"}
                </Badge>
              </div>

              {hazardStatus !== "nominal" && (
                <AlertTriangle
                  className={`w-4 h-4 ${
                    hazardStatus === "critical"
                      ? "text-destructive"
                      : "text-warning"
                  }`}
                />
              )}

              {/* Live Charts Toggle */}
              <button
                onClick={() => setShowLiveCharts(!showLiveCharts)}
                className="flex items-center space-x-1 px-2 py-1 bg-primary/10 border border-primary rounded text-xs font-mono-mission text-primary hover:bg-primary/20 transition-colors"
              >
                <span>LIVE CHARTS</span>
                {showLiveCharts ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Live Charts Panel (Expandable) */}
      {showLiveCharts && (
        <div className="relative z-10 p-4 border-b border-border bg-card/80 backdrop-blur-sm">
          <LiveChartsPanel />
        </div>
      )}

      {/* Main Interface */}
      <div className="relative z-10 flex-1 p-4 overflow-hidden">
        <div className="h-full grid grid-cols-12 gap-3">
          {/* Left Column - Controls and Metrics */}
          <div className="col-span-3 flex flex-col gap-3 overflow-hidden">
            {/* Control Panel */}
            <div className="flex-1 min-h-0">
              <ControlPanel
                isSimulationRunning={isRunning}
                isPaused={isPaused}
                onPlaySimulation={handlePlaySimulation}
                onPauseSimulation={handlePauseSimulation}
                onResetSimulation={handleResetSimulation}
                onInjectHazard={handleInjectHazard}
                onRecalculateRoute={handleRecalculateRoute}
                currentPhase={state?.currentPhase || "Pre-Launch"}
                fuelRemaining={state?.fuelRemaining || 100}
              />
            </div>

            {/* Metrics Dashboard */}
            <div className="flex-1 min-h-0">
              <MetricsDashboard
                activeTrajectory={currentTrajectory}
                metrics={metrics}
                state={state}
              />
            </div>

            {/* AI Decision Support */}
            <div className="flex-1 min-h-0">
              <AIDecisionSupport />
            </div>
          </div>

          {/* Center Column - 3D Views */}
          <div className="col-span-6 flex flex-col gap-3 overflow-hidden">
            {/* 3D Trajectory View */}
            <div className="h-96 bg-card/90 backdrop-blur-sm rounded-lg border border-border overflow-hidden">
              <TrajectoryView3D
                activeTrajectory={currentTrajectory}
                onTrajectorySelect={handleTrajectorySelect}
                simulationState={state}
              />
            </div>

            {/* 3D Simulation Graph */}
            <div className="h-64 bg-card/90 backdrop-blur-sm rounded-lg border border-border overflow-hidden">
              <SimulationGraph3D
                activeTrajectory={currentTrajectory}
                metrics={metrics}
              />
            </div>

            {/* Time Control Panel */}
            <div className="h-auto">
              <TimeControlPanel />
            </div>
          </div>

          {/* Right Column - Hazards and Logs */}
          <div className="col-span-3 flex flex-col gap-3 overflow-hidden">
            {/* Crew Resources Panel */}
            <div className="min-h-[200px]">
              <CrewResourcePanel />
            </div>

            {/* Hazard Detection Panel */}
            <div className="flex-1 min-h-0">
              <HazardPanel
                onHazardInject={handleHazardInject}
                activeHazards={state?.activeHazards || []}
              />
            </div>

            {/* Mission Logs */}
            <div className="flex-1 min-h-0">
              <MissionLogs logs={missionLogs} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
