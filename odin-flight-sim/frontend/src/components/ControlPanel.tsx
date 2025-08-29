import { useState, useEffect } from "react";
import { Play, Pause, RotateCcw, Zap, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { currentMissionStats } from "@/data/missionData";

interface ControlPanelProps {
  isSimulationRunning: boolean;
  isPaused?: boolean;
  onPlaySimulation: () => void;
  onPauseSimulation: () => void;
  onResetSimulation: () => void;
  onInjectHazard: () => void;
  onRecalculateRoute: () => void;
  currentPhase?: string;
  fuelRemaining?: number;
}

export default function ControlPanel({
  isSimulationRunning,
  isPaused = false,
  onPlaySimulation,
  onPauseSimulation,
  onResetSimulation,
  onInjectHazard,
  onRecalculateRoute,
  currentPhase = currentMissionStats.currentPhase,
  fuelRemaining = currentMissionStats.fuelRemaining,
}: ControlPanelProps) {
  const [missionPhase, setMissionPhase] = useState(currentPhase);

  // Update mission phase when prop changes
  useEffect(() => {
    setMissionPhase(currentPhase);
  }, [currentPhase]);

  return (
    <Card className="h-full bg-card border-border flex flex-col overflow-hidden">
      <div className="p-3 border-b border-border shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-foreground">MISSION CONTROL</h2>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="font-mono-mission text-xs">
              {currentMissionStats.missionElapsedTime}
            </Badge>
            <div
              className={`w-2 h-2 rounded-full ${
                currentMissionStats.systemsStatus === "All Green"
                  ? "bg-accent glow-success"
                  : "bg-destructive"
              }`}
            />
          </div>
        </div>
      </div>

      <div className="flex-1 p-3 space-y-3 overflow-y-auto">
        {/* Mission Status */}
        <div className="grid grid-cols-1 gap-2">
          <div className="space-y-1">
            <label className="text-xs font-mono-mission text-muted-foreground">
              MISSION PHASE
            </label>
            <div className="bg-muted rounded p-2">
              <span className="text-xs font-mono-mission text-foreground truncate block">
                {missionPhase}
              </span>
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-xs font-mono-mission text-muted-foreground">
              CREW STATUS
            </label>
            <div className="bg-muted rounded p-2">
              <span
                className={`text-xs font-mono-mission ${
                  currentMissionStats.crewStatus === "Nominal"
                    ? "text-accent"
                    : "text-destructive"
                }`}
              >
                {currentMissionStats.crewStatus}
              </span>
            </div>
          </div>
        </div>

        {/* Primary Controls */}
        <div className="space-y-2">
          <label className="text-xs font-mono-mission text-muted-foreground">
            SIMULATION
          </label>
          <div className="flex flex-col space-y-2">
            {!isSimulationRunning || isPaused ? (
              <Button
                onClick={onPlaySimulation}
                className="bg-accent text-accent-foreground hover:glow-success font-mono-mission text-xs h-8"
              >
                <Play className="w-3 h-3 mr-1" />
                {isPaused ? "RESUME" : "PLAY"}
              </Button>
            ) : (
              <Button
                onClick={onPauseSimulation}
                className="bg-warning text-warning-foreground hover:glow-hazard font-mono-mission text-xs h-8"
              >
                <Pause className="w-3 h-3 mr-1" />
                PAUSE
              </Button>
            )}

            <Button
              onClick={onResetSimulation}
              variant="outline"
              className="font-mono-mission border-border hover:bg-muted text-xs h-8"
            >
              <RotateCcw className="w-3 h-3 mr-1" />
              RESET
            </Button>
          </div>
        </div>

        {/* Mission Operations */}
        <div className="space-y-2">
          <label className="text-xs font-mono-mission text-muted-foreground">
            OPERATIONS
          </label>
          <div className="flex flex-col space-y-2">
            <Button
              onClick={onInjectHazard}
              variant="outline"
              className="font-mono-mission border-warning text-warning hover:bg-warning hover:text-warning-foreground text-xs h-8"
            >
              <Zap className="w-3 h-3 mr-1" />
              HAZARD
            </Button>

            <Button
              onClick={onRecalculateRoute}
              variant="outline"
              className="font-mono-mission border-primary text-primary hover:bg-primary hover:text-primary-foreground text-xs h-8"
            >
              <Settings className="w-3 h-3 mr-1" />
              RECALC
            </Button>
          </div>
        </div>

        {/* System Status Indicators */}
        <div className="space-y-2">
          <label className="text-xs font-mono-mission text-muted-foreground">
            SYSTEMS
          </label>
          <div className="grid grid-cols-2 gap-1 text-xs">
            <div className="flex items-center justify-between bg-muted rounded p-1">
              <span className="font-mono-mission truncate">Nav</span>
              <div className="w-2 h-2 bg-accent rounded-full" />
            </div>
            <div className="flex items-center justify-between bg-muted rounded p-1">
              <span className="font-mono-mission truncate">Prop</span>
              <div className="w-2 h-2 bg-accent rounded-full" />
            </div>
            <div className="flex items-center justify-between bg-muted rounded p-1">
              <span className="font-mono-mission truncate">Life</span>
              <div className="w-2 h-2 bg-accent rounded-full" />
            </div>
            <div className="flex items-center justify-between bg-muted rounded p-1">
              <span className="font-mono-mission truncate">Comm</span>
              <div className="w-2 h-2 bg-accent rounded-full" />
            </div>
          </div>
        </div>

        {/* Fuel Status */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono-mission text-muted-foreground">
              Fuel
            </span>
            <span className="text-xs font-mono-mission text-accent">
              {fuelRemaining.toFixed(1)}%
            </span>
          </div>
          <Progress value={fuelRemaining} className="h-1" />
        </div>

        {/* Emergency Protocols */}
        <div className="p-2 bg-destructive/10 border border-destructive rounded">
          <div className="flex items-center space-x-2 mb-1">
            <div className="w-2 h-2 bg-destructive rounded-full animate-pulse" />
            <span className="text-xs font-mono-mission text-destructive font-bold">
              EMERGENCY READY
            </span>
          </div>
          <p className="text-xs text-muted-foreground font-mono-mission">
            Abort sequence armed.
          </p>
        </div>
      </div>
    </Card>
  );
}
