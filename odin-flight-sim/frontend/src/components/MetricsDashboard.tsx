import { useState, useEffect } from "react";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Fuel,
  Clock,
  Shield,
  Zap,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { trajectories, currentMissionStats } from "@/data/missionData";
import type { LiveMetrics, SimulationState } from "@/lib/simulationEngine";

interface Metric {
  id: string;
  label: string;
  value: number;
  unit: string;
  trend: "up" | "down" | "stable";
  status: "good" | "warning" | "critical";
  icon: React.ElementType;
}

interface MetricsDashboardProps {
  activeTrajectory: string;
  metrics?: LiveMetrics | null;
  state?: SimulationState | null;
}

export default function MetricsDashboard({
  activeTrajectory,
  metrics,
  state,
}: MetricsDashboardProps) {
  const [localMetrics, setLocalMetrics] = useState<Metric[]>([]);

  useEffect(() => {
    const activeTrajectoryData =
      trajectories.find((t) => t.id === activeTrajectory) || trajectories[0];

    // Use live metrics if available, otherwise fall back to trajectory data
    const deltaV = metrics?.deltaV || activeTrajectoryData.deltaV;
    const travelTime = metrics?.travelTime || activeTrajectoryData.travelTime;
    const radiationExposure =
      metrics?.radiationExposure || activeTrajectoryData.radiationExposure;
    const fuelConsumption =
      metrics?.fuelConsumption || activeTrajectoryData.fuelConsumption;

    const updatedMetrics: Metric[] = [
      {
        id: "deltaV",
        label: "Delta-V Required",
        value: deltaV,
        unit: "m/s",
        trend: deltaV > 3200 ? "up" : deltaV < 3100 ? "down" : "stable",
        status: deltaV > 4000 ? "critical" : deltaV > 3500 ? "warning" : "good",
        icon: Zap,
      },
      {
        id: "travelTime",
        label: "Travel Time",
        value: travelTime,
        unit: "hours",
        trend: travelTime > 75 ? "up" : travelTime < 70 ? "down" : "stable",
        status: travelTime > 90 ? "warning" : "good",
        icon: Clock,
      },
      {
        id: "radiation",
        label: "Radiation Exposure",
        value: radiationExposure,
        unit: "%",
        trend:
          radiationExposure > 80
            ? "up"
            : radiationExposure < 30
            ? "down"
            : "stable",
        status:
          radiationExposure > 85
            ? "critical"
            : radiationExposure > 60
            ? "warning"
            : "good",
        icon: Shield,
      },
      {
        id: "fuel",
        label: "Fuel Consumption",
        value: fuelConsumption,
        unit: "kg",
        trend:
          fuelConsumption > 2600
            ? "up"
            : fuelConsumption < 2500
            ? "down"
            : "stable",
        status: fuelConsumption > 3000 ? "warning" : "good",
        icon: Fuel,
      },
    ];

    setLocalMetrics(updatedMetrics);
  }, [activeTrajectory, metrics]);

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return TrendingUp;
      case "down":
        return TrendingDown;
      case "stable":
        return Minus;
    }
  };

  const getStatusColor = (status: "good" | "warning" | "critical") => {
    switch (status) {
      case "good":
        return "text-accent";
      case "warning":
        return "text-warning";
      case "critical":
        return "text-destructive";
    }
  };

  const getTrendColor = (
    trend: "up" | "down" | "stable",
    status: "good" | "warning" | "critical"
  ) => {
    if (status === "critical") return "text-destructive";
    if (status === "warning") return "text-warning";
    return "text-muted-foreground";
  };

  return (
    <Card className="h-full bg-card border-border flex flex-col overflow-hidden">
      <div className="p-3 border-b border-border shrink-0">
        <h2 className="text-sm font-bold text-foreground">DECISION METRICS</h2>
        <p className="text-xs text-muted-foreground font-mono-mission truncate">
          Real-time trajectory analysis
        </p>
      </div>

      <div className="flex-1 p-3 space-y-3 overflow-y-auto">
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-2">
          {localMetrics.map((metric) => {
            const IconComponent = metric.icon;
            const TrendIcon = getTrendIcon(metric.trend);

            return (
              <div
                key={metric.id}
                className={`bg-muted/50 rounded p-2 border transition-all hover:bg-muted min-h-0 ${
                  metric.status === "critical"
                    ? "border-destructive glow-hazard"
                    : metric.status === "warning"
                    ? "border-warning"
                    : "border-border"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <IconComponent
                    className={`w-3 h-3 ${getStatusColor(metric.status)}`}
                  />
                  <TrendIcon
                    className={`w-2 h-2 ${getTrendColor(
                      metric.trend,
                      metric.status
                    )}`}
                  />
                </div>

                <div className="space-y-1">
                  <p className="text-xs font-mono-mission text-muted-foreground truncate">
                    {metric.label}
                  </p>
                  <div
                    className={`text-sm font-bold font-mono-mission ${getStatusColor(
                      metric.status
                    )}`}
                  >
                    {metric.value.toLocaleString()}
                    <span className="text-xs ml-1 text-muted-foreground">
                      {metric.unit}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Mission Progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono-mission text-muted-foreground">
              Mission Progress
            </span>
            <span className="text-xs font-mono-mission text-primary">
              {state
                ? `${Math.round(
                    (state.currentTime /
                      (trajectories.find((t) => t.id === activeTrajectory)
                        ?.travelTime || 72)) *
                      100
                  )}%`
                : "0%"}
            </span>
          </div>
          <Progress
            value={
              state
                ? Math.min(
                    (state.currentTime /
                      (trajectories.find((t) => t.id === activeTrajectory)
                        ?.travelTime || 72)) *
                      100,
                    100
                  )
                : 0
            }
            className="h-1"
          />
          <div className="grid grid-cols-4 gap-1 text-xs font-mono-mission text-muted-foreground">
            <span className="truncate">Launch</span>
            <span className="truncate">TLI</span>
            <span className="truncate">Orbit</span>
            <span className="truncate">Land</span>
          </div>
        </div>

        {/* Fuel Status */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono-mission text-muted-foreground">
              Fuel
            </span>
            <span className="text-xs font-mono-mission text-accent">
              {state?.fuelRemaining?.toFixed(1) ||
                currentMissionStats.fuelRemaining}
              %
            </span>
          </div>
          <Progress
            value={state?.fuelRemaining || currentMissionStats.fuelRemaining}
            className="h-1"
          />
        </div>

        {/* Current Status */}
        <div className="grid grid-cols-1 gap-2">
          <div className="bg-muted/30 rounded p-2">
            <p className="text-xs font-mono-mission text-muted-foreground mb-1 truncate">
              Distance to Moon
            </p>
            <p className="text-xs font-mono-mission text-primary">
              {metrics?.distanceToMoon?.toLocaleString() ||
                currentMissionStats.distanceToMoon.toLocaleString()}{" "}
              km
            </p>
          </div>
          <div className="bg-muted/30 rounded p-2">
            <p className="text-xs font-mono-mission text-muted-foreground mb-1 truncate">
              Current Velocity
            </p>
            <p className="text-xs font-mono-mission text-primary">
              {metrics?.currentVelocity?.toFixed(1) ||
                currentMissionStats.currentVelocity}{" "}
              km/s
            </p>
          </div>
        </div>

        {/* AI Recommendations */}
        <div className="p-2 bg-primary/10 border border-primary rounded">
          <div className="flex items-center space-x-2 mb-1">
            <div className="w-2 h-2 bg-primary rounded-full animate-pulse-glow" />
            <span className="text-xs font-mono-mission text-primary font-bold">
              AI RECOMMENDATION
            </span>
          </div>
          <p className="text-xs text-foreground font-mono-mission">
            {metrics?.hazardLevel === "Critical"
              ? "Immediate trajectory change required for crew safety."
              : metrics?.hazardLevel === "Warning"
              ? "Monitor hazards closely. Consider route optimization."
              : "Current trajectory optimal. Monitor solar activity."}
          </p>
        </div>
      </div>
    </Card>
  );
}
