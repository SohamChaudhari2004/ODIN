import { useState, useEffect } from "react";
import {
  Heart,
  Droplets,
  Wind,
  Zap,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { useSimulation } from "@/hooks/useSimulation";

interface ResourceData {
  oxygen: number;
  water: number;
  food: number;
  power: number;
  temperature: number;
  pressure: number;
}

export default function CrewResourcePanel() {
  const { state, isRunning } = useSimulation();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [resources, setResources] = useState<ResourceData>({
    oxygen: 100,
    water: 100,
    food: 100,
    power: 100,
    temperature: 22,
    pressure: 101.3,
  });

  useEffect(() => {
    if (state && isRunning) {
      // Simulate resource consumption based on mission time
      const missionProgress = state.currentTime / 72; // Normalize to 72-hour baseline mission
      const consumptionRate = 1 + state.activeHazards.length * 0.1; // Hazards increase consumption

      setResources((prev) => ({
        oxygen: Math.max(0, 100 - missionProgress * 25 * consumptionRate),
        water: Math.max(0, 100 - missionProgress * 20 * consumptionRate),
        food: Math.max(0, 100 - missionProgress * 15 * consumptionRate),
        power: Math.max(10, 100 - missionProgress * 30 * consumptionRate), // Never goes below 10%
        temperature: 22 + (Math.random() - 0.5) * 4, // ±2°C variation
        pressure: 101.3 + (Math.random() - 0.5) * 2, // ±1 kPa variation
      }));
    }
  }, [state, isRunning]);

  // Reset resources when simulation resets
  useEffect(() => {
    if (state?.currentTime === 0) {
      setResources({
        oxygen: 100,
        water: 100,
        food: 100,
        power: 100,
        temperature: 22,
        pressure: 101.3,
      });
    }
  }, [state?.currentTime]);

  const getResourceStatus = (value: number) => {
    if (value > 50) return "good";
    if (value > 20) return "warning";
    return "critical";
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "good":
        return "text-accent";
      case "warning":
        return "text-warning";
      case "critical":
        return "text-destructive";
      default:
        return "text-muted-foreground";
    }
  };

  const resourceItems = [
    {
      key: "oxygen",
      label: "Oxygen",
      value: resources.oxygen,
      unit: "%",
      icon: Wind,
      critical: resources.oxygen < 20,
    },
    {
      key: "water",
      label: "Water",
      value: resources.water,
      unit: "%",
      icon: Droplets,
      critical: resources.water < 15,
    },
    {
      key: "food",
      label: "Food",
      value: resources.food,
      unit: "%",
      icon: Heart,
      critical: resources.food < 10,
    },
    {
      key: "power",
      label: "Power",
      value: resources.power,
      unit: "%",
      icon: Zap,
      critical: resources.power < 20,
    },
  ];

  return (
    <Card className="h-full bg-card border-border flex flex-col">
      <div className="p-3 border-b border-border shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-foreground">CREW RESOURCES</h2>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="font-mono-mission text-xs">
              {state ? `T+${state.currentTime.toFixed(1)}h` : "STANDBY"}
            </Badge>
            <div
              className={`w-2 h-2 rounded-full ${
                resources.oxygen > 50 &&
                resources.water > 50 &&
                resources.food > 50 &&
                resources.power > 20
                  ? "bg-accent glow-success"
                  : "bg-warning animate-pulse"
              }`}
            />
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="p-1 text-muted-foreground hover:text-foreground transition-colors"
              title={isCollapsed ? "Expand" : "Collapse"}
            >
              {isCollapsed ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronUp className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </div>

      {!isCollapsed ? (
        <div className="flex-1 p-3 space-y-3 overflow-y-auto">
          {/* Resource Grid */}
          <div className="grid grid-cols-2 gap-2">
            {resourceItems.map(
              ({ key, label, value, unit, icon: IconComponent, critical }) => {
                const status = getResourceStatus(value);
                return (
                  <div
                    key={key}
                    className={`bg-muted/50 rounded p-2 border transition-all ${
                      critical
                        ? "border-destructive glow-hazard animate-pulse"
                        : status === "warning"
                        ? "border-warning"
                        : "border-border"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <IconComponent
                        className={`w-3 h-3 ${getStatusColor(status)}`}
                      />
                      <span
                        className={`text-xs font-mono-mission font-bold ${getStatusColor(
                          status
                        )}`}
                      >
                        {value.toFixed(1)}
                        {unit}
                      </span>
                    </div>
                    <p className="text-xs font-mono-mission text-muted-foreground truncate mb-1">
                      {label}
                    </p>
                    <Progress value={value} className="h-1" />
                  </div>
                );
              }
            )}
          </div>

          {/* Environmental Status */}
          <div className="space-y-2">
            <label className="text-xs font-mono-mission text-muted-foreground">
              ENVIRONMENT
            </label>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-muted/30 rounded p-2">
                <p className="text-xs font-mono-mission text-muted-foreground mb-1">
                  Temperature
                </p>
                <p className="text-xs font-mono-mission text-primary">
                  {resources.temperature.toFixed(1)}°C
                </p>
              </div>
              <div className="bg-muted/30 rounded p-2">
                <p className="text-xs font-mono-mission text-muted-foreground mb-1">
                  Pressure
                </p>
                <p className="text-xs font-mono-mission text-primary">
                  {resources.pressure.toFixed(1)} kPa
                </p>
              </div>
            </div>
          </div>

          {/* Critical Alerts */}
          {resourceItems.some((item) => item.critical) && (
            <div className="p-2 bg-destructive/10 border border-destructive rounded">
              <div className="flex items-center space-x-2 mb-1">
                <div className="w-2 h-2 bg-destructive rounded-full animate-pulse" />
                <span className="text-xs font-mono-mission text-destructive font-bold">
                  RESOURCE CRITICAL
                </span>
              </div>
              <p className="text-xs text-muted-foreground font-mono-mission">
                {resourceItems
                  .filter((item) => item.critical)
                  .map((item) => item.label.toLowerCase())
                  .join(", ")}{" "}
                levels critical
              </p>
            </div>
          )}

          {/* Resource Management Tips */}
          <div className="p-2 bg-primary/10 border border-primary rounded">
            <div className="flex items-center space-x-2 mb-1">
              <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
              <span className="text-xs font-mono-mission text-primary font-bold">
                RESOURCE TIP
              </span>
            </div>
            <p className="text-xs text-foreground font-mono-mission">
              {resources.power < 30
                ? "Consider reducing non-essential systems to conserve power."
                : resources.oxygen < 40
                ? "Monitor atmospheric recycling systems closely."
                : resources.water < 50
                ? "Water reclamation at 98% efficiency. Usage within limits."
                : "All resource levels nominal. Continue current consumption rates."}
            </p>
          </div>
        </div>
      ) : (
        <div className="p-3">
          <div className="grid grid-cols-4 gap-2 text-xs">
            {resourceItems.map(({ key, label, value, critical }) => (
              <div
                key={key}
                className={`text-center ${
                  critical
                    ? "text-destructive"
                    : value < 50
                    ? "text-warning"
                    : "text-primary"
                }`}
              >
                <div className="font-mono-mission font-bold">
                  {Math.round(value)}%
                </div>
                <div className="text-muted-foreground">{label}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
