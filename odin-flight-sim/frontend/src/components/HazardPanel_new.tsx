import { useState, useEffect, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AlertTriangle,
  Zap,
  RadioIcon as Radio,
  Atom,
  Shield,
  Sun,
  Orbit,
} from "lucide-react";
import { useDynamicMissionData } from "@/data/useDynamicMissionData";

// Interface for hazard data
interface Hazard {
  id: string;
  type: string;
  severity: "Low" | "Medium" | "High" | "Critical";
  description: string;
  timestamp: string;
  coordinates?: { lat: number; lon: number };
}

interface HazardPanelProps {
  onHazardInject?: (hazard: Hazard) => void;
}

const hazardIcons: Record<string, any> = {
  "Solar Flare": Sun,
  CME: Orbit,
  "Radiation Storm": Atom,
  "Communication Disruption": Radio,
  "Navigation Error": AlertTriangle,
  "System Failure": Zap,
  "Magnetic Anomaly": Shield,
};

const severityColors: Record<string, string> = {
  Critical: "bg-destructive text-destructive-foreground",
  High: "bg-orange-500 text-white",
  Medium: "bg-warning text-warning-foreground",
  Low: "bg-accent text-accent-foreground",
};

export default function HazardPanel({ onHazardInject }: HazardPanelProps) {
  // Use dynamic mission data hook
  const { hazards, loading, error } = useDynamicMissionData({
    refreshInterval: 15000, // Refresh every 15 seconds for hazards
    enableWebSocket: true,
    predictionHorizon: 72,
  });

  const [localActiveHazards, setLocalActiveHazards] = useState<Hazard[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(true);

  useEffect(() => {
    if (!isMonitoring || !hazards) return;

    const interval = setInterval(() => {
      // Simulate new hazard detection
      if (Math.random() < 0.1 && localActiveHazards.length < 3) {
        const availableHazards = hazards.filter(
          (h) => !localActiveHazards.some((ah) => ah.id === h.id)
        );

        if (availableHazards.length > 0) {
          const newHazard =
            availableHazards[
              Math.floor(Math.random() * availableHazards.length)
            ];
          setLocalActiveHazards((prev) => [...prev, newHazard]);
          onHazardInject?.(newHazard);
        }
      }
    }, 10000); // Reduced frequency

    return () => clearInterval(interval);
  }, [isMonitoring, hazards, onHazardInject, localActiveHazards]);

  const handleDismissHazard = (hazardId: string) => {
    setLocalActiveHazards((prev) => prev.filter((h) => h.id !== hazardId));
  };

  const injectRandomHazard = useCallback(() => {
    if (!hazards) return;

    const availableHazards = hazards.filter(
      (h) => !localActiveHazards.some((ah) => ah.id === h.id)
    );

    if (availableHazards.length > 0) {
      const randomHazard =
        availableHazards[Math.floor(Math.random() * availableHazards.length)];
      setLocalActiveHazards((prev) => [...prev, randomHazard]);
      onHazardInject?.(randomHazard);
    }
  }, [hazards, localActiveHazards, onHazardInject]);

  // Handle external hazard injection events
  useEffect(() => {
    const handleInjectEvent = () => {
      injectRandomHazard();
    };

    window.addEventListener("inject-hazard", handleInjectEvent);
    return () => window.removeEventListener("inject-hazard", handleInjectEvent);
  }, [injectRandomHazard]);

  // Show loading state
  if (loading) {
    return (
      <Card className="h-full bg-card border-border">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-bold text-foreground">
            HAZARD DETECTION
          </h2>
        </div>
        <div className="p-4 flex items-center justify-center h-[calc(100%-80px)]">
          <p className="text-muted-foreground font-mono-mission">
            Loading hazard data...
          </p>
        </div>
      </Card>
    );
  }

  // Show error state
  if (error) {
    return (
      <Card className="h-full bg-card border-border">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-bold text-foreground">
            HAZARD DETECTION
          </h2>
        </div>
        <div className="p-4 flex items-center justify-center h-[calc(100%-80px)]">
          <p className="text-destructive font-mono-mission">
            Error loading hazard data: {error}
          </p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="h-full bg-card border-border">
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isMonitoring
                  ? "bg-accent glow-success animate-pulse"
                  : "bg-muted"
              }`}
            />
            <h2 className="text-lg font-bold text-foreground">
              HAZARD DETECTION
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsMonitoring(!isMonitoring)}
              className={`px-3 py-1 rounded text-xs font-mono-mission transition-colors ${
                isMonitoring
                  ? "bg-accent text-accent-foreground glow-success"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              {isMonitoring ? "MONITORING" : "OFFLINE"}
            </button>
            <button
              onClick={injectRandomHazard}
              className="px-3 py-1 bg-warning text-warning-foreground rounded text-xs font-mono-mission hover:glow-hazard transition-all"
            >
              INJECT HAZARD
            </button>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-3 h-[calc(100%-80px)] overflow-y-auto">
        {localActiveHazards.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center mb-4">
              <Sun className="w-8 h-8 text-accent" />
            </div>
            <p className="text-muted-foreground font-mono-mission">
              All systems nominal
              <br />
              No hazards detected
            </p>
          </div>
        ) : (
          localActiveHazards.map((hazard) => {
            const IconComponent = hazardIcons[hazard.type];
            return (
              <div
                key={hazard.id}
                className={`border rounded-lg p-3 transition-all ${
                  hazard.severity === "Critical"
                    ? "border-destructive glow-hazard"
                    : hazard.severity === "High"
                    ? "border-destructive"
                    : hazard.severity === "Medium"
                    ? "border-warning"
                    : "border-accent"
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <IconComponent
                      className={`w-4 h-4 ${
                        hazard.severity === "Critical"
                          ? "text-destructive"
                          : hazard.severity === "High"
                          ? "text-destructive"
                          : hazard.severity === "Medium"
                          ? "text-warning"
                          : "text-accent"
                      }`}
                    />
                    <span className="font-mono-mission text-sm font-bold text-foreground">
                      {hazard.type}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge className={severityColors[hazard.severity]}>
                      {hazard.severity}
                    </Badge>
                    <button
                      onClick={() => handleDismissHazard(hazard.id)}
                      className="text-muted-foreground hover:text-foreground text-xs"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                <p className="text-xs text-muted-foreground mb-2 font-mono-mission">
                  {hazard.timestamp}
                </p>

                <p className="text-sm text-foreground leading-relaxed">
                  {hazard.description}
                </p>

                {hazard.coordinates && (
                  <div className="mt-2 pt-2 border-t border-border">
                    <p className="text-xs font-mono-mission text-muted-foreground">
                      Coordinates: {hazard.coordinates.lat}°,{" "}
                      {hazard.coordinates.lon}°
                    </p>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </Card>
  );
}
