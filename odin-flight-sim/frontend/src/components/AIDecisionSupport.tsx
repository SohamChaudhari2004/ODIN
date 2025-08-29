import { useState, useEffect, useCallback } from "react";
import {
  Brain,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useSimulation } from "@/hooks/useSimulation";
import { useOdin } from "@/hooks/useOdin";

interface AIRecommendation {
  id: string;
  type: "trajectory" | "hazard" | "resource" | "timing";
  priority: "low" | "medium" | "high" | "critical";
  title: string;
  description: string;
  action?: string;
  confidence: number; // 0-100
  timestamp: number;
}

export default function AIDecisionSupport() {
  const {
    state,
    metrics,
    isRunning,
    currentTrajectory,
    setTrajectory,
    addMissionLog,
  } = useSimulation();

  // ODIN AI Integration
  const {
    isConnected: odinConnected,
    aiRecommendations,
    missionStatus,
    getAIRecommendations,
    explainDecision,
    lastDecisionExplanation,
  } = useOdin();

  const [recommendations, setRecommendations] = useState<AIRecommendation[]>(
    []
  );
  const [aiThinking, setAiThinking] = useState(false);
  const [selectedDecision, setSelectedDecision] = useState<string | null>(null);

  // Fetch ODIN AI recommendations
  const fetchOdinRecommendations = useCallback(async () => {
    if (!odinConnected || !missionStatus?.mission_id) return;

    try {
      setAiThinking(true);
      await getAIRecommendations(missionStatus.mission_id);
    } catch (error) {
      console.error("Failed to fetch ODIN recommendations:", error);
    } finally {
      setAiThinking(false);
    }
  }, [odinConnected, missionStatus?.mission_id, getAIRecommendations]);

  // Convert ODIN recommendations to UI format
  useEffect(() => {
    if (aiRecommendations.length > 0) {
      const convertedRecommendations: AIRecommendation[] =
        aiRecommendations.map((rec, index) => ({
          id: `odin-rec-${index}`,
          type: "trajectory",
          priority: "high",
          title: "ODIN AI Recommendation",
          description: rec,
          confidence: 90,
          timestamp: Date.now(),
        }));
      setRecommendations((prev) => [
        ...convertedRecommendations,
        ...prev.slice(0, 5),
      ]);
    }
  }, [aiRecommendations]);

  // Generate AI recommendations based on current state
  useEffect(() => {
    if (!state || !metrics || !isRunning) return;

    const generateRecommendations = () => {
      const newRecommendations: AIRecommendation[] = [];

      // Fetch ODIN recommendations if connected
      if (odinConnected && missionStatus) {
        fetchOdinRecommendations();
      }

      // Hazard-based recommendations
      if (metrics.hazardLevel === "Critical") {
        newRecommendations.push({
          id: `rec-hazard-${Date.now()}`,
          type: "hazard",
          priority: "critical",
          title: "Critical Hazard Response",
          description:
            "Multiple hazards detected. Immediate trajectory change recommended.",
          action: "Switch to Safe Route Alpha",
          confidence: 95,
          timestamp: Date.now(),
        });
      } else if (
        metrics.hazardLevel === "Warning" &&
        currentTrajectory === "baseline"
      ) {
        newRecommendations.push({
          id: `rec-safety-${Date.now()}`,
          type: "trajectory",
          priority: "medium",
          title: "Safety Optimization",
          description:
            "Current trajectory has elevated risk. Consider safer alternative.",
          action: "Switch to Safe Route Alpha",
          confidence: 78,
          timestamp: Date.now(),
        });
      }

      // Fuel efficiency recommendations
      if (state.fuelRemaining < 60 && currentTrajectory !== "fuel-efficient") {
        newRecommendations.push({
          id: `rec-fuel-${Date.now()}`,
          type: "resource",
          priority: "high",
          title: "Fuel Conservation",
          description:
            "Fuel consumption higher than optimal. Switch to fuel-efficient trajectory.",
          action: "Switch to Fuel Efficient Route",
          confidence: 85,
          timestamp: Date.now(),
        });
      }

      // Radiation exposure recommendations
      if (metrics.radiationExposure > 80) {
        newRecommendations.push({
          id: `rec-radiation-${Date.now()}`,
          type: "hazard",
          priority: "high",
          title: "Radiation Mitigation",
          description:
            "Radiation exposure approaching dangerous levels. Route adjustment advised.",
          action: "Implement radiation shielding protocol",
          confidence: 90,
          timestamp: Date.now(),
        });
      }

      // Timing optimization
      if (state.currentTime > 24 && metrics.travelTime > 85) {
        newRecommendations.push({
          id: `rec-time-${Date.now()}`,
          type: "timing",
          priority: "medium",
          title: "Mission Duration Optimization",
          description: "Extended mission duration may impact crew resources.",
          action: "Consider faster trajectory option",
          confidence: 72,
          timestamp: Date.now(),
        });
      }

      // Performance optimization
      if (metrics.deltaV > 3500 && currentTrajectory !== "fuel-efficient") {
        newRecommendations.push({
          id: `rec-performance-${Date.now()}`,
          type: "trajectory",
          priority: "low",
          title: "Delta-V Optimization",
          description:
            "Current trajectory requires high delta-V. More efficient options available.",
          action: "Optimize for fuel efficiency",
          confidence: 68,
          timestamp: Date.now(),
        });
      }

      // Remove old recommendations (keep only last 3)
      setRecommendations((prev) =>
        [
          ...prev.filter((rec) => Date.now() - rec.timestamp < 300000), // 5 minutes
          ...newRecommendations,
        ].slice(-3)
      );
    };

    // Simulate AI thinking time
    setAiThinking(true);
    const thinkingTime = 2000 + Math.random() * 3000; // 2-5 seconds

    const timer = setTimeout(() => {
      generateRecommendations();
      setAiThinking(false);
    }, thinkingTime);

    return () => clearTimeout(timer);
  }, [
    state,
    metrics,
    isRunning,
    currentTrajectory,
    odinConnected,
    missionStatus,
    fetchOdinRecommendations,
  ]);

  const handleAcceptRecommendation = (rec: AIRecommendation) => {
    if (rec.type === "trajectory" || rec.action?.includes("Switch to")) {
      if (rec.action?.includes("Safe Route Alpha")) {
        setTrajectory("safe-route-alpha");
      } else if (rec.action?.includes("Fuel Efficient")) {
        setTrajectory("fuel-efficient");
      }
    }

    addMissionLog({
      source: "ODIN-AI",
      message: `AI recommendation accepted: ${rec.title}`,
      priority:
        rec.priority === "critical"
          ? "Critical"
          : rec.priority === "high"
          ? "Warning"
          : "Info",
    });

    // Remove the accepted recommendation
    setRecommendations((prev) => prev.filter((r) => r.id !== rec.id));
  };

  const handleDismissRecommendation = (recId: string) => {
    setRecommendations((prev) => prev.filter((r) => r.id !== recId));
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case "critical":
        return AlertCircle;
      case "high":
        return TrendingUp;
      case "medium":
        return Clock;
      default:
        return CheckCircle;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "critical":
        return "text-destructive";
      case "high":
        return "text-warning";
      case "medium":
        return "text-primary";
      default:
        return "text-accent";
    }
  };

  return (
    <Card className="h-full bg-card border-border flex flex-col">
      <div className="p-3 border-b border-border shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-primary" />
            <h2 className="text-sm font-bold text-foreground">
              AI DECISION SUPPORT
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <Badge
              variant={odinConnected ? "default" : "destructive"}
              className="font-mono-mission text-xs"
            >
              ODIN {odinConnected ? "CONNECTED" : "OFFLINE"}
            </Badge>
            <div
              className={`w-2 h-2 rounded-full ${
                aiThinking
                  ? "bg-warning animate-pulse"
                  : odinConnected
                  ? "bg-primary animate-pulse-glow"
                  : "bg-destructive"
              }`}
            />
            {lastDecisionExplanation && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setSelectedDecision(lastDecisionExplanation)}
                className="text-xs"
              >
                Explain
              </Button>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 p-3 space-y-3 overflow-y-auto">
        {aiThinking && (
          <div className="flex items-center space-x-2 p-2 bg-primary/10 border border-primary rounded">
            <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
            <span className="text-xs font-mono-mission text-primary">
              AI analyzing mission parameters...
            </span>
          </div>
        )}

        {recommendations.length === 0 && !aiThinking ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Brain className="w-12 h-12 text-muted-foreground mb-3" />
            <p className="text-xs text-muted-foreground font-mono-mission">
              AI monitoring mission status
              <br />
              No recommendations at this time
            </p>
          </div>
        ) : (
          recommendations.map((rec) => {
            const IconComponent = getPriorityIcon(rec.priority);
            return (
              <div
                key={rec.id}
                className={`border rounded-lg p-3 transition-all ${
                  rec.priority === "critical"
                    ? "border-destructive glow-hazard"
                    : rec.priority === "high"
                    ? "border-warning"
                    : rec.priority === "medium"
                    ? "border-primary"
                    : "border-accent"
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <IconComponent
                      className={`w-3 h-3 ${getPriorityColor(rec.priority)}`}
                    />
                    <span className="font-mono-mission text-xs font-bold text-foreground">
                      {rec.title}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge
                      variant="outline"
                      className={`text-xs ${
                        rec.priority === "critical"
                          ? "border-destructive text-destructive"
                          : rec.priority === "high"
                          ? "border-warning text-warning"
                          : rec.priority === "medium"
                          ? "border-primary text-primary"
                          : "border-accent text-accent"
                      }`}
                    >
                      {rec.confidence}% confidence
                    </Badge>
                  </div>
                </div>

                <p className="text-xs text-foreground mb-2 leading-relaxed">
                  {rec.description}
                </p>

                {rec.action && (
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs font-mono-mission text-muted-foreground">
                      Recommended: {rec.action}
                    </span>
                    <div className="flex space-x-1">
                      <Button
                        onClick={() => handleAcceptRecommendation(rec)}
                        size="sm"
                        className="h-6 px-2 text-xs font-mono-mission bg-accent text-accent-foreground hover:glow-success"
                      >
                        ACCEPT
                      </Button>
                      <Button
                        onClick={() => handleDismissRecommendation(rec.id)}
                        size="sm"
                        variant="outline"
                        className="h-6 px-2 text-xs font-mono-mission border-border hover:bg-muted"
                      >
                        DISMISS
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}

        {/* AI Status Display */}
        <div className="mt-auto p-2 bg-muted/30 rounded">
          <div className="flex items-center justify-between text-xs font-mono-mission">
            <span className="text-muted-foreground">AI Status:</span>
            <span className="text-primary">
              {aiThinking ? "Analyzing..." : "Monitoring"}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs font-mono-mission mt-1">
            <span className="text-muted-foreground">Recommendations:</span>
            <span className="text-primary">
              {recommendations.length} active
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
}
