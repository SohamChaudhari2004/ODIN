import { useState } from "react";
import { Play, Pause, FastForward, Rewind } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { useSimulation } from "@/hooks/useSimulation";

export default function TimeControlPanel() {
  const {
    state,
    isRunning,
    isPaused,
    startSimulation,
    pauseSimulation,
    resumeSimulation,
    resetSimulation,
    setTimeScale,
  } = useSimulation();

  const [timeScale, setTimeScaleLocal] = useState(1);

  const handleTimeScaleChange = (value: number[]) => {
    const scale = value[0];
    setTimeScaleLocal(scale);
    setTimeScale(scale);
  };

  const handlePlayPause = () => {
    if (!isRunning) {
      startSimulation();
    } else if (isPaused) {
      resumeSimulation();
    } else {
      pauseSimulation();
    }
  };

  const presetSpeeds = [0.5, 1, 2, 5, 10];

  return (
    <Card className="bg-card/90 backdrop-blur-sm border-border">
      <div className="p-3">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-mono-mission text-primary font-bold">
            TIME CONTROL
          </h3>
          <Badge variant="outline" className="text-xs font-mono-mission">
            {timeScale}x SPEED
          </Badge>
        </div>

        {/* Main Controls */}
        <div className="flex items-center space-x-2 mb-3">
          <Button
            onClick={handlePlayPause}
            size="sm"
            className={`${
              isRunning && !isPaused
                ? "bg-warning text-warning-foreground"
                : "bg-accent text-accent-foreground hover:glow-success"
            } font-mono-mission`}
          >
            {isRunning && !isPaused ? (
              <Pause className="w-3 h-3" />
            ) : (
              <Play className="w-3 h-3" />
            )}
          </Button>

          <Button
            onClick={resetSimulation}
            size="sm"
            variant="outline"
            className="font-mono-mission border-border hover:bg-muted"
          >
            <Rewind className="w-3 h-3" />
          </Button>
        </div>

        {/* Time Scale Slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono-mission text-muted-foreground">
              Speed
            </span>
            <span className="text-xs font-mono-mission text-primary">
              {timeScale.toFixed(1)}x
            </span>
          </div>

          <Slider
            value={[timeScale]}
            onValueChange={handleTimeScaleChange}
            min={0.1}
            max={10}
            step={0.1}
            className="w-full"
          />

          {/* Preset Speed Buttons */}
          <div className="flex justify-between">
            {presetSpeeds.map((speed) => (
              <Button
                key={speed}
                onClick={() => handleTimeScaleChange([speed])}
                size="sm"
                variant={timeScale === speed ? "default" : "outline"}
                className="text-xs font-mono-mission h-6 px-2"
              >
                {speed}x
              </Button>
            ))}
          </div>
        </div>

        {/* Status Display */}
        {state && (
          <div className="mt-3 p-2 bg-muted/30 rounded">
            <div className="grid grid-cols-2 gap-1 text-xs font-mono-mission">
              <div className="text-muted-foreground">
                Status:{" "}
                <span
                  className={`${
                    isRunning && !isPaused ? "text-accent" : "text-warning"
                  }`}
                >
                  {isRunning && !isPaused ? "RUNNING" : "PAUSED"}
                </span>
              </div>
              <div className="text-muted-foreground">
                Phase:{" "}
                <span className="text-primary">{state.currentPhase}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
