import { useState, useEffect, useRef } from "react";
import { Terminal, Info, AlertTriangle, AlertCircle } from "lucide-react";
import { missionLogs, type MissionLog } from "@/data/missionData";
import { Card } from "@/components/ui/card";

const priorityIcons = {
  Info: Info,
  Warning: AlertTriangle,
  Critical: AlertCircle,
};

const priorityColors = {
  Info: "text-primary",
  Warning: "text-warning",
  Critical: "text-destructive",
};

const sourceColors = {
  "ODIN-AI": "text-primary",
  "Flight Controller": "text-accent",
  Navigation: "text-warning",
  "Hazard Detection": "text-destructive",
};

interface MissionLogsProps {
  logs?: MissionLog[];
}

export default function MissionLogs({ logs: propLogs = [] }: MissionLogsProps) {
  const [logs, setLogs] = useState<MissionLog[]>([...missionLogs, ...propLogs]);
  const [autoScroll, setAutoScroll] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Replace all logs when prop changes
    setLogs([...missionLogs, ...propLogs]);
  }, [propLogs]);

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "nearest",
      });
    }
  }, [logs, autoScroll]);

  const handleScroll = () => {
    if (containerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
      const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10;
      setAutoScroll(isAtBottom);
    }
  };

  const clearLogs = () => {
    setLogs([]);
  };

  const generateAILog = () => {
    const aiMessages = [
      "Analyzing optimal trajectory parameters for fuel efficiency.",
      "Radiation shielding calculations updated based on solar activity.",
      "Trajectory optimization complete. Recommending minor course correction.",
      "Monitoring debris field. All clear for current flight path.",
      "Fuel consumption within nominal parameters. Mission timeline on track.",
      "AI co-pilot systems running optimal route calculations.",
      "Space weather analysis indicates favorable conditions for next 48 hours.",
    ];

    const newLog: MissionLog = {
      id: `ai-${Date.now()}`,
      timestamp: new Date().toTimeString().split(" ")[0],
      source: "ODIN-AI",
      message: aiMessages[Math.floor(Math.random() * aiMessages.length)],
      priority: "Info",
    };

    setLogs((prev) => [...prev, newLog]);
  };

  return (
    <Card className="h-full bg-card border-border">
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Terminal className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-bold text-foreground">
              AI CO-PILOT LOGS
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={generateAILog}
              className="px-3 py-1 bg-primary text-primary-foreground rounded text-xs font-mono-mission hover:glow-primary transition-all"
            >
              GENERATE AI LOG
            </button>
            <button
              onClick={clearLogs}
              className="px-3 py-1 bg-muted text-muted-foreground rounded text-xs font-mono-mission hover:bg-secondary hover:text-secondary-foreground transition-colors"
            >
              CLEAR
            </button>
          </div>
        </div>
      </div>

      <div
        ref={containerRef}
        className="p-4 h-[calc(100%-80px)] overflow-y-auto space-y-2"
        onScroll={handleScroll}
      >
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Terminal className="w-16 h-16 text-muted-foreground mb-4" />
            <p className="text-muted-foreground font-mono-mission">
              No mission logs available
            </p>
          </div>
        ) : (
          logs.map((log) => {
            const IconComponent = priorityIcons[log.priority];
            return (
              <div
                key={log.id}
                className={`border-l-2 pl-3 py-2 ${
                  log.priority === "Critical"
                    ? "border-l-destructive bg-destructive/5"
                    : log.priority === "Warning"
                    ? "border-l-warning bg-warning/5"
                    : "border-l-primary bg-primary/5"
                } rounded-r transition-all hover:bg-opacity-80`}
              >
                <div className="flex items-start justify-between mb-1">
                  <div className="flex items-center space-x-2">
                    <IconComponent
                      className={`w-3 h-3 ${priorityColors[log.priority]}`}
                    />
                    <span
                      className={`text-xs font-mono-mission font-bold ${
                        sourceColors[log.source]
                      }`}
                    >
                      {log.source}
                    </span>
                  </div>
                  <span className="text-xs font-mono-mission text-muted-foreground">
                    {log.timestamp}
                  </span>
                </div>

                <p className="text-sm text-foreground leading-relaxed font-mono-mission">
                  {log.message}
                </p>
              </div>
            );
          })
        )}
        <div ref={logsEndRef} />
      </div>

      {!autoScroll && (
        <div className="absolute bottom-4 right-4">
          <button
            onClick={() => {
              setAutoScroll(true);
              logsEndRef.current?.scrollIntoView({
                behavior: "smooth",
                block: "nearest",
                inline: "nearest",
              });
            }}
            className="bg-primary text-primary-foreground rounded-full p-2 text-xs glow-primary"
          >
            ↓
          </button>
        </div>
      )}
    </Card>
  );
}
