import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useSimulation } from "@/hooks/useSimulation";

interface DataPoint {
  time: number;
  velocity: number;
  fuel: number;
  radiation: number;
  distance: number;
}

export default function LiveChartsPanel() {
  const { state, metrics, isRunning } = useSimulation();
  const [chartData, setChartData] = useState<DataPoint[]>([]);
  const [maxPoints] = useState(50); // Keep last 50 data points

  useEffect(() => {
    if (state && metrics && isRunning) {
      const newDataPoint: DataPoint = {
        time: state.currentTime,
        velocity: metrics.currentVelocity || 0,
        fuel: state.fuelRemaining,
        radiation: metrics.radiationExposure,
        distance: metrics.distanceToMoon / 1000, // Convert to thousands of km
      };

      setChartData((prev) => {
        const updated = [...prev, newDataPoint];
        // Keep only the last maxPoints
        return updated.slice(-maxPoints);
      });
    }
  }, [state, metrics, isRunning, maxPoints]);

  // Clear data when simulation resets
  useEffect(() => {
    if (state?.currentTime === 0) {
      setChartData([]);
    }
  }, [state?.currentTime]);

  const formatTooltipValue = (value: number, name: string) => {
    switch (name) {
      case "velocity":
        return [`${value.toFixed(1)} km/s`, "Velocity"];
      case "fuel":
        return [`${value.toFixed(1)}%`, "Fuel Remaining"];
      case "radiation":
        return [`${value.toFixed(1)}%`, "Radiation Exposure"];
      case "distance":
        return [`${(value * 1000).toLocaleString()} km`, "Distance to Moon"];
      default:
        return [value, name];
    }
  };

  const CustomTooltip = ({
    active,
    payload,
    label,
  }: {
    active?: boolean;
    payload?: Array<{
      color: string;
      dataKey: string;
      value: number;
    }>;
    label?: number;
  }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-lg p-2 shadow-lg">
          <p className="text-xs font-mono-mission text-muted-foreground">
            Time: {label?.toFixed(1)}h
          </p>
          {payload.map((entry, index: number) => (
            <p
              key={index}
              className="text-xs font-mono-mission"
              style={{ color: entry.color }}
            >
              {formatTooltipValue(entry.value, entry.dataKey)[1]}:{" "}
              {formatTooltipValue(entry.value, entry.dataKey)[0]}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Velocity and Distance Chart */}
      <Card className="bg-card/90 backdrop-blur-sm border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono-mission text-primary flex items-center justify-between">
            VELOCITY & DISTANCE
            <Badge variant="outline" className="text-xs">
              {chartData.length} points
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
              />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                axisLine={{ stroke: "hsl(var(--border))" }}
              />
              <YAxis
                yAxisId="left"
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                axisLine={{ stroke: "hsl(var(--border))" }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                axisLine={{ stroke: "hsl(var(--border))" }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="velocity"
                stroke="hsl(var(--primary))"
                strokeWidth={2}
                dot={false}
                name="Velocity"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="distance"
                stroke="hsl(var(--accent))"
                strokeWidth={2}
                dot={false}
                name="Distance"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Fuel and Radiation Chart */}
      <Card className="bg-card/90 backdrop-blur-sm border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-mono-mission text-primary flex items-center justify-between">
            FUEL & RADIATION
            <Badge variant="outline" className="text-xs">
              Live Data
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
              />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                axisLine={{ stroke: "hsl(var(--border))" }}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                axisLine={{ stroke: "hsl(var(--border))" }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="fuel"
                stackId="1"
                stroke="hsl(var(--accent))"
                fill="hsl(var(--accent))"
                fillOpacity={0.3}
                name="Fuel"
              />
              <Area
                type="monotone"
                dataKey="radiation"
                stackId="2"
                stroke="hsl(var(--warning))"
                fill="hsl(var(--warning))"
                fillOpacity={0.3}
                name="Radiation"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
