import { Canvas } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { trajectories } from "@/data/missionData";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { LiveMetrics } from "@/lib/simulationEngine";

interface DataPoint {
  x: number;
  y: number;
  z: number;
  color: string;
  label: string;
}

function Sphere3D({
  position,
  color,
  label,
  size = 0.2,
}: {
  position: [number, number, number];
  color: string;
  label: string;
  size?: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.5;
    }
  });

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[size]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.2}
        />
      </mesh>
      <Text
        position={[0, size + 0.3, 0]}
        fontSize={0.15}
        color="#00FFFF"
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>
    </group>
  );
}

function GridLines() {
  const gridPoints = useMemo(() => {
    const points = [];

    // X-axis grid lines
    for (let i = -5; i <= 5; i++) {
      points.push([
        [i, -5, 0],
        [i, 5, 0],
      ]);
    }

    // Y-axis grid lines
    for (let i = -5; i <= 5; i++) {
      points.push([
        [-5, i, 0],
        [5, i, 0],
      ]);
    }

    return points;
  }, []);

  return (
    <>
      {gridPoints.map((points, index) => (
        <Line
          key={index}
          points={points}
          color="#333333"
          lineWidth={1}
          transparent
          opacity={0.3}
        />
      ))}
    </>
  );
}

function AxisLabels() {
  return (
    <>
      <Text
        position={[6, 0, 0]}
        fontSize={0.2}
        color="#FF4500"
        anchorX="center"
        anchorY="middle"
      >
        ΔV (km/s)
      </Text>
      <Text
        position={[0, 6, 0]}
        fontSize={0.2}
        color="#32CD32"
        anchorX="center"
        anchorY="middle"
      >
        Time (hours)
      </Text>
      <Text
        position={[0, 0, 6]}
        fontSize={0.2}
        color="#FFD700"
        anchorX="center"
        anchorY="middle"
      >
        Radiation (%)
      </Text>
    </>
  );
}

interface SimulationGraph3DProps {
  activeTrajectory?: string;
  metrics?: LiveMetrics | null;
}

export default function SimulationGraph3D({
  activeTrajectory = "baseline",
  metrics,
}: SimulationGraph3DProps) {
  // Transform trajectory data into 3D coordinates
  const dataPoints: DataPoint[] = useMemo(() => {
    return trajectories.map((traj) => {
      const x = (traj.deltaV - 3000) / 200; // Normalize deltaV to -5 to 5 range
      const y = (traj.travelTime - 60) / 10; // Normalize time to -5 to 5 range
      const z = (traj.radiationExposure - 50) / 20; // Normalize radiation to -5 to 5 range

      const color =
        traj.risk === "Low"
          ? "#00FF00"
          : traj.risk === "High"
          ? "#FF4500"
          : "#FFFF00";

      return {
        x: Math.max(-5, Math.min(5, x)),
        y: Math.max(-5, Math.min(5, y)),
        z: Math.max(-5, Math.min(5, z)),
        color,
        label: traj.name.split(" ").slice(0, 2).join(" "),
      };
    });
  }, []);

  // Connect trajectories with lines
  const trajectoryLines = useMemo(() => {
    const lines = [];
    for (let i = 0; i < dataPoints.length - 1; i++) {
      const p1 = dataPoints[i];
      const p2 = dataPoints[i + 1];
      lines.push([
        [p1.x, p1.y, p1.z],
        [p2.x, p2.y, p2.z],
      ]);
    }
    return lines;
  }, [dataPoints]);

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-sm font-mono-mission text-primary">
          3D TRAJECTORY ANALYSIS
        </CardTitle>
      </CardHeader>
      <CardContent className="h-[calc(100%-60px)] p-0">
        <div className="w-full h-full bg-gradient-cosmic relative">
          <Canvas camera={{ position: [8, 8, 8], fov: 50 }}>
            {/* Lighting */}
            <ambientLight intensity={0.4} />
            <pointLight position={[10, 10, 5]} intensity={1} color="#FFFFFF" />
            <directionalLight position={[-10, -10, -5]} intensity={0.5} />

            {/* Grid and axes */}
            <GridLines />
            <AxisLabels />

            {/* Data points */}
            {dataPoints.map((point, index) => (
              <Sphere3D
                key={index}
                position={[point.x, point.y, point.z]}
                color={point.color}
                label={point.label}
                size={trajectories[index].id === activeTrajectory ? 0.3 : 0.2}
              />
            ))}

            {/* Connection lines between trajectories */}
            {trajectoryLines.map((points, index) => (
              <Line
                key={index}
                points={points}
                color="#00FFFF"
                lineWidth={2}
                transparent
                opacity={0.6}
              />
            ))}

            {/* Controls */}
            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              minDistance={5}
              maxDistance={20}
            />
          </Canvas>

          {/* Legend overlay */}
          <div className="absolute top-4 right-4 bg-card/90 backdrop-blur-sm border border-border rounded-lg p-3">
            <h4 className="text-xs font-mono-mission text-primary mb-2">
              LEGEND
            </h4>
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-accent" />
                <span className="text-xs text-muted-foreground">Low Risk</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-warning" />
                <span className="text-xs text-muted-foreground">
                  Medium Risk
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-destructive" />
                <span className="text-xs text-muted-foreground">High Risk</span>
              </div>
            </div>
          </div>

          {/* Metrics display */}
          <div className="absolute bottom-4 left-4 bg-card/90 backdrop-blur-sm border border-border rounded-lg p-3">
            <h4 className="text-xs font-mono-mission text-primary mb-2">
              ACTIVE TRAJECTORY
            </h4>
            {(() => {
              const active = trajectories.find(
                (t) => t.id === activeTrajectory
              );
              const deltaV = metrics?.deltaV || active?.deltaV;
              const travelTime = metrics?.travelTime || active?.travelTime;
              const radiationExposure =
                metrics?.radiationExposure || active?.radiationExposure;

              return active ? (
                <div className="space-y-1">
                  <div className="text-xs text-muted-foreground">
                    <span className="text-warning">ΔV:</span> {deltaV} m/s
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <span className="text-accent">Time:</span> {travelTime}h
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <span className="text-destructive">Radiation:</span>{" "}
                    {radiationExposure?.toFixed(1)}%
                  </div>
                  {metrics?.hazardLevel &&
                    metrics.hazardLevel !== "Nominal" && (
                      <div className="text-xs text-muted-foreground">
                        <span className="text-warning">Status:</span>{" "}
                        {metrics.hazardLevel}
                      </div>
                    )}
                </div>
              ) : null;
            })()}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
