import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, Sphere, Line } from "@react-three/drei";
import { useRef, useState, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { Maximize2 } from "lucide-react";
import * as THREE from "three";
import { trajectories, type Trajectory } from "@/data/missionData";
import type { SimulationState } from "@/lib/simulationEngine";
import { useOdin } from "@/hooks/useOdin";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface EarthProps {
  position: [number, number, number];
}

function Earth({ position }: EarthProps) {
  const earthRef = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.1;
    }
  });

  return (
    <group position={position}>
      {/* Earth */}
      <Sphere ref={earthRef} args={[2]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#4A90E2" />
      </Sphere>

      {/* Earth atmosphere glow */}
      <Sphere args={[2.2]} position={[0, 0, 0]}>
        <meshBasicMaterial color="#87CEEB" transparent opacity={0.3} />
      </Sphere>

      {/* Earth label */}
      <mesh position={[0, -3, 0]}>
        <planeGeometry args={[2, 0.5]} />
        <meshBasicMaterial color="#00ffff" transparent opacity={0.8} />
      </mesh>
    </group>
  );
}

interface MoonProps {
  position: [number, number, number];
}

function Moon({ position }: MoonProps) {
  const moonRef = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (moonRef.current) {
      moonRef.current.rotation.y += delta * 0.05;
    }
  });

  return (
    <group position={position}>
      {/* Moon */}
      <Sphere ref={moonRef} args={[0.8]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#C0C0C0" />
      </Sphere>

      {/* Moon label */}
      <mesh position={[0, -1.5, 0]}>
        <planeGeometry args={[1.5, 0.3]} />
        <meshBasicMaterial color="#00ffff" transparent opacity={0.8} />
      </mesh>
    </group>
  );
}

interface TrajectoryLineProps {
  trajectory: Trajectory;
  color: string;
  isActive: boolean;
}

function TrajectoryLine({ trajectory, color, isActive }: TrajectoryLineProps) {
  const points = trajectory.points.map(
    (point) =>
      [point.x * 0.1, point.y * 0.1, point.z * 0.1] as [number, number, number]
  );

  return (
    <Line
      points={points}
      color={color}
      lineWidth={isActive ? 3 : 2}
      transparent
      opacity={isActive ? 1 : 0.7}
    />
  );
}

interface SpacecraftProps {
  position: [number, number, number];
}

function Spacecraft({ position }: SpacecraftProps) {
  const spacecraftRef = useRef<THREE.Group>(null);

  useFrame((state, delta) => {
    if (spacecraftRef.current) {
      spacecraftRef.current.rotation.z += delta * 0.5;
    }
  });

  return (
    <group ref={spacecraftRef} position={position}>
      {/* Spacecraft body */}
      <mesh>
        <boxGeometry args={[0.3, 0.1, 0.1]} />
        <meshStandardMaterial color="#FFD700" />
      </mesh>

      {/* Solar panels */}
      <mesh position={[0, 0.2, 0]}>
        <boxGeometry args={[0.1, 0.3, 0.05]} />
        <meshStandardMaterial color="#4169E1" />
      </mesh>
      <mesh position={[0, -0.2, 0]}>
        <boxGeometry args={[0.1, 0.3, 0.05]} />
        <meshStandardMaterial color="#4169E1" />
      </mesh>

      {/* Thruster glow */}
      <mesh position={[-0.2, 0, 0]}>
        <sphereGeometry args={[0.05]} />
        <meshBasicMaterial color="#FF4500" transparent opacity={0.8} />
      </mesh>
    </group>
  );
}

interface TrajectoryView3DProps {
  activeTrajectory: string;
  onTrajectorySelect?: (trajectoryId: string) => void;
  simulationState?: SimulationState | null;
}

interface TrajectoryScene3DProps {
  activeTrajectory: string;
  onTrajectorySelect?: (trajectoryId: string) => void;
  simulationState?: SimulationState | null;
  showControls?: boolean;
  trajectoryData?: Trajectory[];
}

function TrajectoryScene3D({
  activeTrajectory,
  onTrajectorySelect,
  simulationState,
  showControls = true,
  trajectoryData = trajectories,
}: TrajectoryScene3DProps) {
  const [animationProgress, setAnimationProgress] = useState(0);

  const activeTrajectoryData =
    trajectoryData.find((t) => t.id === activeTrajectory) || trajectoryData[0];

  // Use simulation state for spacecraft position if available
  let spacecraftPosition;
  if (simulationState && simulationState.isRunning) {
    spacecraftPosition = simulationState.currentPosition;
  } else {
    const currentPoint =
      activeTrajectoryData.points[
        Math.floor(animationProgress * (activeTrajectoryData.points.length - 1))
      ];
    spacecraftPosition = currentPoint;
  }

  // Calculate actual progress for display
  const progress = simulationState?.isRunning
    ? Math.min(simulationState.currentTime / activeTrajectoryData.travelTime, 1)
    : animationProgress;

  return (
    <>
      <Canvas camera={{ position: [30, 20, 30], fov: 50 }}>
        {/* Lighting */}
        <ambientLight intensity={0.3} />
        <pointLight position={[0, 0, 0]} intensity={2} color="#FFE4B5" />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          color="#FFFFFF"
        />

        {/* Background stars */}
        <Stars
          radius={300}
          depth={60}
          count={3000}
          factor={7}
          saturation={0.8}
          fade
        />

        {/* Earth-Moon system */}
        <Earth position={[0, 0, 0]} />
        <Moon position={[20, 6, 2]} />

        {/* Trajectory lines */}
        {trajectoryData.map((trajectory) => (
          <TrajectoryLine
            key={trajectory.id}
            trajectory={trajectory}
            color={
              trajectory.id === activeTrajectory
                ? "#00FFFF"
                : trajectory.risk === "Low"
                ? "#00FF00"
                : trajectory.risk === "High"
                ? "#FF4500"
                : "#FFFF00"
            }
            isActive={trajectory.id === activeTrajectory}
          />
        ))}

        {/* Spacecraft */}
        <Spacecraft
          position={[
            spacecraftPosition.x * 0.1,
            spacecraftPosition.y * 0.1,
            spacecraftPosition.z * 0.1,
          ]}
        />

        {/* Camera controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxDistance={100}
          minDistance={15}
        />
      </Canvas>

      {/* Overlay controls */}
      {showControls && (
        <>
          {/* 3D View overlay controls */}
          <div className="absolute top-4 left-4 bg-card/80 backdrop-blur-sm border border-border rounded-lg p-3">
            <h3 className="text-sm font-mono-mission text-primary mb-2">
              TRAJECTORY SELECTION
            </h3>
            {trajectoryData.map((trajectory) => (
              <button
                key={trajectory.id}
                onClick={() => onTrajectorySelect?.(trajectory.id)}
                className={`block w-full text-left text-xs py-1 px-2 rounded mb-1 font-mono-mission transition-colors ${
                  trajectory.id === activeTrajectory
                    ? "bg-primary text-primary-foreground glow-primary"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
              >
                {trajectory.name}
              </button>
            ))}
          </div>

          {/* Mission Progress */}

          <div className="absolute bottom-4 left-4 right-32 bg-card/80 backdrop-blur-sm border border-border rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono-mission text-muted-foreground">
                Mission Progress
              </span>
              <span className="text-xs font-mono-mission text-primary">
                {Math.round(progress * 100)}%
              </span>
            </div>
            {simulationState?.isRunning ? (
              <div className="w-full h-1 bg-muted rounded-lg">
                <div
                  className="h-full bg-primary rounded-lg transition-all duration-300"
                  style={{ width: `${progress * 100}%` }}
                />
              </div>
            ) : (
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={animationProgress}
                onChange={(e) =>
                  setAnimationProgress(parseFloat(e.target.value))
                }
                className="w-full h-1 bg-muted rounded-lg appearance-none cursor-pointer"
              />
            )}

            {simulationState && (
              <div className="mt-2 text-xs font-mono-mission text-muted-foreground">
                <div>
                  Time:{" "}
                  <span className="text-primary">
                    {simulationState.currentTime.toFixed(1)}h
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Current Phase */}
          <div className="absolute py-7 bottom-4 right-4 bg-card/80 backdrop-blur-sm border border-border rounded-lg p-3">
            <div className="text-xs font-mono-mission text-muted-foreground mb-1">
              CURRENT PHASE
            </div>
            <div className="text-sm font-mono-mission text-primary">
              {simulationState?.currentPhase || "STANDBY"}
            </div>
          </div>
        </>
      )}
    </>
  );
}

export default function TrajectoryView3D({
  activeTrajectory,
  onTrajectorySelect,
  simulationState,
}: TrajectoryView3DProps) {
  // ODIN integration for real-time trajectory updates
  const {
    currentTrajectory: odinTrajectory,
    availableTrajectories: odinTrajectories,
    isConnected: odinConnected,
    calculateTrajectory,
    getTrajectoryOptions,
  } = useOdin();

  const [trajectoryData, setTrajectoryData] =
    useState<Trajectory[]>(trajectories);

  // Sync ODIN trajectory data with local state
  useEffect(() => {
    if (odinConnected && odinTrajectories.length > 0) {
      // Convert ODIN trajectory format to local format
      const convertedTrajectories: Trajectory[] = odinTrajectories.map(
        (odinTraj, index) => ({
          id: `odin-trajectory-${index}`,
          name: odinTraj.name,
          deltaV: odinTraj.total_delta_v,
          travelTime: odinTraj.total_duration,
          radiationExposure: Math.max(0, 100 - odinTraj.safety_score * 100),
          fuelConsumption: odinTraj.total_delta_v * 0.1, // Estimate based on delta-V
          risk: (odinTraj.safety_score > 0.8
            ? "Low"
            : odinTraj.safety_score > 0.6
            ? "Medium"
            : "High") as "Low" | "Medium" | "High",
          points: odinTraj.waypoints
            ? odinTraj.waypoints.map((wp, wpIndex) => ({
                x: wp.position[0] / 1000, // Convert to km scale
                y: wp.position[1] / 1000,
                z: wp.position[2] / 1000,
                time:
                  wpIndex *
                  (odinTraj.total_duration / odinTraj.waypoints.length),
              }))
            : [
                { x: 0, y: 0, z: 0, time: 0 },
                { x: 100, y: 50, z: 20, time: odinTraj.total_duration * 0.5 },
                { x: 384, y: 0, z: 0, time: odinTraj.total_duration }, // Moon distance
              ],
        })
      );

      // Merge with existing trajectories
      setTrajectoryData([...convertedTrajectories, ...trajectories]);
    }
  }, [odinConnected, odinTrajectories]);

  // Handle trajectory selection
  const handleTrajectorySelect = (trajectoryId: string) => {
    onTrajectorySelect?.(trajectoryId);

    // If it's an ODIN trajectory, sync with backend
    if (trajectoryId.startsWith("odin-trajectory-") && odinConnected) {
      const odinIndex = parseInt(trajectoryId.replace("odin-trajectory-", ""));
      if (odinTrajectories[odinIndex]) {
        console.log("Selected ODIN trajectory:", odinTrajectories[odinIndex]);
      }
    }
  };

  return (
    <div className="w-full h-full bg-gradient-cosmic relative">
      {/* ODIN Status Indicator */}
      {odinConnected && (
        <div className="absolute top-4 left-4 bg-card/80 backdrop-blur-sm border border-border rounded-lg p-2 z-10">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
            <span className="text-xs font-mono-mission text-primary">
              ODIN ACTIVE
            </span>
          </div>
        </div>
      )}

      {/* Main 3D Scene */}
      <div className="absolute inset-0">
        <TrajectoryScene3D
          activeTrajectory={activeTrajectory}
          onTrajectorySelect={handleTrajectorySelect}
          simulationState={simulationState}
          showControls={true}
          trajectoryData={trajectoryData}
        />
      </div>

      {/* Expand Button */}
      <div className="absolute top-4 right-4">
        <Dialog>
          <DialogTrigger asChild>
            <button className="p-2 bg-card/80 backdrop-blur-sm border border-border rounded-lg text-primary hover:text-primary-foreground hover:bg-primary transition-colors">
              <Maximize2 className="w-4 h-4" />
            </button>
          </DialogTrigger>
          <DialogContent className="max-w-[95vw] max-h-[95vh] w-[90vw] h-[90vh] p-0">
            <DialogHeader className="sr-only">
              <DialogTitle>3D Trajectory View - Expanded</DialogTitle>
            </DialogHeader>
            <div className="w-full h-full bg-gradient-cosmic relative rounded-lg overflow-hidden">
              <TrajectoryScene3D
                activeTrajectory={activeTrajectory}
                onTrajectorySelect={onTrajectorySelect}
                simulationState={simulationState}
                showControls={true}
              />
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
