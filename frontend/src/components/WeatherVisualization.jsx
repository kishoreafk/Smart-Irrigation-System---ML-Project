import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Environment, Float, Sparkles } from '@react-three/drei';
import ParticleField from './ParticleField';
import DataOrb from './DataOrb';
import FloatingIsland from './FloatingIsland';
import AnimatedRing from './AnimatedRing';

function WeatherVisualization({ data }) {
  const groupRef = useRef();

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.1) * 0.2;
    }
  });

  if (!data?.weather) return null;

  return (
    <>
      <Environment preset="night" />
      <ParticleField count={3000} data={data} />
      
      <group ref={groupRef}>
        <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
          <DataOrb
            position={[-5, 0, 0]}
            value={`${data.weather.temperature.toFixed(1)}°C`}
            label="TEMP"
            color="#ff6b6b"
            maxValue={50}
          />
          <AnimatedRing position={[-5, 0, 0]} color="#ff6b6b" radius={2} />
        </Float>

        <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.8}>
          <DataOrb
            position={[0, 0, 0]}
            value={`${data.weather.humidity}%`}
            label="HUMID"
            color="#74b9ff"
            maxValue={100}
          />
          <AnimatedRing position={[0, 0, 0]} color="#74b9ff" radius={2.2} />
        </Float>

        <Float speed={2.5} rotationIntensity={0.4} floatIntensity={0.6}>
          <DataOrb
            position={[5, 0, 0]}
            value={`${data.weather.wind_speed.toFixed(1)}`}
            label="WIND"
            color="#a29bfe"
            maxValue={20}
          />
          <AnimatedRing position={[5, 0, 0]} color="#a29bfe" radius={2.5} />
        </Float>

        {data.predictions && (
          <>
            <Float speed={1.8} rotationIntensity={0.6} floatIntensity={0.7}>
              <group position={[-3, -4, 2]}>
                <FloatingIsland
                  position={[0, 0, 0]}
                  data={data}
                  type={data.predictions.irrigation_status ? 'temp' : 'humidity'}
                />
                <Text
                  position={[0, 2, 0]}
                  fontSize={0.4}
                  color="#ffffff"
                  anchorX="center"
                  anchorY="middle"
                >
                  {data.predictions.irrigation_status ? 'IRRIGATION' : 'NO IRRIGATION'}
                </Text>
              </group>
            </Float>

            <Float speed={2.2} rotationIntensity={0.5} floatIntensity={0.9}>
              <group position={[3, -4, 2]}>
                <FloatingIsland
                  position={[0, 0, 0]}
                  data={data}
                  type="wind"
                />
                <Text
                  position={[0, 2, 0]}
                  fontSize={0.4}
                  color="#ffffff"
                  anchorX="center"
                  anchorY="middle"
                >
                  WATER: {data.predictions.water_requirement.toFixed(1)}L
                </Text>
              </group>
            </Float>
          </>
        )}

        <Sparkles
          count={200}
          scale={25}
          size={3}
          speed={0.6}
          opacity={0.8}
          color="#64b5f6"
        />

        <Text
          position={[0, 6, 0]}
          fontSize={1.5}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.08}
          outlineColor="#000000"
        >
          CHENNAI
        </Text>
        <Text
          position={[0, 5.2, 0]}
          fontSize={0.4}
          color="#64b5f6"
          anchorX="center"
          anchorY="middle"
        >
          13.0878° N 80.2785° E
        </Text>
      </group>
    </>
  );
}

export default WeatherVisualization;
