import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useSpring, animated } from '@react-spring/three';
import { RoundedBox } from '@react-three/drei';

export default function FloatingIsland({ position, data, type }) {
  const meshRef = useRef();
  
  const { scale } = useSpring({
    scale: data ? 1 : 0,
    config: { mass: 5, tension: 200, friction: 50 }
  });

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime + position[0]) * 0.3;
      meshRef.current.rotation.y += 0.01;
    }
  });

  const getColor = () => {
    switch(type) {
      case 'temp': return '#ff6b6b';
      case 'humidity': return '#74b9ff';
      case 'wind': return '#a29bfe';
      default: return '#00b894';
    }
  };

  return (
    <animated.group scale={scale}>
      <RoundedBox ref={meshRef} args={[2, 2, 2]} position={position} radius={0.2}>
        <meshStandardMaterial
          color={getColor()}
          roughness={0.2}
          metalness={0.8}
          emissive={getColor()}
          emissiveIntensity={0.3}
        />
      </RoundedBox>
    </animated.group>
  );
}
