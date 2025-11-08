import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Text } from '@react-three/drei';
import { useSpring, animated } from '@react-spring/three';
import * as THREE from 'three';

export default function DataOrb({ position, value, label, color, maxValue = 100 }) {
  const orbRef = useRef();
  const glowRef = useRef();
  
  const scale = typeof value === 'number' ? (value / maxValue) * 2 + 0.5 : 1;
  
  const { animatedScale } = useSpring({
    animatedScale: scale,
    config: { mass: 1, tension: 280, friction: 60 }
  });

  useFrame((state) => {
    if (orbRef.current) {
      orbRef.current.rotation.y += 0.01;
      orbRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.1;
    }
    if (glowRef.current) {
      glowRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.1);
    }
  });

  return (
    <group position={position}>
      <animated.group ref={orbRef} scale={animatedScale}>
        <Sphere args={[1, 32, 32]}>
          <meshPhysicalMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.5}
            roughness={0.1}
            metalness={0.9}
            clearcoat={1}
            clearcoatRoughness={0.1}
            transmission={0.3}
            thickness={0.5}
          />
        </Sphere>
        
        <Sphere ref={glowRef} args={[1.2, 32, 32]}>
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.2}
            side={THREE.BackSide}
          />
        </Sphere>
      </animated.group>
      
      <Text
        position={[0, scale + 1, 0]}
        fontSize={0.4}
        color={color}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {label}
      </Text>
      
      <Text
        position={[0, scale + 0.3, 0]}
        fontSize={0.6}
        color="white"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {value}
      </Text>
    </group>
  );
}
