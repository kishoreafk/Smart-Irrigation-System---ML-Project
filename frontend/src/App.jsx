import React, { useState, useEffect, Suspense, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { motion } from 'framer-motion';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { Container, Grid, Paper, Typography, Box, CircularProgress, Alert } from '@mui/material';
import WeatherVisualization from './components/WeatherVisualization';
import DataPanel from './components/DataPanel';
import gsap from 'gsap';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#64b5f6',
    },
    secondary: {
      main: '#81c784',
    },
    background: {
      default: 'transparent',
      paper: 'rgba(255, 255, 255, 0.1)',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const titleRef = useRef();

  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/predict');
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 300000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (titleRef.current && data) {
      gsap.fromTo(titleRef.current,
        { scale: 0.8, opacity: 0 },
        { scale: 1, opacity: 1, duration: 1.5, ease: 'elastic.out(1, 0.5)' }
      );
    }
  }, [data]);

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="100vh"
          flexDirection="column"
          gap={2}
        >
          <CircularProgress size={60} />
          <Typography variant="h6">Loading weather data and predictions...</Typography>
        </Box>
      </ThemeProvider>
    );
  }

  if (error) {
    return (
      <ThemeProvider theme={theme}>
        <Container maxWidth="md" sx={{ mt: 4 }}>
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
          <Box display="flex" justifyContent="center">
            <motion.button
              onClick={fetchData}
              style={{
                padding: '12px 24px',
                background: 'linear-gradient(45deg, #64b5f6 30%, #81c784 90%)',
                border: 'none',
                borderRadius: '8px',
                color: 'white',
                fontSize: '16px',
                cursor: 'pointer',
                boxShadow: '0 3px 5px 2px rgba(100, 181, 246, .3)',
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Retry
            </motion.button>
          </Box>
        </Container>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ 
        minHeight: '100vh', 
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <Box sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3), transparent 50%), radial-gradient(circle at 80% 80%, rgba(72, 52, 212, 0.3), transparent 50%)',
          pointerEvents: 'none'
        }} />
        <Container maxWidth="xl" sx={{ py: 4, position: 'relative', zIndex: 1 }}>
          <motion.div
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Typography
              ref={titleRef}
              variant="h3"
              component="h1"
              gutterBottom
              align="center"
              sx={{
                color: 'white',
                fontWeight: 'bold',
                textShadow: '0 0 20px rgba(100, 181, 246, 0.5), 2px 2px 4px rgba(0,0,0,0.3)',
                mb: 4,
                background: 'linear-gradient(45deg, #64b5f6, #81c784, #ffb74d)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}
            >
              🌤️ SMART IRRIGATION WITH ML
            </Typography>
          </motion.div>

          <Grid container spacing={3}>
            {/* 3D Visualization */}
            <Grid item xs={12} md={8}>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              >
                <Paper
                  elevation={3}
                  sx={{
                    height: '600px',
                    background: 'rgba(15, 15, 35, 0.6)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid rgba(100, 181, 246, 0.3)',
                    borderRadius: '24px',
                    overflow: 'hidden',
                    boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37), inset 0 0 20px rgba(100, 181, 246, 0.1)'
                  }}
                >
                  <Canvas camera={{ position: [0, 0, 15], fov: 60 }}>
                    <Suspense fallback={null}>
                      <ambientLight intensity={0.3} />
                      <pointLight position={[10, 10, 10]} intensity={1} />
                      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#64b5f6" />
                      <spotLight position={[0, 10, 0]} angle={0.3} penumbra={1} intensity={0.5} color="#81c784" />
                      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={0.5} />
                      <WeatherVisualization data={data} />
                      <OrbitControls 
                        enablePan={false} 
                        enableZoom={true} 
                        enableRotate={true}
                        autoRotate
                        autoRotateSpeed={0.5}
                        minDistance={10}
                        maxDistance={25}
                      />
                    </Suspense>
                  </Canvas>
                </Paper>
              </motion.div>
            </Grid>

            {/* Data Panel */}
            <Grid item xs={12} md={4}>
              <DataPanel data={data} onRefresh={fetchData} />
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
