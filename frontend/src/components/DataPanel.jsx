import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useSpring, animated } from 'react-spring';
import {
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Chip,
  Button,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import {
  WbSunny,
  Cloud,
  Opacity,
  Air,
  Refresh,
  LocationOn,
  TrendingUp,
  Science,
  WaterDrop
} from '@mui/icons-material';
import gsap from 'gsap';

function DataPanel({ data, onRefresh }) {
  const cardRefs = useRef([]);
  
  const springProps = useSpring({
    from: { opacity: 0, transform: 'translateX(100px)' },
    to: { opacity: 1, transform: 'translateX(0px)' },
    config: { tension: 200, friction: 20 }
  });

  useEffect(() => {
    if (cardRefs.current.length > 0) {
      gsap.fromTo(cardRefs.current,
        { scale: 0.8, opacity: 0, y: 20 },
        { scale: 1, opacity: 1, y: 0, duration: 0.6, stagger: 0.1, ease: 'back.out(1.7)' }
      );
    }
  }, [data]);

  if (!data) return null;

  const { weather, predictions, input_features } = data;

  const weatherCards = [
    {
      icon: <WbSunny sx={{ fontSize: 40, color: '#ff6b6b' }} />,
      title: 'Temperature',
      value: `${weather.temperature.toFixed(1)}°C`,
      color: '#ff6b6b'
    },
    {
      icon: <Opacity sx={{ fontSize: 40, color: '#74b9ff' }} />,
      title: 'Humidity',
      value: `${weather.humidity}%`,
      color: '#74b9ff'
    },
    {
      icon: <Air sx={{ fontSize: 40, color: '#a29bfe' }} />,
      title: 'Wind Speed',
      value: `${weather.wind_speed.toFixed(1)} m/s`,
      color: '#a29bfe'
    },
    {
      icon: <Cloud sx={{ fontSize: 40, color: '#636e72' }} />,
      title: 'Pressure',
      value: `${weather.pressure} hPa`,
      color: '#636e72'
    }
  ];

  return (
    <Box sx={{ height: '600px', overflow: 'auto' }}>
      <motion.div
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 3,
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '16px',
            height: '100%'
          }}
        >
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
            <Typography variant="h5" component="h2" sx={{ color: 'white', fontWeight: 'bold' }}>
              📊 Data Dashboard
            </Typography>
            <motion.div whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}>
              <Button
                variant="contained"
                startIcon={<Refresh />}
                onClick={onRefresh}
                sx={{
                  background: 'linear-gradient(45deg, #64b5f6 30%, #81c784 90%)',
                  boxShadow: '0 3px 5px 2px rgba(100, 181, 246, .3)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #42a5f5 30%, #66bb6a 90%)',
                  }
                }}
              >
                Refresh
              </Button>
            </motion.div>
          </Box>

          {/* Location */}
          <Box display="flex" alignItems="center" mb={3}>
            <LocationOn sx={{ mr: 1, color: '#64b5f6' }} />
            <Typography variant="h6" sx={{ color: 'white' }}>
              {weather.city}
            </Typography>
          </Box>

          {/* Weather Cards */}
          <Grid container spacing={2} mb={3}>
            {weatherCards.map((card, index) => (
              <Grid item xs={6} key={index}>
                <Card
                  ref={el => cardRefs.current[index] = el}
                  sx={{
                    background: `linear-gradient(135deg, ${card.color}20, ${card.color}10)`,
                    border: `1px solid ${card.color}60`,
                    borderRadius: '16px',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-5px)',
                      boxShadow: `0 8px 24px ${card.color}40`,
                      border: `1px solid ${card.color}`,
                    }
                  }}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Box display="flex" alignItems="center" mb={1}>
                      {card.icon}
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                      {card.title}
                    </Typography>
                    <Typography variant="h6" sx={{ color: 'white', fontWeight: 'bold' }}>
                      {card.value}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          <Divider sx={{ my: 3, backgroundColor: 'rgba(255,255,255,0.3)' }} />

          {/* ML Predictions */}
          <Typography variant="h6" gutterBottom sx={{ color: 'white', fontWeight: 'bold', mb: 2 }}>
            🤖 ML Predictions
          </Typography>

          <Grid container spacing={2} mb={3}>
            <Grid item xs={12}>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <Card
                  sx={{
                    background: predictions.irrigation_status
                      ? 'linear-gradient(45deg, #ff7675 30%, #fd79a8 90%)'
                      : 'linear-gradient(45deg, #00b894 30%, #00cec9 90%)',
                    borderRadius: '12px'
                  }}
                >
                  <CardContent sx={{ p: 2 }}>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Box display="flex" alignItems="center">
                        <Science sx={{ mr: 1, color: 'white' }} />
                        <Typography variant="h6" sx={{ color: 'white', fontWeight: 'bold' }}>
                          Irrigation Status
                        </Typography>
                      </Box>
                      <Chip
                        label={predictions.irrigation_status ? "Needed" : "Not Needed"}
                        sx={{
                          backgroundColor: 'rgba(255,255,255,0.2)',
                          color: 'white',
                          fontWeight: 'bold'
                        }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>

            <Grid item xs={12}>
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <Card
                  sx={{
                    background: 'linear-gradient(45deg, #fdcb6e 30%, #e17055 90%)',
                    borderRadius: '12px'
                  }}
                >
                  <CardContent sx={{ p: 2 }}>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Box display="flex" alignItems="center">
                        <WaterDrop sx={{ mr: 1, color: 'white' }} />
                        <Typography variant="h6" sx={{ color: 'white', fontWeight: 'bold' }}>
                          Water Requirement
                        </Typography>
                      </Box>
                      <Typography variant="h5" sx={{ color: 'white', fontWeight: 'bold' }}>
                        {predictions.water_requirement.toFixed(2)} L
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          </Grid>

          {/* Input Features */}
          <Typography variant="h6" gutterBottom sx={{ color: 'white', fontWeight: 'bold', mb: 2 }}>
            📈 Input Features
          </Typography>

          <List dense sx={{ backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
            {Object.entries(input_features).map(([key, value], index) => (
              <motion.div
                key={key}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <ListItem>
                  <ListItemIcon sx={{ minWidth: '30px' }}>
                    <TrendingUp sx={{ color: '#64b5f6', fontSize: 18 }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Typography variant="body2" sx={{ color: 'white', fontWeight: 'medium' }}>
                        {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                      </Typography>
                    }
                    secondary={
                      <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                        {typeof value === 'number' ? value.toFixed(2) : value}
                      </Typography>
                    }
                  />
                </ListItem>
              </motion.div>
            ))}
          </List>
        </Paper>
      </motion.div>
    </Box>
  );
}

export default DataPanel;
