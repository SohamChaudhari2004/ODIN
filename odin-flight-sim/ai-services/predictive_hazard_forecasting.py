import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import json
import random

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available - using simplified prediction models")

try:
    from .space_weather_service import SpaceWeatherDataService
except ImportError:
    # Fallback if module structure is different
    SpaceWeatherDataService = None

logger = logging.getLogger(__name__)

class PredictiveHazardForecasting:
    """ML-based predictive modeling for space weather and orbital hazard forecasting using historical 2012-2018 data"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.training_data = {
            'solar_flare': [],
            'cme': [],
            'radiation': [],
            'debris': [],
            'geomagnetic_storm': []
        }
        self.prediction_horizon = 72  # hours
        self.confidence_threshold = 0.7
        
        # Space weather data service for historical data
        self.space_weather_service = SpaceWeatherDataService() if SpaceWeatherDataService else None
        
        # Orbital debris database simulation (for 2012-2018 period)
        self.debris_catalog = self._initialize_debris_catalog()
        
        # Initialize models if sklearn is available
        if SKLEARN_AVAILABLE:
            self._initialize_ml_models()
            # Generate training data and train models
            self._generate_training_data()
            self._train_models()
        else:
            logger.warning("ML models running in simulation mode")
    
    def _initialize_debris_catalog(self):
        """Initialize orbital debris catalog for 2012-2018 period"""
        # Simulate a simplified debris catalog
        # In a real implementation, this would load from NORAD TLE data
        return {
            "total_objects": 16000,  # Approximate count for 2012-2018 period
            "high_risk_objects": 1200,
            "tracked_objects": 14500,
            "collision_probability_baseline": 1e-6  # per day
        }
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for different hazard types"""
        if SKLEARN_AVAILABLE:
            self.models = {
                'solar_flare': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'cme': RandomForestRegressor(n_estimators=80, random_state=42, max_depth=8),
                'radiation': RandomForestRegressor(n_estimators=60, random_state=42),
                'debris': RandomForestRegressor(n_estimators=50, random_state=42),
                'geomagnetic_storm': RandomForestRegressor(n_estimators=70, random_state=42)
            }
            
            # Initialize scalers for feature normalization
            for hazard_type in self.models.keys():
                self.scalers[hazard_type] = StandardScaler()
        
        # Load pre-trained patterns based on historical space weather data
        self._load_historical_patterns()
    
    def _load_historical_patterns(self):
        """Load historical patterns and train models with simulated data"""
        # Generate synthetic training data based on historical patterns
        # In a real implementation, this would load actual historical data
        
        logger.info("Loading historical space weather patterns (2012-2018)")
        
        # Generate sample training data for each hazard type
        self._generate_training_data()
        
        # Train models if sklearn is available
        if SKLEARN_AVAILABLE and self.models:
            self._train_models()
    
    def _generate_training_data(self):
        """Generate synthetic training data based on historical patterns"""
        # Simulate realistic training data for the 2012-2018 period
        n_samples = 1000
        
        for hazard_type in self.training_data.keys():
            features = []
            targets = []
            
            for _ in range(n_samples):
                # Generate realistic feature vectors
                if hazard_type == 'solar_flare':
                    # Features: solar flux, sunspot number, magnetic field strength
                    feature_vector = [
                        random.uniform(70, 200),    # Solar flux
                        random.randint(0, 150),     # Sunspot number
                        random.uniform(1, 10),      # Magnetic field
                        random.uniform(0, 5)        # X-ray background
                    ]
                    # Target: probability of flare in next 24h
                    target = self._calculate_flare_probability(feature_vector)
                    
                elif hazard_type == 'cme':
                    # Features: solar activity, coronal hole area, magnetic complexity
                    feature_vector = [
                        random.uniform(80, 180),    # Solar flux
                        random.uniform(0, 50),      # Coronal hole area
                        random.uniform(0, 100),     # Magnetic complexity
                        random.uniform(0, 20)       # Recent flare activity
                    ]
                    target = self._calculate_cme_probability(feature_vector)
                    
                elif hazard_type == 'radiation':
                    # Features: solar wind speed, proton flux, magnetic field
                    feature_vector = [
                        random.uniform(300, 800),   # Solar wind speed
                        random.uniform(1, 1000),    # Proton flux
                        random.uniform(2, 15),      # IMF strength
                        random.uniform(0, 10)       # Recent SEP events
                    ]
                    target = self._calculate_radiation_level(feature_vector)
                    
                else:  # debris, geomagnetic_storm
                    # Generic feature vector
                    feature_vector = [random.uniform(0, 100) for _ in range(4)]
                    target = random.uniform(0, 1)
                
                features.append(feature_vector)
                targets.append(target)
            
            self.training_data[hazard_type] = {
                'features': features,
                'targets': targets
            }
        
        logger.info(f"Generated training data for {len(self.training_data)} hazard types")
    
    def _calculate_flare_probability(self, features):
        """Calculate solar flare probability based on features"""
        solar_flux, sunspots, mag_field, xray_bg = features
        
        # Simple heuristic based on solar activity
        prob = 0.0
        if solar_flux > 120:
            prob += 0.3
        if sunspots > 50:
            prob += 0.4
        if mag_field > 5:
            prob += 0.2
        if xray_bg > 2:
            prob += 0.1
        
        return min(prob, 1.0)
    
    def _calculate_cme_probability(self, features):
        """Calculate CME probability based on features"""
        solar_flux, coronal_holes, mag_complexity, recent_flares = features
        
        # CME probability calculation
        prob = 0.0
        if solar_flux > 140:
            prob += 0.2
        if coronal_holes > 30:
            prob += 0.3
        if mag_complexity > 70:
            prob += 0.3
        if recent_flares > 10:
            prob += 0.2
        
        return min(prob, 1.0)
    
    def _calculate_radiation_level(self, features):
        """Calculate radiation level based on features"""
        wind_speed, proton_flux, imf_strength, recent_sep = features
        
        # Radiation level calculation (0-1 scale)
        level = 0.1  # baseline
        if wind_speed > 600:
            level += 0.3
        if proton_flux > 100:
            level += 0.4
        if imf_strength > 10:
            level += 0.2
        if recent_sep > 5:
            level += 0.3
        
        return min(level, 1.0)
    
    def _train_models(self):
        """Train ML models with generated data"""
        if not SKLEARN_AVAILABLE:
            return
        
        for hazard_type, data in self.training_data.items():
            if hazard_type in self.models and data['features']:
                try:
                    features = np.array(data['features'])
                    targets = np.array(data['targets'])
                    
                    # Scale features and fit scaler
                    if hazard_type in self.scalers:
                        features_scaled = self.scalers[hazard_type].fit_transform(features)
                    else:
                        features_scaled = features
                    
                    # Train model
                    self.models[hazard_type].fit(features_scaled, targets)
                    
                    logger.info(f"Trained {hazard_type} prediction model with {len(targets)} samples")
                    
                except Exception as e:
                    logger.error(f"Error training {hazard_type} model: {e}")
                    
        logger.info("ML model training completed")
    
    async def predict_hazards(self, space_weather_data: Dict[str, Any], 
                             prediction_horizon: int = 72) -> Dict[str, Any]:
        """Predict hazards based on current space weather conditions"""
        
        predictions = {
            'prediction_horizon_hours': prediction_horizon,
            'predicted_hazards': [],
            'confidence_scores': {},
            'model_status': 'active' if SKLEARN_AVAILABLE else 'simulation'
        }
        
        try:
            # Extract features from space weather data
            features = self._extract_features_from_weather(space_weather_data)
            
            # Make predictions for each hazard type
            for hazard_type in self.models.keys():
                prediction = await self._predict_single_hazard(hazard_type, features, prediction_horizon)
                if prediction:
                    predictions['predicted_hazards'].append(prediction)
                    predictions['confidence_scores'][hazard_type] = prediction.get('confidence', 0.5)
            
            # Add overall risk assessment
            predictions['overall_risk_level'] = self._assess_overall_risk_level(predictions['predicted_hazards'])
            
        except Exception as e:
            logger.error(f"Error in hazard prediction: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    def _extract_features_from_weather(self, weather_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract ML features from space weather data"""
        
        solar_activity = weather_data.get('solar_activity', {})
        geomagnetic = weather_data.get('geomagnetic_activity', {})
        solar_wind = weather_data.get('solar_wind', {})
        
        features = {
            'solar_flare': [
                solar_activity.get('solar_flux_10_7cm', 120),
                solar_activity.get('sunspot_number', 50),
                solar_wind.get('magnetic_field_nt', 5),
                geomagnetic.get('kp_index', 3)
            ],
            'cme': [
                solar_activity.get('solar_flux_10_7cm', 120),
                solar_wind.get('speed_km_s', 400),
                solar_wind.get('magnetic_field_nt', 5),
                len(weather_data.get('active_events', []))
            ],
            'radiation': [
                solar_wind.get('speed_km_s', 400),
                solar_wind.get('proton_flux', 100),
                solar_wind.get('magnetic_field_nt', 5),
                geomagnetic.get('kp_index', 3)
            ],
            'debris': [
                geomagnetic.get('kp_index', 3),
                solar_activity.get('solar_flux_10_7cm', 120),
                random.uniform(0, 1),  # Orbital perturbation factor
                random.uniform(0, 1)   # Atmospheric density variation
            ],
            'geomagnetic_storm': [
                geomagnetic.get('kp_index', 3),
                solar_wind.get('speed_km_s', 400),
                solar_wind.get('magnetic_field_nt', 5),
                geomagnetic.get('dst_index', -20)
            ]
        }
        
        return features
    
    async def _predict_single_hazard(self, hazard_type: str, features: Dict[str, List[float]], 
                                    horizon: int) -> Optional[Dict[str, Any]]:
        """Predict a single type of hazard"""
        
        if hazard_type not in features:
            return None
        
        try:
            feature_vector = features[hazard_type]
            
            if SKLEARN_AVAILABLE and hazard_type in self.models:
                # Use ML model for prediction
                if hazard_type in self.scalers:
                    feature_vector_scaled = self.scalers[hazard_type].transform([feature_vector])
                else:
                    feature_vector_scaled = [feature_vector]
                
                prediction_value = self.models[hazard_type].predict(feature_vector_scaled)[0]
                confidence = 0.7 + random.uniform(-0.2, 0.2)  # Simulate model confidence
            else:
                # Use simple heuristic
                if hazard_type == 'solar_flare':
                    prediction_value = self._calculate_flare_probability(feature_vector)
                elif hazard_type == 'cme':
                    prediction_value = self._calculate_cme_probability(feature_vector)
                else:
                    prediction_value = random.uniform(0.1, 0.8)
                
                confidence = 0.6
            
            # Only return significant predictions
            if prediction_value > 0.3:
                severity = 'high' if prediction_value > 0.7 else 'moderate' if prediction_value > 0.5 else 'low'
                
                return {
                    'type': hazard_type,
                    'probability': float(prediction_value),
                    'severity': severity,
                    'time_to_peak': random.uniform(6, horizon),
                    'duration_hours': random.uniform(2, 24),
                    'confidence': float(confidence),
                    'predicted_at': datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error predicting {hazard_type}: {e}")
            return None
        
        return None
    
    def _assess_overall_risk_level(self, hazards: List[Dict[str, Any]]) -> str:
        """Assess overall mission risk level"""
        if not hazards:
            return 'low'
        
        high_risk_count = len([h for h in hazards if h.get('severity') == 'high'])
        moderate_risk_count = len([h for h in hazards if h.get('severity') == 'moderate'])
        
        if high_risk_count > 1:
            return 'critical'
        elif high_risk_count > 0:
            return 'high'
        elif moderate_risk_count > 2:
            return 'high'
        elif moderate_risk_count > 0:
            return 'moderate'
        else:
            return 'low'
    
    def _load_historical_patterns(self):
        """Load historical space weather patterns for training"""
        # Simulate historical space weather data patterns
        # In a real implementation, this would load from NASA/NOAA historical databases
        
        # Solar flare patterns (simplified features: sunspot_number, magnetic_field, solar_wind_speed)
        self.historical_patterns = {
            'solar_flare': {
                'features': ['sunspot_number', 'magnetic_field_strength', 'solar_wind_speed', 'x_ray_flux'],
                'typical_values': [50, 100, 400, 1e-6],
                'variance': [30, 50, 100, 5e-7]
            },
            'cme': {
                'features': ['coronal_hole_area', 'magnetic_field_rotation', 'plasma_temperature', 'density'],
                'typical_values': [1000, 45, 2e6, 5],
                'variance': [500, 20, 5e5, 2]
            },
            'radiation': {
                'features': ['solar_particle_flux', 'galactic_cosmic_rays', 'magnetosphere_compression'],
                'typical_values': [100, 50, 0.8],
                'variance': [50, 10, 0.2]
            }
        }
    
    async def predict_solar_activity(self, current_data: Dict[str, Any], forecast_hours: int = 72) -> Dict[str, Any]:
        """Predict solar activity and flare probability"""
        
        if SKLEARN_AVAILABLE:
            return await self._ml_predict_solar_activity(current_data, forecast_hours)
        else:
            return self._simulate_solar_prediction(current_data, forecast_hours)
    
    async def _ml_predict_solar_activity(self, current_data: Dict, forecast_hours: int) -> Dict:
        """ML-based solar activity prediction"""
        try:
            # Extract features from current data
            features = self._extract_solar_features(current_data)
            
            # Generate time series predictions
            time_points = np.arange(1, forecast_hours + 1)
            predictions = []
            
            for hour in time_points:
                # Create feature vector for this time point
                feature_vector = np.array(features + [hour]).reshape(1, -1)
                
                # Predict solar flare probability
                if len(self.training_data['solar_flare']) > 10:
                    flare_prob = self.models['solar_flare'].predict(feature_vector)[0]
                else:
                    # Use simplified model if insufficient training data
                    flare_prob = self._calculate_flare_probability_hourly(features, hour)
                
                predictions.append({
                    'time_offset_hours': hour,
                    'flare_probability': max(0, min(1, flare_prob)),
                    'severity_estimate': self._estimate_flare_severity(flare_prob),
                    'confidence': self._calculate_confidence(features, 'solar_flare')
                })
            
            # Identify high-risk periods
            high_risk_periods = self._identify_high_risk_periods(predictions)
            
            return {
                'forecast_type': 'solar_activity',
                'forecast_horizon_hours': forecast_hours,
                'predictions': predictions,
                'high_risk_periods': high_risk_periods,
                'recommendation': self._generate_solar_recommendation(predictions),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ML solar prediction: {e}")
            return self._simulate_solar_prediction(current_data, forecast_hours)
    
    def _simulate_solar_prediction(self, current_data: Dict, forecast_hours: int) -> Dict:
        """Simulate solar activity prediction when ML is not available"""
        
        # Base probability from current conditions
        base_prob = 0.1  # 10% baseline chance
        
        # Adjust based on current data
        sunspot_activity = current_data.get('sunspot_number', 50)
        if sunspot_activity > 100:
            base_prob += 0.3
        elif sunspot_activity > 75:
            base_prob += 0.1
        
        predictions = []
        for hour in range(1, forecast_hours + 1):
            # Simulate solar cycle variation (11-year cycle simplified)
            cycle_factor = 0.5 + 0.3 * np.sin(hour * 2 * np.pi / (11 * 365 * 24))
            
            # Add random variation
            noise = np.random.normal(0, 0.05)
            
            flare_prob = base_prob * cycle_factor + noise
            flare_prob = max(0, min(1, flare_prob))
            
            predictions.append({
                'time_offset_hours': hour,
                'flare_probability': flare_prob,
                'severity_estimate': self._estimate_flare_severity(flare_prob),
                'confidence': 0.6  # Moderate confidence for simulated data
            })
        
        high_risk_periods = [p for p in predictions if p['flare_probability'] > 0.4]
        
        return {
            'forecast_type': 'solar_activity',
            'forecast_horizon_hours': forecast_hours,
            'predictions': predictions,
            'high_risk_periods': high_risk_periods,
            'recommendation': self._generate_solar_recommendation(predictions),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def predict_cme_impact(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict Coronal Mass Ejection arrival and impact"""
        
        # Extract current solar wind conditions
        solar_wind_speed = current_data.get('solar_wind_speed', 400)  # km/s
        magnetic_field = current_data.get('magnetic_field_strength', 5)  # nT
        
        # Estimate CME travel time (simplified)
        distance_to_earth = 1.496e8  # km (1 AU)
        estimated_speed = solar_wind_speed * 1.5  # CMEs typically 1.5x solar wind speed
        
        arrival_time_hours = (distance_to_earth / estimated_speed) / 3600
        
        # Estimate impact severity
        impact_severity = self._calculate_cme_severity(estimated_speed, magnetic_field)
        
        return {
            'forecast_type': 'cme_impact',
            'estimated_arrival_hours': arrival_time_hours,
            'estimated_speed': estimated_speed,
            'impact_severity': impact_severity,
            'magnetic_field_strength': magnetic_field,
            'duration_estimate_hours': self._estimate_cme_duration(impact_severity),
            'recommendation': self._generate_cme_recommendation(impact_severity, arrival_time_hours),
            'confidence': 0.75,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def predict_radiation_levels(self, current_data: Dict[str, Any], trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict radiation exposure levels along trajectory"""
        
        # Get trajectory points
        trajectory_points = trajectory_data.get('trajectory_points', [])
        
        radiation_predictions = []
        for i, point in enumerate(trajectory_points):
            # Calculate radiation at this orbital position
            altitude = np.linalg.norm(point) - 6371  # km above Earth
            
            # Van Allen belt radiation (simplified model)
            radiation_level = self._calculate_orbital_radiation(altitude, current_data)
            
            radiation_predictions.append({
                'time_offset_hours': i * trajectory_data.get('time_step', 1),
                'altitude_km': altitude,
                'radiation_level_mrem_hr': radiation_level,
                'cumulative_dose_mrem': sum([p.get('radiation_level_mrem_hr', 0) for p in radiation_predictions]) + radiation_level
            })
        
        # Identify high radiation periods
        high_radiation_periods = [p for p in radiation_predictions if p['radiation_level_mrem_hr'] > 50]
        
        total_dose = radiation_predictions[-1]['cumulative_dose_mrem'] if radiation_predictions else 0
        
        return {
            'forecast_type': 'radiation_exposure',
            'trajectory_name': trajectory_data.get('name', 'Current'),
            'predictions': radiation_predictions,
            'high_radiation_periods': high_radiation_periods,
            'total_mission_dose_mrem': total_dose,
            'dose_limit_percentage': (total_dose / 5000) * 100,  # Assuming 5000 mrem annual limit
            'recommendation': self._generate_radiation_recommendation(total_dose, high_radiation_periods),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def predict_debris_conjunctions(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict potential debris conjunctions along trajectory"""
        
        trajectory_points = trajectory_data.get('trajectory_points', [])
        conjunctions = []
        
        # Simulate debris field (in real implementation, would use actual catalog data)
        debris_objects = self._generate_debris_field()
        
        for i, point in enumerate(trajectory_points):
            time_offset = i * trajectory_data.get('time_step', 1)
            
            # Check for close approaches with debris
            for debris in debris_objects:
                distance = self._calculate_debris_distance(point, debris, time_offset)
                
                if distance < 10:  # km - potential conjunction
                    conjunctions.append({
                        'time_offset_hours': time_offset,
                        'debris_id': debris['id'],
                        'closest_approach_km': distance,
                        'relative_velocity_km_s': debris.get('velocity', 7.5),
                        'collision_probability': self._calculate_collision_probability(distance),
                        'recommended_action': self._get_debris_action(distance)
                    })
        
        # Sort by time
        conjunctions.sort(key=lambda x: x['time_offset_hours'])
        
        return {
            'forecast_type': 'debris_conjunctions',
            'trajectory_name': trajectory_data.get('name', 'Current'),
            'total_conjunctions': len(conjunctions),
            'high_risk_conjunctions': [c for c in conjunctions if c['collision_probability'] > 0.001],
            'conjunctions': conjunctions[:20],  # Limit to first 20
            'recommendation': self._generate_debris_recommendation(conjunctions),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _extract_solar_features(self, data: Dict) -> List[float]:
        """Extract features for solar activity prediction"""
        return [
            data.get('sunspot_number', 50),
            data.get('magnetic_field_strength', 5),
            data.get('solar_wind_speed', 400),
            data.get('x_ray_flux', 1e-6)
        ]
    
    def _calculate_flare_probability_hourly(self, features: List[float], hour: int) -> float:
        """Calculate solar flare probability using simplified model for hourly prediction"""
        sunspot_num, mag_field, wind_speed, x_ray = features
        
        # Simplified probability calculation
        prob = (sunspot_num / 200) * 0.4 + (mag_field / 20) * 0.3 + (wind_speed / 800) * 0.2 + (x_ray / 1e-5) * 0.1
        
        # Add time-based decay
        prob *= np.exp(-hour / 48)  # Decay over 48 hours
        
        return max(0, min(1, prob))
    
    def _estimate_flare_severity(self, probability: float) -> str:
        """Estimate flare severity class"""
        if probability > 0.8:
            return "X-class"
        elif probability > 0.6:
            return "M-class"
        elif probability > 0.3:
            return "C-class"
        else:
            return "B-class"
    
    def _calculate_confidence(self, features: List[float], model_type: str) -> float:
        """Calculate prediction confidence"""
        # Base confidence on data quality and model training
        base_confidence = 0.7
        
        # Adjust based on feature completeness
        feature_completeness = sum(1 for f in features if f > 0) / len(features)
        
        return base_confidence * feature_completeness
    
    def _identify_high_risk_periods(self, predictions: List[Dict]) -> List[Dict]:
        """Identify continuous high-risk periods"""
        high_risk_periods = []
        current_period = None
        
        for pred in predictions:
            if pred['flare_probability'] > 0.4:  # High risk threshold
                if current_period is None:
                    current_period = {
                        'start_hour': pred['time_offset_hours'],
                        'max_probability': pred['flare_probability'],
                        'severity': pred['severity_estimate']
                    }
                else:
                    # Extend current period
                    current_period['max_probability'] = max(current_period['max_probability'], pred['flare_probability'])
                    if pred['flare_probability'] > current_period['max_probability']:
                        current_period['severity'] = pred['severity_estimate']
            else:
                if current_period is not None:
                    current_period['end_hour'] = predictions[predictions.index(pred) - 1]['time_offset_hours']
                    current_period['duration_hours'] = current_period['end_hour'] - current_period['start_hour'] + 1
                    high_risk_periods.append(current_period)
                    current_period = None
        
        return high_risk_periods
    
    def _generate_solar_recommendation(self, predictions: List[Dict]) -> str:
        """Generate solar activity recommendations"""
        max_prob = max([p['flare_probability'] for p in predictions], default=0)
        
        if max_prob > 0.8:
            return "CRITICAL: High solar flare probability. Consider mission delay or enhanced shielding."
        elif max_prob > 0.5:
            return "WARNING: Elevated solar activity expected. Monitor closely and prepare contingencies."
        elif max_prob > 0.3:
            return "CAUTION: Moderate solar activity possible. Maintain standard monitoring protocols."
        else:
            return "Solar conditions favorable for mission operations."
    
    def _calculate_cme_severity(self, speed: float, magnetic_field: float) -> int:
        """Calculate CME impact severity (1-10 scale)"""
        speed_factor = min(speed / 1000, 2.0)  # Normalize to ~1000 km/s
        field_factor = min(magnetic_field / 20, 1.5)  # Normalize to ~20 nT
        
        severity = int((speed_factor + field_factor) * 3)
        return max(1, min(10, severity))
    
    def _estimate_cme_duration(self, severity: int) -> float:
        """Estimate CME impact duration"""
        base_duration = 6  # hours
        return base_duration + (severity * 1.5)
    
    def _generate_cme_recommendation(self, severity: int, arrival_hours: float) -> str:
        """Generate CME impact recommendations"""
        if severity > 7:
            return f"SEVERE CME impact expected in {arrival_hours:.1f} hours. Implement emergency protocols."
        elif severity > 5:
            return f"Moderate CME impact in {arrival_hours:.1f} hours. Prepare protective measures."
        else:
            return f"Minor CME impact expected in {arrival_hours:.1f} hours. Standard monitoring sufficient."
    
    def _calculate_orbital_radiation(self, altitude: float, current_data: Dict) -> float:
        """Calculate radiation level at orbital altitude"""
        # Simplified Van Allen belt model
        if 500 < altitude < 2000:  # Inner belt
            base_radiation = 100 * np.exp(-(altitude - 1000)**2 / 500000)
        elif 15000 < altitude < 25000:  # Outer belt
            base_radiation = 50 * np.exp(-(altitude - 20000)**2 / 25000000)
        else:
            base_radiation = 5  # Background cosmic radiation
        
        # Adjust for solar activity
        solar_factor = current_data.get('solar_wind_speed', 400) / 400
        
        return base_radiation * solar_factor
    
    def _generate_radiation_recommendation(self, total_dose: float, high_periods: List) -> str:
        """Generate radiation exposure recommendations"""
        if total_dose > 4000:  # mrem
            return "CRITICAL: Radiation dose exceeds safe limits. Modify trajectory or implement shielding."
        elif total_dose > 2000:
            return "WARNING: High radiation exposure. Monitor crew health and consider trajectory optimization."
        elif len(high_periods) > 5:
            return "Multiple high-radiation periods detected. Schedule activities accordingly."
        else:
            return "Radiation exposure within acceptable limits for mission duration."
    
    def _generate_debris_field(self) -> List[Dict]:
        """Generate simulated debris field"""
        debris = []
        for i in range(100):  # 100 debris objects
            debris.append({
                'id': f"DEBRIS_{i:03d}",
                'position': np.random.normal(0, 2000, 3).tolist(),  # km
                'velocity': np.random.uniform(6, 9),  # km/s
                'size': np.random.exponential(0.5)  # m
            })
        return debris
    
    def _calculate_debris_distance(self, spacecraft_pos: List, debris: Dict, time_offset: float) -> float:
        """Calculate distance to debris object"""
        # Simplified calculation - real implementation would use orbital propagation
        debris_pos = debris['position']
        distance = np.linalg.norm(np.array(spacecraft_pos) - np.array(debris_pos))
        
        # Add time-based position change (simplified)
        velocity_offset = time_offset * debris['velocity']
        distance += np.random.normal(0, velocity_offset * 0.1)
        
        return max(0, distance)
    
    def _calculate_collision_probability(self, distance: float) -> float:
        """Calculate collision probability based on distance"""
        if distance < 1:
            return 0.1  # 10% for very close approach
        elif distance < 5:
            return 0.001  # 0.1% for close approach
        else:
            return 1e-6  # Very low for distant approach
    
    def _get_debris_action(self, distance: float) -> str:
        """Get recommended action for debris conjunction"""
        if distance < 1:
            return "EMERGENCY MANEUVER REQUIRED"
        elif distance < 5:
            return "Consider avoidance maneuver"
        else:
            return "Monitor closely"
    
    def _generate_debris_recommendation(self, conjunctions: List) -> str:
        """Generate debris conjunction recommendations"""
        high_risk = [c for c in conjunctions if c['collision_probability'] > 0.001]
        
        if len(high_risk) > 3:
            return "CRITICAL: Multiple high-risk conjunctions. Plan comprehensive avoidance strategy."
        elif len(high_risk) > 0:
            return f"WARNING: {len(high_risk)} high-risk conjunction(s) detected. Prepare avoidance maneuvers."
        elif len(conjunctions) > 10:
            return "Multiple minor conjunctions detected. Maintain enhanced tracking."
        else:
            return "Debris environment acceptable for current trajectory."
    
    def update_training_data(self, hazard_type: str, features: List[float], outcome: float):
        """Update training data with new observations"""
        if hazard_type in self.training_data:
            self.training_data[hazard_type].append({
                'features': features,
                'outcome': outcome,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Retrain model if enough data
            if len(self.training_data[hazard_type]) >= 20 and SKLEARN_AVAILABLE:
                self._retrain_model(hazard_type)
    
    def _retrain_model(self, hazard_type: str):
        """Retrain model with updated data"""
        if not SKLEARN_AVAILABLE or hazard_type not in self.models:
            return
        
        data = self.training_data[hazard_type]
        if len(data) < 10:
            return
        
        X = np.array([d['features'] for d in data])
        y = np.array([d['outcome'] for d in data])
        
        try:
            self.models[hazard_type].fit(X, y)
            logger.info(f"Retrained {hazard_type} model with {len(data)} samples")
        except Exception as e:
            logger.error(f"Error retraining {hazard_type} model: {e}")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        performance = {}
        
        for model_type in self.models:
            training_samples = len(self.training_data.get(model_type, []))
            performance[model_type] = {
                'training_samples': training_samples,
                'model_available': SKLEARN_AVAILABLE,
                'last_updated': datetime.utcnow().isoformat() if training_samples > 0 else None
            }
        
        return performance
