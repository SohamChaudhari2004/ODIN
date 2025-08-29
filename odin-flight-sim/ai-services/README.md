# AI Services Module

This module provides AI-powered capabilities for the Odin Flight Simulation system, including intelligent co-pilot assistance, predictive hazard forecasting, and decision explainability.

## Components

### 1. AI Co-pilot (`ai_copilot.py`)

Intelligent mission analysis and strategic guidance system.

**Features:**

- Mission brief generation using OpenAI GPT models
- Trajectory decision explanations
- Hazard response strategy recommendations
- Performance analysis and optimization suggestions
- Context-aware recommendations based on mission state

**Usage:**

```python
from ai_services import AICoPilot

copilot = AICoPilot()
mission_brief = await copilot.generate_mission_brief(mission_data)
```

### 2. Predictive Hazard Forecasting (`predictive_hazard_forecasting.py`)

Machine learning-based forecasting for space weather and hazard prediction.

**Features:**

- Solar activity prediction (flares, CMEs)
- Radiation exposure forecasting along trajectories
- Debris conjunction prediction
- ML model training and performance tracking
- Confidence estimation for predictions

**Usage:**

```python
from ai_services import PredictiveHazardForecasting

forecaster = PredictiveHazardForecasting()
solar_forecast = await forecaster.predict_solar_activity(current_data, 72)
```

### 3. Explainability Module (`explainability_module.py`)

Generates human-readable explanations for AI decisions and recommendations.

**Features:**

- Trajectory selection explanations
- Hazard response rationale
- AI recommendation justification
- Multi-audience explanations (crew, mission control, public)
- Decision narrative generation

**Usage:**

```python
from ai_services import ExplainabilityModule

explainer = ExplainabilityModule()
explanation = explainer.explain_trajectory_selection(decision_data, audience="crew")
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for AI Co-pilot functionality
- Optional: Configure model preferences and API endpoints

### Dependencies

- **Required**: `numpy`, `asyncio`, `logging`
- **Optional**: `openai` (for AI Co-pilot), `scikit-learn` (for ML forecasting)

### Graceful Degradation

All services include fallback modes when optional dependencies are unavailable:

- AI Co-pilot provides simulated responses without OpenAI API
- Predictive forecasting uses simplified models without scikit-learn

## Integration

### With Backend Services

```python
# In backend/app/services/
from ai_services import AICoPilot, PredictiveHazardForecasting, ExplainabilityModule

# Initialize AI services
ai_copilot = AICoPilot()
forecaster = PredictiveHazardForecasting()
explainer = ExplainabilityModule()

# Use in trajectory planning
explanation = explainer.explain_trajectory_selection(decision_data)
forecast = await forecaster.predict_radiation_levels(current_data, trajectory_data)
brief = await ai_copilot.generate_mission_brief(mission_state)
```

### API Endpoints

The AI services are designed to integrate with FastAPI endpoints:

```python
@app.post("/api/ai/mission-brief")
async def generate_mission_brief(mission_data: dict):
    brief = await ai_copilot.generate_mission_brief(mission_data)
    return {"mission_brief": brief}

@app.post("/api/ai/hazard-forecast")
async def forecast_hazards(forecast_request: dict):
    forecast = await forecaster.predict_solar_activity(forecast_request)
    return forecast

@app.post("/api/ai/explain-decision")
async def explain_decision(decision_data: dict):
    explanation = explainer.explain_trajectory_selection(decision_data)
    return explanation
```

## Performance Considerations

### AI Co-pilot

- API calls to OpenAI may introduce latency (1-3 seconds)
- Implements context history to improve relevance
- Graceful fallback to local responses when API unavailable

### Predictive Forecasting

- ML model training requires minimum data samples (10-20)
- Prediction accuracy improves with operational data
- Caching recommended for repeated forecasts

### Explainability Module

- Fast local processing for all explanations
- Template-based generation ensures consistent format
- Audience-specific explanations require different complexity levels

## Future Enhancements

1. **Advanced ML Models**: Integration with specialized space weather models
2. **Real-time Learning**: Continuous model updates from mission data
3. **Multi-modal AI**: Vision models for spacecraft imagery analysis
4. **Federated Learning**: Cross-mission knowledge sharing
5. **Advanced NLP**: More sophisticated explanation generation

## Error Handling

All services implement comprehensive error handling:

- Graceful degradation when dependencies unavailable
- Fallback responses for API failures
- Logging for debugging and monitoring
- Input validation and sanitization

## Testing

```bash
# Run AI services tests
python -m pytest ai-services/tests/

# Test with mock data
python ai-services/test_ai_integration.py
```
