# ODIN Flight Sim Setup and Run Guide

## Quick Start Instructions

### 1. Prerequisites

Make sure you have the following installed:

- **Python 3.10+**
- **Node.js 18+**
- **MongoDB** (local installation or cloud)
- **Git**

### 2. Clone Repository

```bash
git clone <your-repo-url>
cd odin-flight-sim
```

### 3. Setup MongoDB

#### Option A: Local MongoDB

```bash
# Install MongoDB (varies by OS)
# On Ubuntu/Debian:
sudo apt-get install mongodb

# Start MongoDB service
sudo systemctl start mongodb

# Verify MongoDB is running
mongo --eval "db.adminCommand('ping')"
```

#### Option B: MongoDB Atlas (Cloud)

1. Create free account at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a cluster
3. Get connection string (replace `<password>` with your password)
4. Use this URL in your `.env` file

### 4. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env

# Edit .env file with your configuration:
# MONGODB_URL=mongodb://localhost:27017  (or your Atlas URL)
# HUGGINGFACE_API_KEY=your_key_here (optional but recommended)
# NASA_API_KEY=DEMO_KEY
```

### 5. AI Services Setup

```bash
cd ai-services

# Install AI dependencies (same virtual environment)
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Add your Hugging Face API key if you have one
```

### 6. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
# or if you prefer:
# yarn install
# bun install
```

### 7. Run the System

#### Terminal 1: Start Backend

```bash
cd backend
python main.py
```

Backend will run at: http://localhost:8000

#### Terminal 2: Start Frontend

```bash
cd frontend
npm run dev
```

Frontend will run at: http://localhost:5173

### 8. Access ODIN Mission Control

1. Open your browser to: http://localhost:5173
2. You should see the ODIN Mission Control Dashboard
3. Click **"Initialize ODIN Mission"** to start
4. Watch the autonomous AI system in action!

## 🧪 Testing the System

### Basic API Test

```bash
# Test backend health
curl http://localhost:8000/health

# Check ODIN status
curl http://localhost:8000/api/odin/status

# Initialize a mission
curl -X POST "http://localhost:8000/api/odin/initialize" \
  -H "Content-Type: application/json" \
  -d '{"destination": "Moon"}'
```

### WebSocket Test

```bash
# Install wscat if you don't have it
npm install -g wscat

# Connect to WebSocket
wscat -c ws://localhost:8000/ws
```

## 🔧 Troubleshooting

### Common Issues

#### "MongoDB connection failed"

- Ensure MongoDB is running: `sudo systemctl status mongodb`
- Check if port 27017 is available: `netstat -tlnp | grep 27017`
- Verify connection string in `.env` file

#### "ODIN AI services not available"

- Check if all dependencies are installed: `pip list | grep -E "(langchain|transformers|poliastro)"`
- Verify Python version: `python --version` (should be 3.10+)
- Check for import errors in the logs

#### "Hugging Face API errors"

- Get a free API key from [Hugging Face](https://huggingface.co/settings/tokens)
- Add it to your `.env` files: `HUGGINGFACE_API_KEY=your_key_here`
- The system will work without it, but with reduced AI capabilities

#### "Frontend build errors"

- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version: `node --version` (should be 18+)
- Try using yarn instead: `yarn install && yarn dev`

#### Port conflicts

- Backend (8000): Change `PORT=8001` in backend/.env
- Frontend (5173): Use `npm run dev -- --port 3000`

### Logs and Debugging

#### Check Backend Logs

```bash
cd backend
python main.py
# Look for startup messages and any errors
```

#### Check AI Services Logs

```bash
cd ai-services
python -c "from odin_main import OdinNavigationSystem; print('AI services working')"
```

#### MongoDB Logs

```bash
# Check MongoDB logs
sudo tail -f /var/log/mongodb/mongod.log
```

## 🚀 Advanced Configuration

### Performance Optimization

#### For Better AI Performance

1. Get a Hugging Face API key (free)
2. Consider using a local GPU if available
3. Adjust model parameters in `.env`:
   ```
   LLM_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct  # Smaller, faster model
   MAX_TOKENS=256  # Reduce for faster responses
   ```

#### For Production Use

1. Use MongoDB Atlas for reliability
2. Set up proper environment variables
3. Use process managers like PM2 for the backend
4. Build and serve the frontend statically

### Custom Configuration

#### Different Space Weather Periods

Edit in backend/.env:

```
HISTORICAL_DATA_START_YEAR=2014
HISTORICAL_DATA_END_YEAR=2016
```

#### Mission Parameters

Edit in AI services configuration:

```
MAX_MISSION_DURATION_HOURS=72
HAZARD_CHECK_INTERVAL_MINUTES=10
```

## 📊 System Status Check

After setup, verify all components:

1. **Backend Health**: http://localhost:8000/health
2. **API Documentation**: http://localhost:8000/docs
3. **ODIN Status**: http://localhost:8000/api/odin/status
4. **Frontend**: http://localhost:5173
5. **MongoDB**: Use MongoDB Compass or `mongo` command

## 🎯 Next Steps

Once everything is running:

1. **Initialize a Mission**: Use the frontend interface
2. **Try Different Scenarios**: Experiment with different historical dates
3. **Monitor Decisions**: Watch the AI decision-making process
4. **Check Decision Logs**: View the autonomous navigation choices
5. **Explore the API**: Use the interactive docs at /docs

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the terminal output for error messages
3. Ensure all dependencies are correctly installed
4. Verify your environment configuration

The system is designed to be robust and should work out of the box with the default configuration!
