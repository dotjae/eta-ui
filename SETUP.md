# ETA-UI Setup Guide

## Prerequisites
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
  ```bash
  # Install uv (Mac/Linux)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Quick Start (Mac)

### 1. Clone and Navigate
```bash
cd ~/Desktop/git/eta-ui
```

### 2. Sync Dependencies
```bash
# uv automatically creates venv and installs everything
uv sync
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GTFS RT API key if you have one
```

### 4. Run the Server
```bash
# uv run automatically uses the project venv
uv run uvicorn backend.app:app --host 0.0.0.0 --port 5001 --reload
```

### 6. Access the UI
Open your browser to: **http://localhost:5001**

## What You'll See

### Single Stop Tab
- Select an active trip
- Choose a destination stop
- Track real-time ETA predictions
- View accuracy metrics when the vehicle arrives

### Multi-Stop Dashboard Tab
- Select a trip to track
- See all remaining stops in a grid
- Live ETA updates for each stop
- Automatic arrival detection
- Overall accuracy metrics when trip completes

## Troubleshooting

### Missing GTFS Database
If you don't have a `gtfs.db` file, the app will start but won't show any trips. You need to:
1. Download GTFS static data
2. Load it into the database
3. Or point to an existing database via `DATABASE_URL` in `.env`

### Port Already in Use
```bash
# Kill process on port 5001
lsof -ti:5001 | xargs kill -9
```

### Module Import Errors
Make sure you're running from the project root:
```bash
cd ~/Desktop/git/eta-ui
python3 -m uvicorn backend.app:app --port 5001
```

## API Endpoints

- `GET /api/routes` - List active routes
- `GET /api/trips` - List active trips
- `GET /api/trips/{trip_id}/stops` - Get future stops
- `POST /api/sessions` - Start single-stop tracking
- `POST /api/multi-sessions` - Start multi-stop tracking
- `WebSocket /ws/{session_id}` - Real-time single-stop updates
- `WebSocket /ws/multi/{session_id}` - Real-time multi-stop updates
