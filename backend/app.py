"""
Live ETA Prediction & Validation UI - FastAPI Backend

Provides REST API for:
- Listing active trips
- Getting future stops for a trip
- Starting/stopping prediction sessions
- Getting real-time predictions
- Retrieving evaluation results
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import json
import logging

# Load settings first and set MODEL_REGISTRY_DIR in environment
# This must happen before any eta_prediction imports that use the registry
from backend.config import get_settings
_settings = get_settings()
if _settings.model_registry_dir and "MODEL_REGISTRY_DIR" not in os.environ:
    os.environ["MODEL_REGISTRY_DIR"] = str(Path(_settings.model_registry_dir).resolve())

# Add eta_prediction to path
eta_prediction_path = Path(__file__).parent.parent / "eta_prediction"
if str(eta_prediction_path) not in sys.path:
    sys.path.insert(0, str(eta_prediction_path))
from backend.database import get_db, GTFSDataAccess
from backend.prediction_session import (
    create_session,
    get_session,
    get_active_sessions,
    delete_session,
    PredictionSession,
)
from backend.prediction_service import PredictionService
from backend.arrival_detector import ArrivalDetector
from backend.multi_stop_session import (
    create_multi_stop_session,
    get_multi_stop_session,
    get_active_multi_stop_sessions,
    delete_multi_stop_session,
    MultiStopSession,
)
from backend.multi_stop_service import MultiStopService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Live ETA Prediction & Validation UI",
    description="Real-time ETA prediction demonstration and validation tool",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
prediction_service: Optional[PredictionService] = None
arrival_detector: Optional[ArrivalDetector] = None
multi_stop_service: Optional[MultiStopService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global prediction_service, arrival_detector, multi_stop_service

    logger.info("Starting Live ETA UI backend...")

    # Create dummy session to initialize services
    # (They need a DB session, but we'll pass it per request)
    settings = get_settings()
    logger.info(f"Loaded settings: feed={settings.feed_name}, poll={settings.poll_interval_seconds}s")

    logger.info("Backend ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down backend...")

    # Cancel all running prediction loops
    if prediction_service:
        for session_id in list(prediction_service._running_loops.keys()):
            await prediction_service.stop_prediction_loop(session_id)

    # Cancel all arrival detectors
    if arrival_detector:
        for session_id in list(arrival_detector._running_detectors.keys()):
            await arrival_detector.stop_detection(session_id)

    # Cancel all multi-stop sessions
    if multi_stop_service:
        for session_id in list(multi_stop_service._running_loops.keys()):
            await multi_stop_service.stop_session(session_id)


# Pydantic models for API
class TripInfo(BaseModel):
    trip_id: str
    route_id: Optional[str]
    vehicle_id: Optional[str]
    current_stop_sequence: Optional[int]
    last_seen: Optional[str]
    lat: Optional[float]
    lon: Optional[float]


class StopInfo(BaseModel):
    stop_id: str
    stop_sequence: int
    stop_name: str
    lat: Optional[float]
    lon: Optional[float]
    arrival_time: Optional[str]
    departure_time: Optional[str]


class SessionCreateRequest(BaseModel):
    trip_id: str
    stop_id: str


class SessionResponse(BaseModel):
    session_id: str
    trip_id: str
    stop_id: str
    route_id: str
    vehicle_id: str
    stop_name: str
    status: str
    started_at: str
    n_predictions: int


# API Endpoints

@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "Live ETA Prediction & Validation UI",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/api/debug/feed-names")
async def get_feed_names(db: Session = Depends(get_db)):
    """Debug: Get all distinct feed names in the database"""
    from backend.database import VehiclePosition, StopTime, Stop
    from sqlalchemy import distinct, func
    from datetime import datetime

    feed_names = db.query(distinct(VehiclePosition.feed_name)).limit(10).all()

    # Get latest timestamp
    latest_ts = db.query(func.max(VehiclePosition.ts)).scalar()

    # Get count with trip_id
    with_trip = db.query(VehiclePosition).filter(VehiclePosition.trip_id.isnot(None)).count()

    # Get count with lat/lon
    with_coords = db.query(VehiclePosition).filter(
        VehiclePosition.lat.isnot(None),
        VehiclePosition.lon.isnot(None)
    ).count()

    # Check static GTFS data
    stop_times_count = db.query(StopTime).count()
    stops_count = db.query(Stop).count()

    # Get feed_ids from static data
    static_feed_ids = db.query(distinct(StopTime.feed_id)).limit(10).all()

    return {
        "feed_names": [f[0] for f in feed_names],
        "latest_timestamp": str(latest_ts) if latest_ts else None,
        "current_utc": str(datetime.utcnow()),
        "records_with_trip_id": with_trip,
        "records_with_coordinates": with_coords,
        "static_stop_times_count": stop_times_count,
        "static_stops_count": stops_count,
        "static_feed_ids": [f[0] for f in static_feed_ids],
    }


class RouteInfo(BaseModel):
    route_id: str
    active_trips: int


@app.get("/api/routes", response_model=List[RouteInfo])
async def get_active_routes(db: Session = Depends(get_db)):
    """
    Get list of currently active routes
    """
    settings = get_settings()
    data_access = GTFSDataAccess(db)

    logger.info(f"Fetching active routes for feed_name='{settings.feed_name}'")

    routes = data_access.get_active_routes(
        feed_name=settings.feed_name,
        since_minutes=60
    )

    logger.info(f"Found {len(routes)} active routes")

    return routes


@app.get("/api/trips", response_model=List[TripInfo])
async def get_active_trips(
    route_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get list of currently active trips, optionally filtered by route
    """
    settings = get_settings()
    data_access = GTFSDataAccess(db)

    logger.info(f"Fetching active trips for feed_name='{settings.feed_name}', route_id={route_id}")

    trips = data_access.get_active_trips(
        feed_name=settings.feed_name,
        since_minutes=60,
        route_id=route_id
    )

    logger.info(f"Found {len(trips)} active trips")

    return trips


@app.get("/api/trips/{trip_id}/stops", response_model=List[StopInfo])
async def get_trip_stops(
    trip_id: str,
    future_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get stops for a trip

    Args:
        trip_id: Trip ID
        future_only: If True, only return stops not yet passed
    """
    settings = get_settings()
    data_access = GTFSDataAccess(db)
    static_feed_id = settings.static_feed_id

    # Get current vehicle position to determine which stops are in the future
    vehicle_pos = data_access.get_latest_vehicle_position(settings.feed_name, trip_id)

    if future_only and vehicle_pos:
        stops = data_access.get_future_stops(
            feed_id=static_feed_id,
            trip_id=trip_id,
            current_stop_sequence=vehicle_pos.get('current_stop_sequence')
        )
    else:
        stops = data_access.get_trip_stops(
            feed_id=static_feed_id,
            trip_id=trip_id
        )

    return stops


@app.post("/api/sessions", response_model=SessionResponse)
async def create_prediction_session(
    request: SessionCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new prediction session for a trip-stop pair

    This starts:
    - Live prediction loop (generates predictions continuously)
    - Arrival detection (monitors for actual arrival)
    """
    global prediction_service, arrival_detector

    settings = get_settings()
    data_access = GTFSDataAccess(db)
    static_feed_id = settings.static_feed_id

    # Initialize services if needed
    if prediction_service is None:
        prediction_service = PredictionService(data_access)
    if arrival_detector is None:
        arrival_detector = ArrivalDetector(data_access)

    # Get trip info
    trips = data_access.get_active_trips(settings.feed_name, since_minutes=60)
    trip_info = None
    for t in trips:
        if t['trip_id'] == request.trip_id:
            trip_info = t
            break

    if not trip_info:
        raise HTTPException(status_code=404, detail="Trip not found or not active")

    # Get stop info
    stops = data_access.get_future_stops(
        feed_id=static_feed_id,
        trip_id=request.trip_id,
        current_stop_sequence=trip_info.get('current_stop_sequence')
    )

    stop_info = None
    for s in stops:
        if s['stop_id'] == request.stop_id:
            stop_info = s
            break

    if not stop_info:
        raise HTTPException(status_code=404, detail="Stop not found or already passed")

    # Create session
    session = create_session(
        trip_id=request.trip_id,
        stop_id=request.stop_id,
        route_id=trip_info['route_id'],
        vehicle_id=trip_info['vehicle_id'],
        stop_name=stop_info['stop_name'],
        stop_sequence=stop_info['stop_sequence'],
    )

    # Start prediction loop and arrival detection
    await prediction_service.start_prediction_loop(
        session=session,
        feed_id=static_feed_id,
        feed_name=settings.feed_name,
    )

    await arrival_detector.start_detection(
        session=session,
        feed_name=settings.feed_name,
    )

    logger.info(f"Created session {session.session_id} for trip {request.trip_id} -> stop {request.stop_id}")

    return session.to_dict()


@app.get("/api/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session_status(session_id: str):
    """
    Get current status of a prediction session
    """
    session = get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.to_dict()


@app.get("/api/sessions/{session_id}/predictions")
async def get_session_predictions(session_id: str):
    """
    Get all predictions for a session
    """
    session = get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        'session_id': session.session_id,
        'status': session.status,
        'predictions': [
            {
                'timestamp': p.timestamp.isoformat(),
                'predicted_arrival': p.predicted_arrival.isoformat(),
                'eta_seconds': p.eta_seconds,
                'distance_meters': p.distance_meters,
                'model_key': p.model_key,
                'model_type': p.model_type,
            }
            for p in session.predictions
        ]
    }


@app.get("/api/sessions/{session_id}/evaluation")
async def get_session_evaluation(session_id: str):
    """
    Get evaluation results for a completed session
    """
    session = get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != "arrived":
        return {
            'session_id': session_id,
            'status': session.status,
            'message': 'Evaluation only available after arrival'
        }

    metrics = session.compute_metrics()
    return metrics


@app.delete("/api/sessions/{session_id}")
async def stop_session(session_id: str):
    """
    Stop and delete a prediction session
    """
    global prediction_service, arrival_detector

    session = get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Stop prediction loop and arrival detection
    if prediction_service:
        await prediction_service.stop_prediction_loop(session_id)

    if arrival_detector:
        await arrival_detector.stop_detection(session_id)

    # Delete session
    delete_session(session_id)

    return {"message": "Session stopped and deleted"}


@app.get("/api/sessions")
async def list_sessions():
    """
    List all active sessions
    """
    active = get_active_sessions()

    return [s.to_dict() for s in active]


# Multi-Stop Session Endpoints

class MultiStopSessionCreateRequest(BaseModel):
    trip_id: str


@app.post("/api/multi-sessions", response_model=Dict[str, Any])
async def create_multi_stop_prediction_session(
    request: MultiStopSessionCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new multi-stop prediction session for an entire trip
    
    This tracks predictions for ALL upcoming stops simultaneously
    and validates as each stop is reached.
    """
    global multi_stop_service

    settings = get_settings()
    data_access = GTFSDataAccess(db)
    static_feed_id = settings.static_feed_id

    # Initialize service if needed
    if multi_stop_service is None:
        multi_stop_service = MultiStopService(data_access)

    # Get trip info
    trips = data_access.get_active_trips(settings.feed_name, since_minutes=60)
    trip_info = None
    for t in trips:
        if t['trip_id'] == request.trip_id:
            trip_info = t
            break

    if not trip_info:
        raise HTTPException(status_code=404, detail="Trip not found or not active")

    # Get all future stops for this trip
    stops = data_access.get_future_stops(
        feed_id=static_feed_id,
        trip_id=request.trip_id,
        current_stop_sequence=trip_info.get('current_stop_sequence')
    )

    if not stops:
        raise HTTPException(status_code=404, detail="No upcoming stops found for this trip")

    # Create multi-stop session
    session = create_multi_stop_session(
        trip_id=request.trip_id,
        route_id=trip_info['route_id'],
        vehicle_id=trip_info['vehicle_id'],
        stops=stops,
        data_dir=settings.validation_data_dir if hasattr(settings, 'validation_data_dir') else None,
    )

    # Start prediction and detection loop
    await multi_stop_service.start_session(
        session=session,
        feed_id=static_feed_id,
        feed_name=settings.feed_name,
    )

    logger.info(
        f"Created multi-stop session {session.session_id} for trip {request.trip_id} "
        f"tracking {len(stops)} stops"
    )

    return session.to_dict()


@app.get("/api/multi-sessions/{session_id}")
async def get_multi_stop_session_status(session_id: str):
    """
    Get current status of a multi-stop prediction session
    """
    session = get_multi_stop_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Multi-stop session not found")

    return session.to_dict()


@app.get("/api/multi-sessions/{session_id}/stop/{stop_id}/predictions")
async def get_stop_predictions(session_id: str, stop_id: str):
    """
    Get all predictions for a specific stop in a multi-stop session
    """
    session = get_multi_stop_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Multi-stop session not found")

    if stop_id not in session.stops:
        raise HTTPException(status_code=404, detail="Stop not found in this session")

    stop = session.stops[stop_id]

    return {
        'session_id': session_id,
        'stop_id': stop_id,
        'stop_name': stop.stop_name,
        'stop_sequence': stop.stop_sequence,
        'status': stop.status,
        'actual_arrival': stop.actual_arrival.isoformat() if stop.actual_arrival else None,
        'n_predictions': len(stop.predictions),
        'predictions': [
            {
                'timestamp': p.timestamp.isoformat(),
                'predicted_arrival': p.predicted_arrival.isoformat(),
                'eta_seconds': p.eta_seconds,
                'distance_meters': p.distance_meters,
                'model_key': p.model_key,
                'model_type': p.model_type,
            }
            for p in stop.predictions
        ]
    }


@app.get("/api/multi-sessions/{session_id}/stop/{stop_id}/metrics")
async def get_stop_metrics(session_id: str, stop_id: str):
    """
    Get validation metrics for a specific stop (only available after arrival)
    """
    session = get_multi_stop_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Multi-stop session not found")

    try:
        metrics = session.get_stop_metrics(stop_id)
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/multi-sessions/{session_id}/metrics")
async def get_multi_session_metrics(session_id: str):
    """
    Get overall validation metrics across all completed stops
    """
    session = get_multi_stop_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Multi-stop session not found")

    return session.compute_overall_metrics()


@app.delete("/api/multi-sessions/{session_id}")
async def stop_multi_session(session_id: str):
    """
    Stop and delete a multi-stop prediction session
    """
    global multi_stop_service

    session = get_multi_stop_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Multi-stop session not found")

    # Stop the service loop
    if multi_stop_service:
        await multi_stop_service.stop_session(session_id)

    # Delete session
    delete_multi_stop_session(session_id)

    return {"message": "Multi-stop session stopped and deleted"}


@app.get("/api/multi-sessions")
async def list_multi_sessions():
    """
    List all active multi-stop sessions
    """
    active = get_active_multi_stop_sessions()

    return [s.to_dict() for s in active]


# WebSocket endpoints

@app.websocket("/ws/multi/{session_id}")
async def multi_stop_websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time multi-stop session updates
    
    Sends updates whenever predictions are made for any stop
    and when arrivals are detected
    """
    await websocket.accept()

    session = get_multi_stop_session(session_id)
    if not session:
        await websocket.send_json({"error": "Multi-stop session not found"})
        await websocket.close()
        return

    logger.info(f"Multi-stop WebSocket connected for session {session_id}")

    try:
        # Track last known prediction count for each stop
        last_counts = {stop_id: 0 for stop_id in session.stops.keys()}
        last_status = {stop_id: session.stops[stop_id].status for stop_id in session.stops.keys()}

        while True:
            session_changed = False

            # Check each stop for updates
            for stop_id, stop in session.stops.items():
                current_count = len(stop.predictions)
                
                # New prediction for this stop
                if current_count > last_counts[stop_id]:
                    latest = stop.predictions[-1]
                    
                    await websocket.send_json({
                        'type': 'prediction',
                        'stop_id': stop_id,
                        'data': {
                            'stop_name': stop.stop_name,
                            'stop_sequence': stop.stop_sequence,
                            'timestamp': latest.timestamp.isoformat(),
                            'predicted_arrival': latest.predicted_arrival.isoformat(),
                            'eta_seconds': latest.eta_seconds,
                            'distance_meters': latest.distance_meters,
                            'model_key': latest.model_key,
                            'model_type': latest.model_type,
                        }
                    })
                    
                    last_counts[stop_id] = current_count
                    session_changed = True

                # Status changed (e.g., arrived)
                if stop.status != last_status[stop_id]:
                    if stop.status == 'arrived':
                        await websocket.send_json({
                            'type': 'arrival',
                            'stop_id': stop_id,
                            'data': {
                                'stop_name': stop.stop_name,
                                'stop_sequence': stop.stop_sequence,
                                'arrival_time': stop.actual_arrival.isoformat() if stop.actual_arrival else None,
                                'message': f'Arrived at {stop.stop_name}!'
                            }
                        })
                    
                    last_status[stop_id] = stop.status
                    session_changed = True

            # Check overall session status
            if session.status not in ['active']:
                await websocket.send_json({
                    'type': 'session_status',
                    'data': {
                        'status': session.status,
                        'message': f'Session {session.status}'
                    }
                })
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"Multi-stop WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Multi-stop WebSocket error: {e}", exc_info=True)
    finally:
        await websocket.close()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time prediction updates

    Sends updates whenever a new prediction is made
    """
    await websocket.accept()

    session = get_session(session_id)
    if not session:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    logger.info(f"WebSocket connected for session {session_id}")

    try:
        last_prediction_count = 0

        while True:
            # Check for new predictions
            current_count = len(session.predictions)

            if current_count > last_prediction_count:
                # New prediction available
                latest = session.predictions[-1]

                await websocket.send_json({
                    'type': 'prediction',
                    'data': {
                        'timestamp': latest.timestamp.isoformat(),
                        'predicted_arrival': latest.predicted_arrival.isoformat(),
                        'eta_seconds': latest.eta_seconds,
                        'distance_meters': latest.distance_meters,
                        'model_key': latest.model_key,
                        'model_type': latest.model_type,
                        'model_scope': latest.model_scope,
                        'features': latest.features,
                    }
                })

                last_prediction_count = current_count

            # Check if arrived
            if session.status == "arrived":
                await websocket.send_json({
                    'type': 'arrival',
                    'data': {
                        'arrival_time': session.actual_arrival.isoformat(),
                        'message': 'Vehicle has arrived!'
                    }
                })
                break

            # Check if session ended
            if session.status not in ['active', 'arrived']:
                await websocket.send_json({
                    'type': 'status',
                    'data': {
                        'status': session.status,
                        'message': f'Session ended with status: {session.status}'
                    }
                })
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        await websocket.close()


# Serve frontend static files
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
logger.info(f"Looking for frontend at: {frontend_path}")

if frontend_path.exists():
    logger.info(f"Mounting frontend static files from: {frontend_path}")

    @app.get("/", response_class=FileResponse)
    async def serve_frontend():
        """Serve the frontend index.html"""
        return FileResponse(frontend_path / "index.html")

    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
else:
    logger.warning(f"Frontend directory not found at: {frontend_path}")


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info"
    )
