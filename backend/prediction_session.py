"""
Prediction session management

Tracks a single observation session for a trip-stop pair:
- Stores all predictions made before arrival
- Detects arrival
- Computes evaluation metrics
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio

# Add eta_prediction to path for imports
eta_prediction_path = Path(__file__).parent.parent / "eta_prediction"
if str(eta_prediction_path) not in sys.path:
    sys.path.insert(0, str(eta_prediction_path))


@dataclass
class Prediction:
    """Single ETA prediction"""
    timestamp: datetime
    predicted_arrival: datetime
    eta_seconds: float
    distance_meters: float
    model_key: str
    model_type: str
    model_scope: str = None
    # Input features for debugging
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionSession:
    """
    Manages a single prediction session for a trip-stop pair
    """
    session_id: str
    trip_id: str
    stop_id: str
    route_id: str
    vehicle_id: str
    stop_name: str
    stop_sequence: int

    # Session state
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    predictions: List[Prediction] = field(default_factory=list)
    actual_arrival: Optional[datetime] = None
    status: str = "active"  # active, arrived, timeout, error

    # Configuration
    poll_interval: int = 10  # seconds

    def add_prediction(self, prediction_data: Dict[str, Any]) -> None:
        """Add a new prediction to the session"""
        pred = Prediction(
            timestamp=datetime.now(timezone.utc),
            predicted_arrival=datetime.fromisoformat(prediction_data['eta_timestamp'].replace('Z', '+00:00')),
            eta_seconds=prediction_data['eta_seconds'],
            distance_meters=prediction_data['distance_to_stop_m'],
            model_key=prediction_data.get('model_key', 'unknown'),
            model_type=prediction_data.get('model_type', 'unknown'),
            model_scope=prediction_data.get('model_scope', 'unknown'),
            features=prediction_data.get('features', {}),
        )
        self.predictions.append(pred)

    def set_arrival(self, arrival_time: datetime) -> None:
        """Mark the actual arrival time"""
        self.actual_arrival = arrival_time
        self.status = "arrived"

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics comparing predictions to actual arrival

        Returns:
            Dictionary with error statistics
        """
        if not self.actual_arrival or not self.predictions:
            return {
                'status': self.status,
                'error': 'No arrival detected or no predictions made',
                'n_predictions': len(self.predictions)
            }

        errors = []
        predictions_with_errors = []

        for pred in self.predictions:
            # Error = predicted - actual (positive = predicted late, negative = predicted early)
            error_seconds = (pred.predicted_arrival - self.actual_arrival).total_seconds()
            errors.append(error_seconds)

            # Time to arrival when prediction was made
            time_to_arrival = (self.actual_arrival - pred.timestamp).total_seconds()

            predictions_with_errors.append({
                'prediction_time': pred.timestamp.isoformat(),
                'predicted_arrival': pred.predicted_arrival.isoformat(),
                'eta_seconds': pred.eta_seconds,
                'distance_meters': pred.distance_meters,
                'time_to_arrival_seconds': time_to_arrival,
                'error_seconds': error_seconds,
                'error_minutes': error_seconds / 60.0,
                'absolute_error_seconds': abs(error_seconds),
            })

        # Summary statistics
        abs_errors = [abs(e) for e in errors]
        final_error = errors[-1] if errors else None

        return {
            'status': self.status,
            'session_id': self.session_id,
            'trip_id': self.trip_id,
            'stop_id': self.stop_id,
            'stop_name': self.stop_name,
            'vehicle_id': self.vehicle_id,
            'route_id': self.route_id,
            'started_at': self.started_at.isoformat(),
            'actual_arrival': self.actual_arrival.isoformat(),
            'n_predictions': len(self.predictions),
            'final_error_seconds': final_error,
            'final_error_minutes': final_error / 60.0 if final_error else None,
            'mae_seconds': sum(abs_errors) / len(abs_errors) if abs_errors else None,
            'mae_minutes': sum(abs_errors) / len(abs_errors) / 60.0 if abs_errors else None,
            'mean_error_seconds': sum(errors) / len(errors) if errors else None,
            'mean_error_minutes': sum(errors) / len(errors) / 60.0 if errors else None,
            'max_absolute_error_seconds': max(abs_errors) if abs_errors else None,
            'min_absolute_error_seconds': min(abs_errors) if abs_errors else None,
            'predictions': predictions_with_errors,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for API responses"""
        return {
            'session_id': self.session_id,
            'trip_id': self.trip_id,
            'stop_id': self.stop_id,
            'route_id': self.route_id,
            'vehicle_id': self.vehicle_id,
            'stop_name': self.stop_name,
            'stop_sequence': self.stop_sequence,
            'started_at': self.started_at.isoformat(),
            'status': self.status,
            'n_predictions': len(self.predictions),
            'actual_arrival': self.actual_arrival.isoformat() if self.actual_arrival else None,
            'latest_prediction': {
                'timestamp': self.predictions[-1].timestamp.isoformat(),
                'predicted_arrival': self.predictions[-1].predicted_arrival.isoformat(),
                'eta_seconds': self.predictions[-1].eta_seconds,
                'distance_meters': self.predictions[-1].distance_meters,
            } if self.predictions else None,
        }


# Global session storage (in production, use Redis or similar)
_sessions: Dict[str, PredictionSession] = {}


def create_session(
    trip_id: str,
    stop_id: str,
    route_id: str,
    vehicle_id: str,
    stop_name: str,
    stop_sequence: int,
) -> PredictionSession:
    """Create a new prediction session"""
    import uuid
    session_id = str(uuid.uuid4())

    session = PredictionSession(
        session_id=session_id,
        trip_id=trip_id,
        stop_id=stop_id,
        route_id=route_id,
        vehicle_id=vehicle_id,
        stop_name=stop_name,
        stop_sequence=stop_sequence,
    )

    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[PredictionSession]:
    """Get an existing session"""
    return _sessions.get(session_id)


def get_active_sessions() -> List[PredictionSession]:
    """Get all active sessions"""
    return [s for s in _sessions.values() if s.status == "active"]


def delete_session(session_id: str) -> bool:
    """Delete a session"""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False
