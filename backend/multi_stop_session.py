"""
Multi-stop prediction session management

Tracks predictions for all upcoming stops on a trip:
- Stores predictions for each stop
- Detects arrival at each stop independently
- Computes running validation metrics
- Persists error data for historical analysis
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import statistics

# Add eta_prediction to path for imports
eta_prediction_path = Path(__file__).parent.parent / "eta_prediction"
if str(eta_prediction_path) not in sys.path:
    sys.path.insert(0, str(eta_prediction_path))

from backend.prediction_session import Prediction


@dataclass
class StopValidation:
    """Validation data for a single stop"""
    stop_id: str
    stop_name: str
    stop_sequence: int
    
    # Predictions made for this stop
    predictions: List[Prediction] = field(default_factory=list)
    
    # Actual arrival (None until detected)
    actual_arrival: Optional[datetime] = None
    
    # Status: pending, active, arrived, skipped
    status: str = "pending"
    
    def add_prediction(self, prediction: Prediction) -> None:
        """Add a prediction for this stop"""
        self.predictions.append(prediction)
        if self.status == "pending":
            self.status = "active"
    
    def set_arrival(self, arrival_time: datetime) -> None:
        """Mark actual arrival at this stop"""
        self.actual_arrival = arrival_time
        self.status = "arrived"
    
    def compute_errors(self) -> Dict[str, Any]:
        """
        Compute prediction errors for this stop
        
        Returns dict with error statistics
        """
        if not self.actual_arrival or not self.predictions:
            return {
                'stop_id': self.stop_id,
                'stop_name': self.stop_name,
                'status': self.status,
                'error': 'No arrival or no predictions',
                'n_predictions': len(self.predictions)
            }
        
        errors = []
        predictions_with_errors = []
        
        for pred in self.predictions:
            # Error = predicted - actual (positive = late, negative = early)
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
        
        return {
            'stop_id': self.stop_id,
            'stop_name': self.stop_name,
            'stop_sequence': self.stop_sequence,
            'status': self.status,
            'actual_arrival': self.actual_arrival.isoformat(),
            'n_predictions': len(self.predictions),
            'final_error_seconds': errors[-1] if errors else None,
            'final_error_minutes': errors[-1] / 60.0 if errors else None,
            'mae_seconds': statistics.mean(abs_errors) if abs_errors else None,
            'mae_minutes': statistics.mean(abs_errors) / 60.0 if abs_errors else None,
            'mean_error_seconds': statistics.mean(errors) if errors else None,
            'mean_error_minutes': statistics.mean(errors) / 60.0 if errors else None,
            'median_error_seconds': statistics.median(errors) if errors else None,
            'stdev_error_seconds': statistics.stdev(errors) if len(errors) > 1 else None,
            'max_absolute_error_seconds': max(abs_errors) if abs_errors else None,
            'min_absolute_error_seconds': min(abs_errors) if abs_errors else None,
            'predictions': predictions_with_errors,
        }


@dataclass
class MultiStopSession:
    """
    Manages predictions and validation for all upcoming stops on a trip
    """
    session_id: str
    trip_id: str
    route_id: str
    vehicle_id: str
    
    # All stops being tracked
    stops: Dict[str, StopValidation] = field(default_factory=dict)
    
    # Session metadata
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: str = "active"  # active, completed, error
    
    # Configuration
    poll_interval: int = 10
    data_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Setup data directory if specified"""
        if self.data_dir:
            self.data_dir = Path(self.data_dir) / self.session_id
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def add_stop(self, stop_id: str, stop_name: str, stop_sequence: int) -> None:
        """Add a stop to track"""
        if stop_id not in self.stops:
            self.stops[stop_id] = StopValidation(
                stop_id=stop_id,
                stop_name=stop_name,
                stop_sequence=stop_sequence,
            )
    
    def add_prediction(self, stop_id: str, prediction_data: Dict[str, Any]) -> None:
        """Add a prediction for a specific stop"""
        if stop_id not in self.stops:
            raise ValueError(f"Stop {stop_id} not being tracked in this session")
        
        pred = Prediction(
            timestamp=datetime.now(timezone.utc),
            predicted_arrival=datetime.fromisoformat(
                prediction_data['eta_timestamp'].replace('Z', '+00:00')
            ),
            eta_seconds=prediction_data['eta_seconds'],
            distance_meters=prediction_data['distance_to_stop_m'],
            model_key=prediction_data.get('model_key', 'unknown'),
            model_type=prediction_data.get('model_type', 'unknown'),
            model_scope=prediction_data.get('model_scope', 'unknown'),
            features=prediction_data.get('features', {}),
        )
        
        self.stops[stop_id].add_prediction(pred)
    
    def set_arrival(self, stop_id: str, arrival_time: datetime) -> None:
        """Mark actual arrival at a stop"""
        if stop_id not in self.stops:
            raise ValueError(f"Stop {stop_id} not being tracked in this session")
        
        self.stops[stop_id].set_arrival(arrival_time)
        
        # Persist metrics if data_dir is set
        if self.data_dir:
            self._persist_stop_metrics(stop_id)
        
        # Check if all stops are complete
        if all(s.status in ['arrived', 'skipped'] for s in self.stops.values()):
            self.status = "completed"
            self.completed_at = datetime.now(timezone.utc)
            if self.data_dir:
                self._persist_session_summary()
    
    def _persist_stop_metrics(self, stop_id: str) -> None:
        """Save metrics for a stop to disk"""
        if not self.data_dir:
            return
        
        stop = self.stops[stop_id]
        metrics = stop.compute_errors()
        
        stop_file = self.data_dir / f"stop_{stop_id}.json"
        with open(stop_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _persist_session_summary(self) -> None:
        """Save overall session summary"""
        if not self.data_dir:
            return
        
        summary = self.compute_overall_metrics()
        
        summary_file = self.data_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_stop_metrics(self, stop_id: str) -> Dict[str, Any]:
        """Get validation metrics for a specific stop"""
        if stop_id not in self.stops:
            raise ValueError(f"Stop {stop_id} not found")
        
        return self.stops[stop_id].compute_errors()
    
    def compute_overall_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all completed stops
        
        Returns overall MAE, bias, etc. across the entire trip
        """
        completed_stops = [
            s for s in self.stops.values() 
            if s.status == 'arrived' and s.actual_arrival
        ]
        
        if not completed_stops:
            return {
                'status': self.status,
                'message': 'No completed stops yet',
                'n_stops_tracked': len(self.stops),
                'n_stops_completed': 0,
            }
        
        # Collect all errors across all stops
        all_errors = []
        all_abs_errors = []
        stop_summaries = []
        
        for stop in completed_stops:
            errors = stop.compute_errors()
            stop_summaries.append({
                'stop_id': stop.stop_id,
                'stop_name': stop.stop_name,
                'stop_sequence': stop.stop_sequence,
                'n_predictions': errors['n_predictions'],
                'mae_seconds': errors['mae_seconds'],
                'final_error_seconds': errors['final_error_seconds'],
            })
            
            # Collect individual prediction errors
            for pred_error in errors.get('predictions', []):
                all_errors.append(pred_error['error_seconds'])
                all_abs_errors.append(pred_error['absolute_error_seconds'])
        
        return {
            'session_id': self.session_id,
            'trip_id': self.trip_id,
            'route_id': self.route_id,
            'vehicle_id': self.vehicle_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'n_stops_tracked': len(self.stops),
            'n_stops_completed': len(completed_stops),
            'n_predictions_total': len(all_errors),
            
            # Overall statistics
            'overall_mae_seconds': statistics.mean(all_abs_errors) if all_abs_errors else None,
            'overall_mae_minutes': statistics.mean(all_abs_errors) / 60.0 if all_abs_errors else None,
            'overall_mean_error_seconds': statistics.mean(all_errors) if all_errors else None,
            'overall_mean_error_minutes': statistics.mean(all_errors) / 60.0 if all_errors else None,
            'overall_median_error_seconds': statistics.median(all_errors) if all_errors else None,
            'overall_stdev_error_seconds': statistics.stdev(all_errors) if len(all_errors) > 1 else None,
            'overall_max_absolute_error_seconds': max(all_abs_errors) if all_abs_errors else None,
            'overall_min_absolute_error_seconds': min(all_abs_errors) if all_abs_errors else None,
            
            # Per-stop summary
            'stops': stop_summaries,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for API responses"""
        return {
            'session_id': self.session_id,
            'trip_id': self.trip_id,
            'route_id': self.route_id,
            'vehicle_id': self.vehicle_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'n_stops_tracked': len(self.stops),
            'n_stops_active': sum(1 for s in self.stops.values() if s.status == 'active'),
            'n_stops_completed': sum(1 for s in self.stops.values() if s.status == 'arrived'),
            'stops': {
                stop_id: {
                    'stop_id': stop.stop_id,
                    'stop_name': stop.stop_name,
                    'stop_sequence': stop.stop_sequence,
                    'status': stop.status,
                    'n_predictions': len(stop.predictions),
                    'actual_arrival': stop.actual_arrival.isoformat() if stop.actual_arrival else None,
                    'latest_prediction': {
                        'timestamp': stop.predictions[-1].timestamp.isoformat(),
                        'predicted_arrival': stop.predictions[-1].predicted_arrival.isoformat(),
                        'eta_seconds': stop.predictions[-1].eta_seconds,
                    } if stop.predictions else None,
                }
                for stop_id, stop in self.stops.items()
            }
        }


# Global multi-stop session storage
_multi_sessions: Dict[str, MultiStopSession] = {}


def create_multi_stop_session(
    trip_id: str,
    route_id: str,
    vehicle_id: str,
    stops: List[Dict[str, Any]],
    data_dir: Optional[str] = None,
) -> MultiStopSession:
    """Create a new multi-stop prediction session"""
    import uuid
    session_id = str(uuid.uuid4())
    
    session = MultiStopSession(
        session_id=session_id,
        trip_id=trip_id,
        route_id=route_id,
        vehicle_id=vehicle_id,
        data_dir=data_dir,
    )
    
    # Add all stops
    for stop in stops:
        session.add_stop(
            stop_id=stop['stop_id'],
            stop_name=stop['stop_name'],
            stop_sequence=stop['stop_sequence'],
        )
    
    _multi_sessions[session_id] = session
    return session


def get_multi_stop_session(session_id: str) -> Optional[MultiStopSession]:
    """Get an existing multi-stop session"""
    return _multi_sessions.get(session_id)


def get_active_multi_stop_sessions() -> List[MultiStopSession]:
    """Get all active multi-stop sessions"""
    return [s for s in _multi_sessions.values() if s.status == "active"]


def delete_multi_stop_session(session_id: str) -> bool:
    """Delete a multi-stop session"""
    if session_id in _multi_sessions:
        del _multi_sessions[session_id]
        return True
    return False
