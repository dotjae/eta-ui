"""
Prediction service

Manages live prediction loops for active sessions
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import asyncio
import logging

# Add eta_prediction to path
eta_prediction_path = Path(__file__).parent.parent / "eta_prediction"
if str(eta_prediction_path) not in sys.path:
    sys.path.insert(0, str(eta_prediction_path))

from eta_service.estimator import estimate_stop_times
from backend.prediction_session import PredictionSession

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Manages live ETA predictions for active sessions
    """

    def __init__(self, data_access):
        """
        Args:
            data_access: GTFSDataAccess instance
        """
        self.data_access = data_access
        self._running_loops = {}  # session_id -> Task

    async def start_prediction_loop(
        self,
        session: PredictionSession,
        feed_id: str,
        feed_name: str,
    ) -> None:
        """
        Start continuous prediction loop for a session

        Args:
            session: PredictionSession instance
            feed_id: GTFS feed ID for static data lookup
            feed_name: GTFS feed name for realtime data
        """
        session_id = session.session_id

        # Create task for this session
        task = asyncio.create_task(
            self._prediction_loop(session, feed_id, feed_name)
        )
        self._running_loops[session_id] = task

        logger.info(f"Started prediction loop for session {session_id}")

    async def stop_prediction_loop(self, session_id: str) -> None:
        """Stop prediction loop for a session"""
        if session_id in self._running_loops:
            task = self._running_loops[session_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._running_loops[session_id]
            logger.info(f"Stopped prediction loop for session {session_id}")

    async def _prediction_loop(
        self,
        session: PredictionSession,
        feed_id: str,
        feed_name: str,
    ) -> None:
        """
        Main prediction loop - runs until arrival or timeout

        Args:
            session: PredictionSession instance
            feed_id: GTFS feed ID
            feed_name: GTFS feed name
        """
        logger.info(f"Prediction loop started for {session.trip_id} -> {session.stop_id}")

        try:
            while session.status == "active":
                # Get latest vehicle position
                vehicle_position = self.data_access.get_latest_vehicle_position(
                    feed_name, session.trip_id
                )

                if not vehicle_position:
                    logger.warning(f"No vehicle position found for trip {session.trip_id}")
                    await asyncio.sleep(session.poll_interval)
                    continue

                # Get future stops (we need this in the format expected by estimator)
                future_stops = self.data_access.get_future_stops(
                    feed_id,
                    session.trip_id,
                    vehicle_position.get('current_stop_sequence')
                )

                if not future_stops:
                    logger.warning(f"No future stops for trip {session.trip_id}")
                    await asyncio.sleep(session.poll_interval)
                    continue

                # Find our target stop in the future stops
                target_stop = None
                for stop in future_stops:
                    if stop['stop_id'] == session.stop_id:
                        target_stop = stop
                        break

                if not target_stop:
                    logger.info(f"Target stop {session.stop_id} already passed")
                    session.status = "passed"
                    break

                # Prepare stop for prediction
                upcoming_stop = {
                    'stop_id': target_stop['stop_id'],
                    'lat': target_stop['lat'],
                    'lon': target_stop['lon'],
                    'stop_sequence': target_stop['stop_sequence'],
                }

                # Get prediction
                try:
                    result = estimate_stop_times(
                        vehicle_position=vehicle_position,
                        upcoming_stops=[upcoming_stop],
                        route_id=session.route_id,
                        trip_id=session.trip_id,
                        max_stops=1,
                    )

                    if result.get('predictions') and len(result['predictions']) > 0:
                        prediction_data = result['predictions'][0]
                        prediction_data['model_key'] = result.get('model_key')
                        prediction_data['model_type'] = result.get('model_type')
                        prediction_data['model_scope'] = result.get('model_scope')

                        # Add to session
                        session.add_prediction(prediction_data)

                        logger.info(
                            f"Prediction: {prediction_data['eta_formatted']} "
                            f"({prediction_data['distance_to_stop_m']:.1f}m away)"
                        )
                    else:
                        logger.warning(f"No prediction returned: {result.get('error')}")

                except Exception as e:
                    logger.error(f"Prediction error: {e}", exc_info=True)

                # Wait before next prediction
                await asyncio.sleep(session.poll_interval)

        except asyncio.CancelledError:
            logger.info(f"Prediction loop cancelled for session {session.session_id}")
            raise
        except Exception as e:
            logger.error(f"Prediction loop error: {e}", exc_info=True)
            session.status = "error"


async def run_single_prediction(
    vehicle_position: Dict[str, Any],
    stop: Dict[str, Any],
    route_id: str,
    trip_id: str,
) -> Dict[str, Any]:
    """
    Run a single prediction (for testing or one-off predictions)

    Args:
        vehicle_position: Vehicle position dict
        stop: Stop dict with id, lat, lon, sequence
        route_id: Route ID
        trip_id: Trip ID

    Returns:
        Prediction result dictionary
    """
    upcoming_stop = {
        'stop_id': stop['stop_id'],
        'lat': stop['lat'],
        'lon': stop['lon'],
        'stop_sequence': stop['stop_sequence'],
    }

    result = estimate_stop_times(
        vehicle_position=vehicle_position,
        upcoming_stops=[upcoming_stop],
        route_id=route_id,
        trip_id=trip_id,
        max_stops=1,
    )

    return result
