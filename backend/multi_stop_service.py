"""
Multi-stop prediction and arrival detection service

Manages predictions for all upcoming stops on a trip simultaneously
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
from backend.multi_stop_session import MultiStopSession

logger = logging.getLogger(__name__)


class MultiStopService:
    """
    Manages predictions and arrival detection for all stops on a trip
    """

    def __init__(self, data_access):
        """
        Args:
            data_access: GTFSDataAccess instance
        """
        self.data_access = data_access
        self._running_loops = {}  # session_id -> Task

    async def start_session(
        self,
        session: MultiStopSession,
        feed_id: str,
        feed_name: str,
    ) -> None:
        """
        Start multi-stop prediction and detection loop

        Args:
            session: MultiStopSession instance
            feed_id: GTFS feed ID for static data
            feed_name: GTFS feed name for realtime data
        """
        session_id = session.session_id

        # Create task for this session
        task = asyncio.create_task(
            self._multi_stop_loop(session, feed_id, feed_name)
        )
        self._running_loops[session_id] = task

        logger.info(f"Started multi-stop loop for session {session_id}")

    async def stop_session(self, session_id: str) -> None:
        """Stop multi-stop loop for a session"""
        if session_id in self._running_loops:
            task = self._running_loops[session_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._running_loops[session_id]
            logger.info(f"Stopped multi-stop loop for session {session_id}")

    async def _multi_stop_loop(
        self,
        session: MultiStopSession,
        feed_id: str,
        feed_name: str,
    ) -> None:
        """
        Main loop - generates predictions for all upcoming stops and detects arrivals

        Args:
            session: MultiStopSession instance
            feed_id: GTFS feed ID
            feed_name: GTFS feed name
        """
        logger.info(f"Multi-stop loop started for trip {session.trip_id}")

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

                current_stop_seq = vehicle_position.get('current_stop_sequence')

                # Check for arrivals first
                if current_stop_seq:
                    await self._check_arrivals(session, current_stop_seq, vehicle_position)

                # Get all future stops
                future_stops = self.data_access.get_future_stops(
                    feed_id,
                    session.trip_id,
                    current_stop_seq
                )

                if not future_stops:
                    logger.info(f"No more future stops for trip {session.trip_id}")
                    session.status = "completed"
                    break

                # Generate predictions for all tracked stops that are still in the future
                upcoming_stops = []
                for stop_data in future_stops:
                    stop_id = stop_data['stop_id']
                    
                    # Only predict for stops we're tracking and haven't arrived at yet
                    if stop_id in session.stops and session.stops[stop_id].status != 'arrived':
                        upcoming_stops.append({
                            'stop_id': stop_data['stop_id'],
                            'lat': stop_data['lat'],
                            'lon': stop_data['lon'],
                            'stop_sequence': stop_data['stop_sequence'],
                        })

                if not upcoming_stops:
                    logger.info("All tracked stops have been reached")
                    session.status = "completed"
                    break

                # Get predictions for all upcoming stops
                try:
                    result = estimate_stop_times(
                        vehicle_position=vehicle_position,
                        upcoming_stops=upcoming_stops,
                        route_id=session.route_id,
                        trip_id=session.trip_id,
                        max_stops=len(upcoming_stops),
                    )

                    if result.get('predictions'):
                        model_key = result.get('model_key')
                        model_type = result.get('model_type')
                        model_scope = result.get('model_scope')

                        for prediction_data in result['predictions']:
                            stop_id = prediction_data['stop_id']
                            
                            # Add model metadata
                            prediction_data['model_key'] = model_key
                            prediction_data['model_type'] = model_type
                            prediction_data['model_scope'] = model_scope

                            # Add to session
                            try:
                                session.add_prediction(stop_id, prediction_data)
                                logger.debug(
                                    f"Prediction for {stop_id}: {prediction_data['eta_formatted']} "
                                    f"({prediction_data['distance_to_stop_m']:.1f}m away)"
                                )
                            except ValueError as e:
                                logger.warning(f"Skipping prediction for {stop_id}: {e}")

                    else:
                        logger.warning(f"No predictions returned: {result.get('error')}")

                except Exception as e:
                    logger.error(f"Prediction error: {e}", exc_info=True)

                # Wait before next prediction cycle
                await asyncio.sleep(session.poll_interval)

        except asyncio.CancelledError:
            logger.info(f"Multi-stop loop cancelled for session {session.session_id}")
            raise
        except Exception as e:
            logger.error(f"Multi-stop loop error: {e}", exc_info=True)
            session.status = "error"

    async def _check_arrivals(
        self,
        session: MultiStopSession,
        current_stop_seq: int,
        vehicle_position: Dict[str, Any]
    ) -> None:
        """
        Check if vehicle has arrived at any tracked stops

        Args:
            session: MultiStopSession instance
            current_stop_seq: Current stop sequence from vehicle position
            vehicle_position: Vehicle position data
        """
        arrival_time = datetime.now(timezone.utc)

        # Try to get more precise timestamp from vehicle position
        if vehicle_position.get('ts'):
            try:
                arrival_time = datetime.fromisoformat(
                    vehicle_position['ts'].replace('Z', '+00:00')
                )
            except:
                pass

        # Check all tracked stops
        for stop_id, stop in session.stops.items():
            # If stop sequence matches or passed, and we haven't marked arrival yet
            if stop.status == 'active' and stop.stop_sequence <= current_stop_seq:
                session.set_arrival(stop_id, arrival_time)
                logger.info(
                    f"Arrival detected at {stop.stop_name} "
                    f"(stop_seq {stop.stop_sequence}) at {arrival_time.isoformat()}"
                )
