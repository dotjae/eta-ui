"""
Arrival detection service

Monitors GTFS-Realtime TripUpdates to detect when a vehicle arrives at a stop
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

from backend.prediction_session import PredictionSession

logger = logging.getLogger(__name__)


class ArrivalDetector:
    """
    Detects actual vehicle arrivals at stops using GTFS-Realtime data
    """

    def __init__(self, data_access):
        """
        Args:
            data_access: GTFSDataAccess instance
        """
        self.data_access = data_access
        self._running_detectors = {}  # session_id -> Task

    async def start_detection(
        self,
        session: PredictionSession,
        feed_name: str,
    ) -> None:
        """
        Start arrival detection loop for a session

        Args:
            session: PredictionSession instance
            feed_name: GTFS feed name for realtime data
        """
        session_id = session.session_id

        # Create task for this session
        task = asyncio.create_task(
            self._detection_loop(session, feed_name)
        )
        self._running_detectors[session_id] = task

        logger.info(f"Started arrival detection for session {session_id}")

    async def stop_detection(self, session_id: str) -> None:
        """Stop arrival detection for a session"""
        if session_id in self._running_detectors:
            task = self._running_detectors[session_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._running_detectors[session_id]
            logger.info(f"Stopped arrival detection for session {session_id}")

    async def _detection_loop(
        self,
        session: PredictionSession,
        feed_name: str,
    ) -> None:
        """
        Main detection loop - monitors for arrival

        Checks vehicle position and stop sequence to detect arrival
        """
        logger.info(f"Arrival detection started for {session.trip_id} -> {session.stop_id}")

        try:
            poll_interval = 5  # Check more frequently than predictions

            while session.status == "active":
                # Get latest vehicle position
                vehicle_position = self.data_access.get_latest_vehicle_position(
                    feed_name, session.trip_id
                )

                if not vehicle_position:
                    await asyncio.sleep(poll_interval)
                    continue

                current_stop_seq = vehicle_position.get('current_stop_sequence')
                
                # Check if vehicle has reached or passed our target stop
                if current_stop_seq and current_stop_seq >= session.stop_sequence:
                    # Vehicle has reached the stop
                    arrival_time = datetime.now(timezone.utc)
                    
                    # Try to get more precise timestamp from vehicle position
                    if vehicle_position.get('ts'):
                        try:
                            arrival_time = datetime.fromisoformat(
                                vehicle_position['ts'].replace('Z', '+00:00')
                            )
                        except:
                            pass
                    
                    session.set_arrival(arrival_time)
                    logger.info(
                        f"Arrival detected at {session.stop_name} "
                        f"(stop_seq {session.stop_sequence}) at {arrival_time.isoformat()}"
                    )
                    break

                await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            logger.info(f"Arrival detection cancelled for session {session.session_id}")
            raise
        except Exception as e:
            logger.error(f"Arrival detection error: {e}", exc_info=True)
            session.status = "error"
