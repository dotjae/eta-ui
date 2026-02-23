"""
Database connection and GTFS data access
"""
from sqlalchemy import create_engine, Column, String, Integer, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from backend.config import get_settings

settings = get_settings()

# Create engine
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Basic GTFS models (extend as needed)
class Stop(Base):
    __tablename__ = "stops"
    stop_id = Column(String, primary_key=True)
    stop_name = Column(String)
    stop_lat = Column(Float)
    stop_lon = Column(Float)


class Trip(Base):
    __tablename__ = "trips"
    trip_id = Column(String, primary_key=True)
    route_id = Column(String)
    service_id = Column(String)
    direction_id = Column(Integer)


class StopTime(Base):
    __tablename__ = "stop_times"
    trip_id = Column(String, primary_key=True)
    stop_id = Column(String, primary_key=True)
    stop_sequence = Column(Integer, primary_key=True)
    arrival_time = Column(String)
    departure_time = Column(String)


def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class GTFSDataAccess:
    """Helper class for GTFS queries"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_stop(self, stop_id: str):
        return self.db.query(Stop).filter(Stop.stop_id == stop_id).first()
    
    def get_trip(self, trip_id: str):
        return self.db.query(Trip).filter(Trip.trip_id == trip_id).first()
    
    def get_stop_times(self, trip_id: str):
        return self.db.query(StopTime).filter(
            StopTime.trip_id == trip_id
        ).order_by(StopTime.stop_sequence).all()
    
    def get_future_stops(self, trip_id: str, from_stop_sequence: int = 0):
        """Get all stops after a given sequence for a trip"""
        return self.db.query(StopTime, Stop).join(
            Stop, StopTime.stop_id == Stop.stop_id
        ).filter(
            StopTime.trip_id == trip_id,
            StopTime.stop_sequence > from_stop_sequence
        ).order_by(StopTime.stop_sequence).all()
