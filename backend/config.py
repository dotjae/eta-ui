"""
Configuration for ETA UI backend
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./gtfs.db"

    # GTFS RT API
    gtfs_rt_api_url: str = "https://api.511.org/transit"
    gtfs_rt_api_key: str = ""

    # Feed identifiers
    feed_name: str = "sf-muni"
    static_feed_id: str = "SF"
    poll_interval_seconds: int = 30

    # Model registry
    model_registry_dir: str = str(Path(__file__).parent.parent / "eta_prediction" / "models" / "registry")

    # Validation data output directory
    validation_data_dir: str = str(Path(__file__).parent.parent / "data" / "validation")

    # Server
    host: str = "0.0.0.0"
    port: int = 5001
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()
