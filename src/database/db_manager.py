"""
Storing/retrieving hole detections.
"""
import sqlite3


class DatabaseManager:
    def __init__(self):
        pass

    def create_tables(self):
        """Initialize database schema"""
        raise NotImplementedError

    def insert_detection(self, image_path: str, confidence: float, gps: dict) -> int:
        """Store detection"""
        raise NotImplementedError

    def get_all_detections(self) -> list:
        """Retrieve all detections"""
        raise NotImplementedError

    def get_detections_by_location(self, location: dict, radius: float) -> list:
        """Retrieve detections near location"""
        raise NotImplementedError
