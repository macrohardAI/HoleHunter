"""
Storing/retrieving hole detections.
"""
import sqlite3
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = 'database/hole.db', schema_path: str = 'src/database/schema.sql'):
        self.db_path = db_path
        self.schema_path = schema_path
        try:
            self.db_path.parent.mkdir(parents = True, exist_ok = True)
        except Exception as e:
            print(f"ERROR : Gagal membuat direktori database : {e}")
            sys.exit(1)

    def connect(self):
        try:
            return sqlite3.connect(str(self.db_path))
        except sqlite3.Error as e:
            print(f"ERROR : Gagal terhubung ke database : {e}")
            return None

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
