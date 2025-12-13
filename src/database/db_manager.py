"""
Storing/retrieving hole detections.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
import math


class DatabaseManager:
    def __init__(self, db_path: str = 'potholes.db'):
        self.db_path = db_path
        self.conn = None
        self.create_connection()
        self.create_tables()

    def create_connection(self):
        """Create database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"❌ Database connection error: {e}")

    def create_tables(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # Detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                latitude REAL,
                longitude REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Summary statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_detections INTEGER DEFAULT 0,
                holes_found INTEGER DEFAULT 0,
                no_holes_found INTEGER DEFAULT 0,
                avg_confidence REAL
            )
        ''')
        
        self.conn.commit()
        print("✅ Database tables initialized")

    def insert_detection(self, image_path: str, class_name: str, confidence: float, 
                        latitude: float = None, longitude: float = None) -> int:
        """Store detection result"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (image_path, class_name, confidence, latitude, longitude)
            VALUES (?, ?, ?, ?, ?)
        ''', (image_path, class_name, confidence, latitude, longitude))
        
        self.conn.commit()
        detection_id = cursor.lastrowid
        print(f"✅ Detection saved (ID: {detection_id})")
        return detection_id

    def get_all_detections(self) -> list:
        """Retrieve all detections"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM detections ORDER BY timestamp DESC')
        
        columns = [description[0] for description in cursor.description]
        detections = []
        
        for row in cursor.fetchall():
            detections.append(dict(zip(columns, row)))
        
        return detections

    def get_detections_by_location(self, latitude: float, longitude: float, 
                                   radius_km: float = 1.0) -> list:
        """Retrieve detections near location (using Haversine formula)"""
        cursor = self.conn.cursor()
        
        # Haversine formula to calculate distance
        cursor.execute('''
            SELECT * FROM detections
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            AND (
                6371 * 2 * ASIN(
                    SQRT(
                        POWER(SIN(RADIANS(? - latitude) / 2), 2) +
                        COS(RADIANS(latitude)) * COS(RADIANS(?)) *
                        POWER(SIN(RADIANS(? - longitude) / 2), 2)
                    )
                )
            ) <= ?
            ORDER BY timestamp DESC
        ''', (latitude, latitude, longitude, radius_km))
        
        columns = [description[0] for description in cursor.description]
        detections = []
        
        for row in cursor.fetchall():
            detections.append(dict(zip(columns, row)))
        
        return detections

    def get_detections_by_class(self, class_name: str) -> list:
        """Get all detections of a specific class"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT * FROM detections WHERE class_name = ? ORDER BY timestamp DESC',
            (class_name,)
        )
        
        columns = [description[0] for description in cursor.description]
        detections = []
        
        for row in cursor.fetchall():
            detections.append(dict(zip(columns, row)))
        
        return detections

    def get_statistics(self) -> dict:
        """Get overall statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM detections')
        total_detections = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM detections WHERE class_name = ?', ('hole',))
        holes_found = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM detections WHERE class_name = ?', ('no_hole',))
        no_holes_found = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(confidence) FROM detections')
        avg_confidence = cursor.fetchone()[0] or 0
        
        return {
            'total_detections': total_detections,
            'holes_found': holes_found,
            'no_holes_found': no_holes_found,
            'avg_confidence': avg_confidence,
            'hole_percentage': (holes_found / total_detections * 100) if total_detections > 0 else 0
        }

    def delete_detection(self, detection_id: int) -> bool:
        """Delete a detection record"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")

    def __del__(self):
        """Ensure connection is closed"""
        self.close()
