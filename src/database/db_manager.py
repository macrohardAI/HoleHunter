"""
Storing/retrieving hole detections.
"""
import sqlite3
from pathlib import Path
import sys

class DatabaseManager:
    def __init__(self, db_path: str = 'database/hole.db', schema_path: str = 'src/database/schema.sql'):
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path)
        try:
            self.db_path.parent.mkdir(parents = True, exist_ok = True)
        except Exception as e:
            print(f"ERROR : Gagal membuat direktori database : {e}")
            sys.exit(1)

    def connect(self):
        try:
            return sqlite3.connect(str(self.db_path))
        except sqlite3.Error as e:
            print(f"ERROR : Gagal terhubung ke database {self.db_path} : {e}")
            return None

    def create_tables(self):
        """Initialize database schema"""
        conn = self.connect()
        if conn is None:
            return
        
        try:
            if not self.schema_path.exists():
                print(f"ERROR : File tidak ditemukan : {self.schema_path}")
                return
            #read all sql scripts
            sql_script = self.schema_path.read_text()

            cursor = conn.cursor()
            cursor.executescript(sql_script)
            conn.commit()
            print(f"Database schema dibuat/di update di : {self.db_path} ")

        except sqlite3.Error as e:
            print(f"ERROR: Gagal mengeksekusi skema SQL. Pesan: {e}")
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan saat membuat tabel: {e}")
        finally:
            conn.close()


    def insert_detection(self, image_path: str, confidence: float, gps: dict) -> int:
        """Store detection"""
        raise NotImplementedError

    def get_all_detections(self) -> list:
        """Retrieve all detections"""
        raise NotImplementedError

    def get_detections_by_location(self, location: dict, radius: float) -> list:
        """Retrieve detections near location"""
        raise NotImplementedError
