"""
Storing/retrieving hole detections.
"""
import sqlite3
from pathlib import Path
import sys

class DatabaseManager:
    def __init__(self, db_path: str = 'database/hole.db', schema_path: str = 'src/database/schema.sql'):
        """Initialize db manajer needs"""
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


    def insert_detection(self, image_filename: str, time_detection: str, latitude: float, longitude: float, prediction_label: str, confidence: float, is_validate: str) -> int:
        """Store hole detection and return -1 if it fails"""
        conn = self.connect()
        if conn is None:
            return -1
        
        query ="""
        INSERT INTO holes (image_filename, time_detection, latitude, longitude, prediction_label, confidence, is_validate)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """

        try:
            cursor = conn.cursor()
            cursor.execute(query, (image_filename, time_detection, latitude, longitude, prediction_label, confidence, is_validate))
            conn.commit()
            return cursor.lastrowid  #Get the new data ID that stored
        except sqlite3.IntegrityError as e:
            """Special error if there are restrictions that have been violated"""
            print(f"ERROR (Integrity): Gagal menyisipkan data. Cek batasan skema: {e}")
            return -1
        except sqlite3.Error as e:
            print(f"ERROR (SQL): Gagal menyisipkan data: {e}")
            return -1
        finally:
            conn.close()

    def get_all_detections(self) -> list:
        """Retrieve all detections fromm table HOLES"""
        conn = self.connect()
        if conn is None:
            return []
        
        query = "SELECT * FROM holes"
        try : 
            """Use row factory so that it can be more easily to read"""
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            results = [dict(row) for row in cursor.fetchall()]  #change query result to list of dicts
            return results
        except sqlite3.Error as e:
            print(f"ERROR : Gagal mengambil data dari database : {e}")
            return []
        finally:
            conn.close()

    def get_detections_by_location(self, location: dict, radius: float) -> list:
        """Retrieve detections near location"""
        raise NotImplementedError
