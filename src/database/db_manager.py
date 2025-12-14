"""
Storing/retrieving hole detections.
"""
import sqlite3
from pathlib import Path
import sys
import math

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance Great-Cycle between 2 points in the earth (km).
    This function will be applied to sqlite"""
    R = 6371  #radius earth (km)

    #change degress to radian
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    
    #difference
    dlon = lon2_rad - lon1_rad     
    dlat = lat2_rad - lat1_rad

    #Haversine Formula (a = alpha, c = central angle of earth)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  #distance (km)


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
            conn = sqlite3.connect(str(self.db_path))
        
            #register function haversine to db
            conn.create_function("HAVERSINE_DIST", 4, haversine)
            return conn
        
        except sqlite3.Error as e:
            print(f"ERROR : Gagal terhubung / register Haversine : {e}")
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


    def insert_detection(self, image_filename: str, time_detection: str, latitude: float, longitude: float, prediction_label: str, confidence: float, is_validate: str = None) -> int:
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


    def get_detections_by_location(self, center_lat: float, center_lon: float, radius_km: float) -> list:
        """Retrieve detections near location"""
        conn = self.connect()
        if conn is None:
            return []
        
        query = f"""
        SELECT 
            *,
            HAVERSINE_DIST(holes.latitude, holes.longitude, ?, ?) AS distance_km
        FROM 
            holes
        HAVING 
            distance_km <= ?
        ORDER BY 
            distance_km ASC;
        """
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            #args: center_lat, center_lon (for Haversine_Dist), and radius_km (for having)
            params = (center_lat, center_lon, radius_km)

            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            
            print(f"SUCCESS: Ditemukan {len(results)} deteksi dalam radius {radius_km} km.")
            return results
        except sqlite3.Error as e:
            print(f"ERROR: Gagal mengambil data geografis: {e}")
            return []
        finally:
            conn.close()


    def get_detections_by_label(self, label: str) -> list :
        """Retrieve all detections based on predict label(e.g : medium hole)"""
        conn = self.connect()
        if conn is None:
            return[]
        query = """
        SELECT * FROM holes 
        WHERE prediction_label = ? 
        """ # Query using placeholder (?) for safety sql injection

        try:
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()

            cursor.execute(query, (label,))

            results = [dict(row) for row in cursor.fetchall()]
            
            print(f"SUCCESS: Ditemukan {len(results)} deteksi dengan label '{label}'.")
            return results
        except sqlite3.Error as e:
            print(f"ERROR: Gagal mengambil data berdasarkan label: {e}")
            return []
        finally:
            conn.close()