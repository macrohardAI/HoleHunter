CREATE TABLE IF NOT EXISTS holes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_filename TEXT NOT NULL,
    time_detection TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    prediction_label TEXT NOT NULL,   --(hole or not hole, small, large)
    confidence REAL TEXT NOT NULL,    --(0-1)
    valildate_human TEXT            --(yes, no)
)

--indeks utk cari lokasi hole faster
CREATE INDEX IF NOT EXISTS location
ON detections(latitude,longitude);

--indeks utk cari label
CREATE INDEX IF NOT EXISTS label
ON detections(prediction_label);