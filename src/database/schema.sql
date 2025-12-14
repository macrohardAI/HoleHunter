CREATE TABLE IF NOT EXISTS holes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_filename TEXT NOT NULL,
    time_detection TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    prediction_label TEXT NOT NULL,   --(hole or not hole, small, large)
    confidence REAL TEXT NOT NULL,    --(0-1)
    is_valildate TEXT            -- by human (yes or no)
)

--indeks utk cari lokasi hole faster
CREATE INDEX IF NOT EXISTS idx_location
ON holes(latitude,longitude);

--indeks utk cari label
CREATE INDEX IF NOT EXISTS idx_label
ON holes(prediction_label);