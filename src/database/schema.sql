CREATE TABLE IF NOT EXISTS holes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_filename TEXT NOT NULL,
    time_detection TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    prediction_label TEXT NOT NULL,
    confidence REAL TEXT NOT NULL,    --(0-1)
    valildate_human TEXT            --(yes, no)
)