SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    status TEXT NOT NULL,
    pipeline_mode TEXT NOT NULL DEFAULT 'training',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    original_name TEXT NOT NULL,
    source_group TEXT NOT NULL,
    frame_index INTEGER,
    timestamp_seconds REAL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    blur_score REAL,
    brightness REAL,
    reflection_ratio REAL,
    perceptual_hash TEXT,
    selection_status TEXT NOT NULL DEFAULT 'pending',
    selection_reasons_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class_name TEXT NOT NULL,
    raw_class_name TEXT,
    confidence REAL NOT NULL,
    bbox_json TEXT NOT NULL,
    status TEXT NOT NULL,
    active_learning_reasons_json TEXT NOT NULL,
    track_id TEXT,
    track_score REAL,
    track_length INTEGER,
    is_track_keyframe INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vlm_analyses (
    prediction_id TEXT PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sam2_masks (
    prediction_id TEXT PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    polygon_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    mask_path TEXT,
    is_bbox_fallback INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_decisions (
    prediction_id TEXT PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS error_bank (
    id TEXT PRIMARY KEY,
    correction_id TEXT UNIQUE NOT NULL REFERENCES corrections(id) ON DELETE CASCADE,
    prediction_id TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    record_json TEXT NOT NULL,
    type_erreur TEXT,
    classe_finale TEXT,
    ajouter_finetuning INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_job ON images(job_id);
CREATE INDEX IF NOT EXISTS idx_images_group ON images(source_group, frame_index);
CREATE INDEX IF NOT EXISTS idx_predictions_image ON predictions(image_id);
CREATE INDEX IF NOT EXISTS idx_corrections_prediction ON corrections(prediction_id, created_at);
CREATE INDEX IF NOT EXISTS idx_error_type ON error_bank(type_erreur, classe_finale);
"""
