CREATE TABLE IF NOT EXISTS service_survey (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
    is_good BOOLEAN
);
