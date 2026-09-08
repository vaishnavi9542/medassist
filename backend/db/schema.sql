CREATE SCHEMA IF NOT EXISTS medassist;
SET search_path TO medassist;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) NOT NULL,
    phone VARCHAR(15),
    notification_preferences TEXT,
    profile_preferences TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    read_at TIMESTAMP WITHOUT TIME ZONE
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    sender_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recipient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    body TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    read_at TIMESTAMP WITHOUT TIME ZONE
);

CREATE TABLE IF NOT EXISTS email_confirmations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(128) UNIQUE NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(128) UNIQUE NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS patient_profile (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    age INTEGER,
    gender VARCHAR(50),
    blood_group VARCHAR(10),
    height FLOAT,
    weight FLOAT,
    bmi FLOAT,
    emergency_contact VARCHAR(50),
    existing_conditions TEXT,
    allergies TEXT,
    dob DATE,
    profile_picture_url VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS provider_profile (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    hospital_name VARCHAR(255),
    specialization VARCHAR(255),
    license_number VARCHAR(100),
    years_experience INTEGER,
    qualification VARCHAR(255),
    department VARCHAR(255),
    profile_picture_url VARCHAR(255),
    availability VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS medical_history (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patient_profile(id) ON DELETE CASCADE,
    disease VARCHAR(255) NOT NULL,
    diagnosed_date DATE,
    treatment TEXT,
    status VARCHAR(100),
    surgery VARCHAR(255),
    medications TEXT,
    allergies TEXT,
    family_history TEXT,
    ongoing_treatment TEXT
);

CREATE TABLE IF NOT EXISTS symptoms (
    id SERIAL PRIMARY KEY,
    symptom_name VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS patient_symptoms (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patient_profile(id) ON DELETE CASCADE,
    symptom_id INTEGER NOT NULL REFERENCES symptoms(id) ON DELETE CASCADE,
    severity INTEGER,
    duration VARCHAR(50),
    frequency VARCHAR(50),
    notes TEXT,
    entered_date TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS disease_predictions (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patient_profile(id) ON DELETE CASCADE,
    predicted_disease VARCHAR(255) NOT NULL,
    confidence FLOAT,
    prediction_date TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    model_info VARCHAR(255) DEFAULT 'MedAssist AI v1',
    status VARCHAR(50) DEFAULT 'pending',
    provider_feedback VARCHAR(50) DEFAULT 'pending',
    provider_comments TEXT,
    feedback_date TIMESTAMP WITHOUT TIME ZONE
);

CREATE TABLE IF NOT EXISTS risk_assessment (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patient_profile(id) ON DELETE CASCADE,
    risk_level VARCHAR(100),
    score FLOAT,
    remarks TEXT
    ,created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patient_profile(id) ON DELETE CASCADE,
    prediction_id INTEGER REFERENCES disease_predictions(id) ON DELETE SET NULL,
    recommendation TEXT,
    medicine VARCHAR(255),
    priority VARCHAR(50),
    recommendation_type VARCHAR(100),
    status VARCHAR(50) DEFAULT 'pending',
    ai_generated VARCHAR(20) DEFAULT 'yes',
    provider_comments TEXT,
    reviewed_at TIMESTAMP WITHOUT TIME ZONE,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patient_profile(id) ON DELETE CASCADE,
    prediction_id INTEGER REFERENCES disease_predictions(id) ON DELETE SET NULL,
    report_name VARCHAR(255),
    report_type VARCHAR(100),
    status VARCHAR(50) DEFAULT 'pending',
    report_url TEXT,
    generated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    prediction_date TIMESTAMP WITHOUT TIME ZONE,
    symptoms TEXT,
    predicted_disease VARCHAR(255),
    confidence_score FLOAT,
    risk_assessment TEXT,
    provider_status VARCHAR(50) DEFAULT 'pending',
    provider_comments TEXT,
    recommendations TEXT
);
