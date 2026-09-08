from pathlib import Path

from sqlalchemy import text

from db.connection import engine, test_connection


def run_schema() -> None:
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")

    with engine.begin() as conn:
        conn.execute(text(sql))
        conn.execute(text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='provider_profile'
                      AND column_name='years_of_experience'
                ) AND NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='provider_profile'
                      AND column_name='years_experience'
                ) THEN
                    ALTER TABLE provider_profile RENAME COLUMN years_of_experience TO years_experience;
                END IF;
            END
            $$;
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS notification_preferences TEXT,
            ADD COLUMN IF NOT EXISTS profile_preferences TEXT;
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(255) NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
                read_at TIMESTAMP WITHOUT TIME ZONE
            );
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                sender_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                recipient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                body TEXT NOT NULL,
                created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
                read_at TIMESTAMP WITHOUT TIME ZONE
            );
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE notifications
            ADD COLUMN IF NOT EXISTS read_at TIMESTAMP WITHOUT TIME ZONE;
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE patient_profile
            ADD COLUMN IF NOT EXISTS user_id INTEGER;
            """
        ))
        conn.execute(text(
            """
            DO $$
            DECLARE
                duplicate_user_ids INTEGER[];
                duplicate_user_id INTEGER;
                canonical_id INTEGER;
            BEGIN
                SELECT array_agg(user_id) INTO duplicate_user_ids
                FROM (
                    SELECT user_id
                    FROM patient_profile
                    GROUP BY user_id
                    HAVING COUNT(*) > 1
                ) duplicates;

                IF duplicate_user_ids IS NOT NULL THEN
                    FOREACH duplicate_user_id IN ARRAY duplicate_user_ids LOOP
                        SELECT MIN(id) INTO canonical_id
                        FROM patient_profile
                        WHERE user_id = duplicate_user_id;

                        UPDATE medical_history SET patient_id = canonical_id
                        WHERE patient_id IN (SELECT id FROM patient_profile WHERE user_id = duplicate_user_id AND id <> canonical_id);
                        UPDATE patient_symptoms SET patient_id = canonical_id
                        WHERE patient_id IN (SELECT id FROM patient_profile WHERE user_id = duplicate_user_id AND id <> canonical_id);
                        UPDATE disease_predictions SET patient_id = canonical_id
                        WHERE patient_id IN (SELECT id FROM patient_profile WHERE user_id = duplicate_user_id AND id <> canonical_id);
                        UPDATE risk_assessment SET patient_id = canonical_id
                        WHERE patient_id IN (SELECT id FROM patient_profile WHERE user_id = duplicate_user_id AND id <> canonical_id);
                        UPDATE recommendations SET patient_id = canonical_id
                        WHERE patient_id IN (SELECT id FROM patient_profile WHERE user_id = duplicate_user_id AND id <> canonical_id);
                        UPDATE reports SET patient_id = canonical_id
                        WHERE patient_id IN (SELECT id FROM patient_profile WHERE user_id = duplicate_user_id AND id <> canonical_id);

                        DELETE FROM patient_profile
                        WHERE user_id = duplicate_user_id AND id <> canonical_id;
                    END LOOP;
                END IF;
            END $$;
            """
        ))
        conn.execute(text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_patient_profile_user_id
            ON patient_profile (user_id);
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE patient_profile
            ADD COLUMN IF NOT EXISTS bmi FLOAT,
            ADD COLUMN IF NOT EXISTS dob DATE,
            ADD COLUMN IF NOT EXISTS emergency_contact VARCHAR(50),
            ADD COLUMN IF NOT EXISTS existing_conditions TEXT,
            ADD COLUMN IF NOT EXISTS allergies TEXT,
            ADD COLUMN IF NOT EXISTS profile_picture_url VARCHAR(255);
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE provider_profile
            ADD COLUMN IF NOT EXISTS profile_picture_url VARCHAR(255),
            ADD COLUMN IF NOT EXISTS availability VARCHAR(255);
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE disease_predictions
            ADD COLUMN IF NOT EXISTS model_info VARCHAR(255) DEFAULT 'MedAssist AI v1',
            ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'pending',
            ADD COLUMN IF NOT EXISTS provider_feedback VARCHAR(50) DEFAULT 'pending',
            ADD COLUMN IF NOT EXISTS provider_comments TEXT,
            ADD COLUMN IF NOT EXISTS feedback_date TIMESTAMP WITHOUT TIME ZONE;
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE recommendations
            ADD COLUMN IF NOT EXISTS prediction_id INTEGER REFERENCES disease_predictions(id) ON DELETE SET NULL;
            ALTER TABLE reports
            ADD COLUMN IF NOT EXISTS prediction_id INTEGER REFERENCES disease_predictions(id) ON DELETE SET NULL;
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE risk_assessment
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW();
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE medical_history
            ADD COLUMN IF NOT EXISTS surgery VARCHAR(255),
            ADD COLUMN IF NOT EXISTS medications TEXT,
            ADD COLUMN IF NOT EXISTS allergies TEXT,
            ADD COLUMN IF NOT EXISTS family_history TEXT,
            ADD COLUMN IF NOT EXISTS ongoing_treatment TEXT;
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE recommendations
            ADD COLUMN IF NOT EXISTS priority VARCHAR(50),
            ADD COLUMN IF NOT EXISTS recommendation_type VARCHAR(100),
            ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'pending',
            ADD COLUMN IF NOT EXISTS ai_generated VARCHAR(20) DEFAULT 'yes',
            ADD COLUMN IF NOT EXISTS provider_comments TEXT,
            ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP WITHOUT TIME ZONE;
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE patient_symptoms
            ADD COLUMN IF NOT EXISTS duration VARCHAR(50),
            ADD COLUMN IF NOT EXISTS frequency VARCHAR(50),
            ADD COLUMN IF NOT EXISTS notes TEXT;
            """
        ))
        conn.execute(text(
            """
            ALTER TABLE reports
            ADD COLUMN IF NOT EXISTS report_type VARCHAR(100),
            ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'pending',
            ADD COLUMN IF NOT EXISTS report_url TEXT,
            ADD COLUMN IF NOT EXISTS symptoms TEXT,
            ADD COLUMN IF NOT EXISTS prediction_date TIMESTAMP WITHOUT TIME ZONE,
            ADD COLUMN IF NOT EXISTS predicted_disease VARCHAR(255),
            ADD COLUMN IF NOT EXISTS confidence_score FLOAT,
            ADD COLUMN IF NOT EXISTS risk_assessment TEXT,
            ADD COLUMN IF NOT EXISTS provider_status VARCHAR(50) DEFAULT 'pending',
            ADD COLUMN IF NOT EXISTS provider_comments TEXT,
            ADD COLUMN IF NOT EXISTS recommendations TEXT;
            """
        ))
        conn.execute(text(
            """
            UPDATE disease_predictions
            SET status = CASE lower(trim(COALESCE(status, 'pending')))
                WHEN 'accept' THEN 'approved'
                WHEN 'approve' THEN 'approved'
                WHEN 'reject' THEN 'rejected'
                WHEN 'approved' THEN 'approved'
                WHEN 'rejected' THEN 'rejected'
                ELSE 'pending'
            END,
            provider_feedback = CASE lower(trim(COALESCE(provider_feedback, 'pending')))
                WHEN 'accept' THEN 'approved'
                WHEN 'approve' THEN 'approved'
                WHEN 'reject' THEN 'rejected'
                WHEN 'approved' THEN 'approved'
                WHEN 'rejected' THEN 'rejected'
                ELSE 'pending'
            END;

            UPDATE reports
            SET status = CASE lower(trim(COALESCE(status, 'pending')))
                WHEN 'approved' THEN 'approved'
                WHEN 'rejected' THEN 'rejected'
                ELSE 'pending'
            END,
            provider_status = CASE lower(trim(COALESCE(provider_status, status, 'pending')))
                WHEN 'accept' THEN 'approved'
                WHEN 'approve' THEN 'approved'
                WHEN 'approved' THEN 'approved'
                WHEN 'reject' THEN 'rejected'
                WHEN 'rejected' THEN 'rejected'
                ELSE 'pending'
            END;
            """
        ))


def main() -> None:
    test_connection()
    run_schema()
    print("PostgreSQL connection successful and schema initialized.")


if __name__ == "__main__":
    main()
