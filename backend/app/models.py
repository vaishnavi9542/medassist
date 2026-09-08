from sqlalchemy import Column, Integer, String, Text, DateTime, func, UniqueConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy import Float, ForeignKey, Date

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)
    phone = Column(String(15))
    notification_preferences = Column(Text)
    profile_preferences = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class EmailConfirmation(Base):
    __tablename__ = 'email_confirmations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    token = Column(String(128), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class PatientProfile(Base):
    __tablename__ = 'patient_profile'
    __table_args__ = (UniqueConstraint('user_id', name='uq_patient_profile_user_id'),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    age = Column(Integer)
    gender = Column(String(50))
    blood_group = Column(String(10))
    height = Column(Float)
    weight = Column(Float)
    bmi = Column(Float)
    emergency_contact = Column(String(50))
    existing_conditions = Column(Text)
    allergies = Column(Text)
    dob = Column(Date)
    profile_picture_url = Column(String(255))


class ProviderProfile(Base):
    __tablename__ = 'provider_profile'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    hospital_name = Column(String(255))
    specialization = Column(String(255))
    license_number = Column(String(100))
    years_experience = Column(Integer)
    qualification = Column(String(255))
    department = Column(String(255))
    profile_picture_url = Column(String(255))
    availability = Column(String(255))


class ApiToken(Base):
    __tablename__ = 'api_tokens'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    token = Column(String(128), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
class Notification(Base):
    __tablename__ = 'notifications'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    read_at = Column(DateTime)


class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sender_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    recipient_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    read_at = Column(DateTime)


class MedicalHistory(Base):
    __tablename__ = 'medical_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patient_profile.id', ondelete='CASCADE'), nullable=False)
    disease = Column(String(255), nullable=False)
    diagnosed_date = Column(Date)
    treatment = Column(Text)
    status = Column(String(100))
    surgery = Column(String(255))
    medications = Column(Text)
    allergies = Column(Text)
    family_history = Column(Text)
    ongoing_treatment = Column(Text)


class Symptom(Base):
    __tablename__ = 'symptoms'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symptom_name = Column(String(255), nullable=False)


class PatientSymptom(Base):
    __tablename__ = 'patient_symptoms'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patient_profile.id', ondelete='CASCADE'), nullable=False)
    symptom_id = Column(Integer, ForeignKey('symptoms.id', ondelete='CASCADE'), nullable=False)
    severity = Column(Integer)
    duration = Column(String(50))
    frequency = Column(String(50))
    notes = Column(Text)
    entered_date = Column(DateTime, server_default=func.now())


class DiseasePrediction(Base):
    __tablename__ = 'disease_predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patient_profile.id', ondelete='CASCADE'), nullable=False)
    predicted_disease = Column(String(255), nullable=False)
    confidence = Column(Float)
    prediction_date = Column(DateTime, server_default=func.now())
    model_info = Column(String(255), default='MedAssist AI v1')
    status = Column(String(50), default='pending')
    provider_feedback = Column(String(50), default='pending')
    provider_comments = Column(Text)
    feedback_date = Column(DateTime)


class RiskAssessment(Base):
    __tablename__ = 'risk_assessment'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patient_profile.id', ondelete='CASCADE'), nullable=False)
    risk_level = Column(String(100))
    score = Column(Float)
    remarks = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class Recommendation(Base):
    __tablename__ = 'recommendations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patient_profile.id', ondelete='CASCADE'), nullable=False)
    prediction_id = Column(Integer, ForeignKey('disease_predictions.id', ondelete='SET NULL'))
    recommendation = Column(Text)
    medicine = Column(String(255))
    priority = Column(String(50))
    recommendation_type = Column(String(100))
    status = Column(String(50), default='pending')
    ai_generated = Column(String(20), default='yes')
    provider_comments = Column(Text)
    reviewed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())


class Report(Base):
    __tablename__ = 'reports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('patient_profile.id', ondelete='CASCADE'), nullable=False)
    prediction_id = Column(Integer, ForeignKey('disease_predictions.id', ondelete='SET NULL'))
    report_name = Column(String(255))
    report_type = Column(String(100))
    status = Column(String(50), default='pending')
    report_url = Column(Text)
    generated_at = Column(DateTime, server_default=func.now())
    prediction_date = Column(DateTime)
    symptoms = Column(Text)
    predicted_disease = Column(String(255))
    confidence_score = Column(Float)
    risk_assessment = Column(Text)
    provider_status = Column(String(50), default='pending')
    provider_comments = Column(Text)
    recommendations = Column(Text)
