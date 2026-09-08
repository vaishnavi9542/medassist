import base64
import io
import json
import os
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    UserCreate,
    UserOut,
    PatientProfileUpdate,
    MedicalHistoryCreate,
    MedicalHistoryUpdate,
    PatientSymptomCreate,
    PredictionRequest,
    PredictionFeedbackRequest,
    RiskAssessmentRequest,
    SettingsUpdate,
    ProviderRecommendationCreate,
    ProviderReportCreate,
    RecommendationReviewRequest,
)
from app.crud import create_user
from app.crud import pwd_ctx
from app.models import User
from app.recommendation_engine import generate_recommendations_for_prediction
from app.models import EmailConfirmation
from app.models import ApiToken, PatientProfile, ProviderProfile, MedicalHistory, PatientSymptom, DiseasePrediction, RiskAssessment, Recommendation, Report, Symptom
from app.models import ApiToken, ChatMessage, Notification, PatientProfile, ProviderProfile, MedicalHistory, PatientSymptom, DiseasePrediction, RiskAssessment, Recommendation, Report, Symptom
from db.connection import get_db_session
from db.init_db import run_schema

app = FastAPI(title="MedAssist API")

# @app.on_event("startup")
# def initialize_database():
#     run_schema()

configured_origins = [
    origin.strip()
    for origin in os.getenv('CORS_ORIGINS', os.getenv('FRONTEND_URL', '')).split(',')
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins + [
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_DIR = Path(__file__).resolve().parents[1] / "ml" / "models"
BEST_MODEL_KERAS = MODEL_DIR / "best_model.keras"
BEST_MODEL_PICKLE = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

DEFAULT_MODEL_INPUT = {
    "gender": "other",
    "age_exact": 30,
    "smoking": "no",
    "alcohol": "no",
    "exercise_level": "medium",
    "diet_quality": "average",
    "family_history": "no",
    "weight_kg": 70,
    "height_cm": 170,
    "bmi": 24.0,
    "bp": "normal",
    "blood_sugar": "normal",
    "heart_rate": 72,
    "cholesterol": 180,
}

SYMPTOM_FEATURES = [f"symptom_{i}" for i in range(1, 8)]
VALID_APPROVAL_STATUSES = {'pending', 'approved', 'rejected'}
STATUS_ALIASES = {'accept': 'approved', 'approve': 'approved', 'reject': 'rejected'}


def normalize_approval_status(value: Optional[str], default: str = 'pending') -> str:
    normalized = str(value or default).strip().lower()
    normalized = STATUS_ALIASES.get(normalized, normalized)
    return normalized if normalized in VALID_APPROVAL_STATUSES else default


def get_prediction_status(prediction: DiseasePrediction) -> str:
    """Resolve status for legacy rows where status and provider feedback differ."""
    feedback_status = normalize_approval_status(prediction.provider_feedback, default='')
    record_status = normalize_approval_status(prediction.status, default='')
    if feedback_status in ('approved', 'rejected'):
        return feedback_status
    if record_status in ('approved', 'rejected'):
        return record_status
    return 'pending'


def get_prediction_artifacts() -> Tuple[Any, Any, Any]:
    if not PREPROCESSOR_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError("Preprocessor or label encoder not found in ml/models")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if BEST_MODEL_PICKLE.exists():
        model = joblib.load(BEST_MODEL_PICKLE)
    else:
        raise FileNotFoundError("RandomForest model not found in ml/models")

    return preprocessor, label_encoder, model


def build_prediction_input(symptoms: List[str], profile) -> pd.DataFrame:
    values = DEFAULT_MODEL_INPUT.copy()

    if profile:
        if getattr(profile, "gender", None):
            values["gender"] = profile.gender
        if getattr(profile, "age", None) is not None:
            values["age_exact"] = int(profile.age)
        if getattr(profile, "weight", None) is not None:
            values["weight_kg"] = float(profile.weight)
        if getattr(profile, "height", None) is not None:
            values["height_cm"] = float(profile.height)
        if getattr(profile, "bmi", None) is not None:
            values["bmi"] = float(profile.bmi)

    for index, feature in enumerate(SYMPTOM_FEATURES):
        values[feature] = symptoms[index] if index < len(symptoms) else ""

    return pd.DataFrame([values])


def predict_disease_from_model(symptoms: List[str], profile) -> Tuple[str, float]:
    preprocessor, label_encoder, model = get_prediction_artifacts()
    X_input = build_prediction_input(symptoms, profile)
    X_transformed = preprocessor.transform(X_input)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_transformed)
        class_index = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities, axis=1)[0])
    else:
        prediction = model.predict(X_transformed)
        class_index = int(prediction[0])
        confidence = 1.0

    disease = label_encoder.inverse_transform([class_index])[0]
    return disease, confidence


def build_report_payload(
    patient_id: int,
    symptoms: List[str],
    predicted_disease: str,
    confidence_score: float,
    risk_assessment: str,
    provider_status: str = 'approved',
    provider_comments: str = '',
    recommendations: str = '',
) -> dict:
    normalized_symptoms = [str(item).strip() for item in symptoms if str(item).strip()]
    payload = {
        'patient_id': patient_id,
        'symptoms': normalized_symptoms,
        'predicted_disease': str(predicted_disease).strip(),
        'confidence_score': float(confidence_score) if confidence_score is not None else 0.0,
        'risk_assessment': str(risk_assessment).strip() if risk_assessment is not None else 'No risk assessment available',
        'provider_status': normalize_approval_status(provider_status, default='approved'),
        'provider_comments': str(provider_comments).strip() if provider_comments is not None else '',
        'recommendations': str(recommendations).strip() if recommendations is not None else '',
        'date': datetime.utcnow().date().isoformat(),
    }
    payload['healthcare_advisory'] = build_healthcare_advisory(
        predicted_disease=payload['predicted_disease'],
        risk_assessment=payload['risk_assessment'],
        symptoms=normalized_symptoms,
    )
    return payload


def evaluate_patient_risk(profile, symptoms: Optional[List[str]] = None, predicted_disease: Optional[str] = None, medical_history: Optional[str] = None, lifestyle: Optional[dict] = None) -> dict:
    score = 15.0
    factors = []

    age = getattr(profile, 'age', None)
    if age is not None:
        if age >= 65:
            score += 30
            factors.append('advanced age')
        elif age >= 50:
            score += 18
            factors.append('age-related risk')
        elif age >= 40:
            score += 8
            factors.append('moderate age risk')

    if profile is not None:
        conditions_text = ' '.join([
            str(getattr(profile, 'existing_conditions', '') or ''),
            str(getattr(profile, 'allergies', '') or ''),
            str(medical_history or ''),
        ]).lower()
        chronic_terms = ['diabetes', 'hypertension', 'heart', 'asthma', 'copd', 'cancer', 'lung', 'kidney', 'stroke', 'arthritis', 'obesity']
        if any(term in conditions_text for term in chronic_terms):
            score += 25
            factors.append('chronic conditions')

        bmi = getattr(profile, 'bmi', None)
        if bmi is not None and bmi >= 30:
            score += 12
            factors.append('high BMI')

    symptom_list = [str(item).strip().lower() for item in (symptoms or []) if str(item).strip()]
    severe_symptoms = {
        'shortness of breath', 'chest pain', 'difficulty breathing', 'persistent cough', 'severe fatigue',
        'confusion', 'fainting', 'bluish lips', 'loss of consciousness', 'high fever', 'severe pain',
    }
    if symptom_list:
        if len(symptom_list) >= 5:
            score += 15
            factors.append('multiple symptoms')
        if any(symptom in severe_symptoms for symptom in symptom_list):
            score += 18
            factors.append('severe symptom burden')

    pred_text = (predicted_disease or '').lower()
    high_risk_diseases = ['asthma', 'covid', 'diabetes', 'heart disease', 'stroke', 'lung', 'cancer', 'infection']
    if any(term in pred_text for term in high_risk_diseases):
        score += 12
        factors.append('predicted disease risk')

    lifestyle = lifestyle or {}
    if isinstance(lifestyle, dict):
        smoking = str(lifestyle.get('smoking', '')).lower()
        alcohol = str(lifestyle.get('alcohol', '')).lower()
        exercise = str(lifestyle.get('exercise', '')).lower()
        if smoking in {'yes', 'current', 'heavy'}:
            score += 16
            factors.append('smoking')
        if alcohol in {'heavy', 'frequent', 'yes'}:
            score += 10
            factors.append('alcohol use')
        if exercise in {'low', 'sedentary', 'none', 'minimal'}:
            score += 8
            factors.append('low activity')

    if not factors:
        factors.append('general wellness factors')

    score = round(min(score, 100), 1)
    if score >= 70:
        level = 'High'
        warning = 'Consult a healthcare provider promptly and seek urgent care if symptoms worsen.'
    elif score >= 40:
        level = 'Moderate'
        warning = 'Monitor symptoms closely and follow up with a clinician if they change.'
    else:
        level = 'Low'
        warning = 'Continue routine monitoring and maintain follow-up health habits.'

    remarks = f"{level} risk: {', '.join(factors)}."
    return {
        'risk_level': level,
        'score': score,
        'remarks': remarks,
        'factors': sorted(set(factors), key=lambda item: item),
        'warning': warning,
    }


def generate_preliminary_recommendations(symptoms: List[str], predicted_disease: str) -> List[dict]:
    disease = (predicted_disease or '').strip().lower()
    symptom_text = ' '.join((symptom or '').strip().lower() for symptom in symptoms if str(symptom).strip())
    recommendations = []

    if 'flu' in disease or 'influenza' in disease or 'fever' in symptom_text or 'cough' in symptom_text:
        recommendations.append({
            'recommendation': 'Rest, stay hydrated, and monitor fever and breathing symptoms closely.',
            'medicine': 'Hydration support and symptomatic relief',
            'priority': 'high',
            'recommendation_type': 'AI-generated',
        })
    if 'covid' in disease:
        recommendations.append({
            'recommendation': 'Follow isolation guidance and consider medical evaluation if breathing worsens or oxygen levels drop.',
            'medicine': 'Clinician-guided testing and monitoring',
            'priority': 'high',
            'recommendation_type': 'AI-generated',
        })
    if 'diabetes' in disease or 'high blood sugar' in disease:
        recommendations.append({
            'recommendation': 'Monitor glucose regularly and maintain meal timing to stabilize blood sugar levels.',
            'medicine': 'Blood glucose monitoring plan',
            'priority': 'high',
            'recommendation_type': 'AI-generated',
        })
    if 'asthma' in disease or 'shortness of breath' in symptom_text:
        recommendations.append({
            'recommendation': 'Avoid known triggers, use prescribed inhaler regimen, and seek emergency care for severe breathing difficulty.',
            'medicine': 'Asthma control regimen',
            'priority': 'high',
            'recommendation_type': 'AI-generated',
        })
    if not recommendations:
        recommendations.append({
            'recommendation': 'Follow up with a healthcare professional to validate the diagnosis and continue appropriate monitoring.',
            'medicine': 'Clinical follow-up plan',
            'priority': 'medium',
            'recommendation_type': 'AI-generated',
        })
    return recommendations


def build_healthcare_advisory(predicted_disease: str, risk_assessment: str, symptoms: List[str]) -> dict:
    disease = str(predicted_disease or '').strip().lower()
    symptom_list = [str(item).strip().lower() for item in (symptoms or []) if str(item).strip()]
    risk_text = str(risk_assessment or '').strip()
    risk_level = risk_text.split(' risk', 1)[0].strip().lower() if ' risk' in risk_text.lower() else ''

    preventive = [
        'Keep a current list of symptoms, medicines, allergies, and medical conditions for your clinician.',
        'Use reliable hand hygiene, avoid sharing personal items, and rest when you feel unwell.',
    ]
    lifestyle = [
        'Drink water regularly, choose balanced meals, and prioritise consistent sleep.',
        'Use gentle activity as tolerated and stop if symptoms worsen.',
    ]
    follow_up = 'Arrange a follow-up with a qualified healthcare professional to confirm this AI-assisted assessment.'
    seek_care = 'Seek urgent medical care for severe or rapidly worsening symptoms.'

    if any(term in disease for term in ('malaria', 'dengue', 'infection')):
        preventive.insert(0, 'Use insect protection and avoid mosquito exposure where mosquito-borne illness is possible.')
        lifestyle[0] = 'Prioritise fluids and rest; ask a clinician about safe fever or pain relief.'
        follow_up = 'Arrange prompt clinical testing and follow-up; do not self-treat based on the prediction alone.'
    elif any(term in disease for term in ('flu', 'influenza', 'cold')):
        preventive.insert(0, 'Limit close contact while fever or significant respiratory symptoms are present and follow local health guidance.')
        follow_up = 'Follow up if symptoms do not improve, return after improving, or breathing symptoms develop.'
    elif 'covid' in disease:
        preventive.insert(0, 'Follow current local testing, isolation, and masking guidance.')
        follow_up = 'Consider clinician-guided testing and follow-up, especially if you have underlying conditions.'
    elif 'asthma' in disease or any('breath' in symptom for symptom in symptom_list):
        preventive.insert(0, 'Avoid known smoke, dust, and environmental triggers and keep prescribed treatment accessible.')
        seek_care = 'Seek urgent care for severe breathing difficulty, blue lips, confusion, or inability to speak comfortably.'
    elif 'diabetes' in disease or 'blood sugar' in disease:
        preventive.insert(0, 'Monitor glucose as directed and keep scheduled chronic-care appointments.')
        follow_up = 'Arrange clinician follow-up to confirm testing and review a safe monitoring plan.'

    if risk_level == 'high' or any(term in symptom_list for term in ('chest pain', 'fainting', 'confusion', 'bluish lips')):
        seek_care = 'Seek urgent medical care now for severe, rapidly worsening, or emergency warning symptoms.'
        follow_up = 'Contact a healthcare professional promptly for an in-person assessment and personalised care plan.'

    return {
        'preventive_care': preventive,
        'lifestyle_advice': lifestyle,
        'follow_up_guidance': follow_up,
        'when_to_seek_care': seek_care,
        'disclaimer': 'This advisory is educational and does not replace diagnosis or treatment from a qualified healthcare professional.',
    }


def format_report_date(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value))
        return f'{parsed.strftime("%B")} {parsed.day}, {parsed.year}'
    except (TypeError, ValueError):
        return str(value or 'Not available')


def format_confidence(value: Any) -> str:
    try:
        return f'{float(value) * 100:.2f}%'
    except (TypeError, ValueError):
        return 'Not available'


def build_report_text(payload: dict) -> str:
    symptoms = ', '.join(payload.get('symptoms', [])) if isinstance(payload.get('symptoms'), list) else str(payload.get('symptoms', ''))
    return (
        'MEDASSIST AI HEALTH REPORT\n\n'
        f"Date: {format_report_date(payload.get('date'))}\n"
        f"Patient ID: {payload.get('patient_id', 'Unknown')}\n\n"
        'SYMPTOMS\n'
        f'{symptoms or "No symptoms recorded"}\n\n'
        'AI ASSESSMENT\n'
        f"Predicted disease: {payload.get('predicted_disease', 'Unknown')}\n"
        f"Confidence score: {format_confidence(payload.get('confidence_score'))}\n"
        f"Risk assessment: {payload.get('risk_assessment', 'No risk assessment available')}\n\n"
        'PROVIDER REVIEW\n'
        f"Approval status: {str(payload.get('provider_status', 'pending')).title()}\n"
        f"Provider comments: {payload.get('provider_comments') or 'No provider comments.'}\n\n"
        'RECOMMENDATIONS\n'
        f"{payload.get('recommendations') or 'No recommendations available.'}\n\n"
        'Generated by MedAssist AI\n'
    )


def build_report_pdf(report_payload: dict) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.65 * inch,
        title='MedAssist AI Health Report',
        author='MedAssist AI',
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=22, leading=27, textColor=colors.HexColor('#062b4f'), alignment=TA_CENTER, spaceAfter=4)
    subtitle_style = ParagraphStyle('ReportSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=11, leading=14, textColor=colors.HexColor('#475569'), alignment=TA_CENTER, spaceAfter=18)
    section_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.HexColor('#0b79ff'), spaceBefore=8, spaceAfter=8)
    body_style = ParagraphStyle('ReportBody', parent=styles['BodyText'], fontName='Helvetica', fontSize=10.5, leading=15, textColor=colors.HexColor('#334155'))
    label_style = ParagraphStyle('ReportLabel', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor('#0f172a'))
    footer_style = ParagraphStyle('ReportFooter', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#64748b'), alignment=TA_CENTER)

    def safe_text(value: Any, fallback: str) -> str:
        text = str(value).strip() if value is not None else ''
        return text or fallback

    symptoms = report_payload.get('symptoms') if isinstance(report_payload.get('symptoms'), list) else []
    symptoms_text = '<br/>'.join(f'• {safe_text(item, "Not available")}' for item in symptoms) or 'No symptoms recorded'
    advisory = report_payload.get('healthcare_advisory') or build_healthcare_advisory(
        report_payload.get('predicted_disease'), report_payload.get('risk_assessment'), symptoms
    )
    advisory_list = lambda key: '<br/>'.join(f'• {safe_text(item, "Not available")}' for item in advisory.get(key, [])) or 'No additional guidance available.'
    story = [
        Paragraph('MEDASSIST AI', title_style),
        Paragraph('Approved Health Report', subtitle_style),
        Paragraph('REPORT INFORMATION', section_style),
        Table([
            [Paragraph('<b>Date</b>', body_style), Paragraph(format_report_date(report_payload.get('date')), body_style)],
            [Paragraph('<b>Patient ID</b>', body_style), Paragraph(safe_text(report_payload.get('patient_id'), 'Unknown'), body_style)],
        ], colWidths=[1.35 * inch, 5.7 * inch], style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fbff')),
            ('BOX', (0, 0), (-1, -1), 0.7, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ])),
        Spacer(1, 12),
        Paragraph('SYMPTOMS', section_style),
        Paragraph(symptoms_text, body_style),
        Paragraph('AI ASSESSMENT', section_style),
        Table([
            [Paragraph('<b>Predicted disease</b>', body_style), Paragraph(safe_text(report_payload.get('predicted_disease'), 'Not available'), body_style)],
            [Paragraph('<b>Confidence</b>', body_style), Paragraph(format_confidence(report_payload.get('confidence_score')), body_style)],
            [Paragraph('<b>Risk assessment</b>', body_style), Paragraph(safe_text(report_payload.get('risk_assessment'), 'No risk assessment available'), body_style)],
        ], colWidths=[1.75 * inch, 5.3 * inch], style=TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.7, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ])),
        Paragraph('PROVIDER REVIEW', section_style),
        Paragraph(f'<b>Approval status:</b> {safe_text(report_payload.get("provider_status"), "Pending").title()}', body_style),
        Spacer(1, 5),
        Paragraph(f'<b>Provider comments:</b> {safe_text(report_payload.get("provider_comments"), "No provider comments.")}', body_style),
        Paragraph('RECOMMENDATIONS', section_style),
        Paragraph(safe_text(report_payload.get('recommendations'), 'No recommendations available.'), body_style),
        Paragraph('HEALTHCARE ADVISORY', section_style),
        Paragraph('<b>Preventive care</b><br/>' + advisory_list('preventive_care'), body_style),
        Spacer(1, 5),
        Paragraph('<b>Lifestyle advice</b><br/>' + advisory_list('lifestyle_advice'), body_style),
        Spacer(1, 5),
        Paragraph(f'<b>Follow-up guidance:</b> {safe_text(advisory.get("follow_up_guidance"), "Arrange follow-up with a healthcare professional.")}', body_style),
        Spacer(1, 5),
        Paragraph(f'<b>When to seek care:</b> {safe_text(advisory.get("when_to_seek_care"), "Seek medical care if symptoms worsen.")}', body_style),
        Spacer(1, 5),
        Paragraph(f'<i>{safe_text(advisory.get("disclaimer"), "This advisory is educational and does not replace professional medical care.")}</i>', footer_style),
        Spacer(1, 28),
        Paragraph('Generated by MedAssist AI', footer_style),
    ]

    document.build(story)
    return buffer.getvalue()


def build_report_download_url(report_payload: dict) -> str:
    encoded_pdf = base64.b64encode(build_report_pdf(report_payload)).decode('ascii')
    return f'data:application/pdf;base64,{encoded_pdf}'


def parse_report_symptoms(value: Optional[str]) -> List[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (TypeError, ValueError):
        pass
    return [item.strip() for item in str(value).split(',') if item.strip()]


def serialize_report(report: Report, include_patient_id: bool = False) -> dict:
    report_url = report.report_url
    if not report_url or not str(report_url).startswith('data:application/pdf;base64,'):
        report_payload = build_report_payload(
            patient_id=report.patient_id,
            symptoms=parse_report_symptoms(report.symptoms),
            predicted_disease=report.predicted_disease or 'Not available',
            confidence_score=report.confidence_score,
            risk_assessment=report.risk_assessment or 'No risk assessment available',
            provider_status=normalize_approval_status(report.provider_status or report.status),
            provider_comments=report.provider_comments or '',
            recommendations=report.recommendations or '',
        )
        if report.generated_at:
            report_payload['date'] = report.generated_at.date().isoformat()
        report_url = build_report_download_url(report_payload)
    result = {
        'id': report.id,
        'prediction_id': report.prediction_id,
        'report_name': report.report_name,
        'report_type': report.report_type,
        'report_url': report_url,
        'generated_at': report.generated_at.isoformat() if report.generated_at else None,
        'prediction_date': report.prediction_date.isoformat() if report.prediction_date else None,
        'symptoms': parse_report_symptoms(report.symptoms),
        'predicted_disease': report.predicted_disease,
        'confidence_score': report.confidence_score,
        'risk_assessment': report.risk_assessment,
        'provider_status': normalize_approval_status(report.provider_status or report.status),
        'provider_comments': report.provider_comments,
        'recommendations': report.recommendations,
        'healthcare_advisory': build_healthcare_advisory(
            report.predicted_disease or '', report.risk_assessment or '', parse_report_symptoms(report.symptoms)
        ),
        'status': normalize_approval_status(report.status or report.provider_status),
    }
    if include_patient_id:
        result['patient_id'] = report.patient_id
    return result


def get_latest_patient_risk(session, patient_id: int) -> Optional[RiskAssessment]:
    return session.execute(
        select(RiskAssessment)
        .where(RiskAssessment.patient_id == patient_id)
        .order_by(RiskAssessment.created_at.desc())
    ).scalar_one_or_none()


def get_patient_symptom_names(session, patient_id: int) -> List[str]:
    rows = session.execute(
        select(PatientSymptom, Symptom)
        .join(Symptom, PatientSymptom.symptom_id == Symptom.id)
        .where(PatientSymptom.patient_id == patient_id)
        .order_by(PatientSymptom.entered_date.desc())
    ).all()
    return [symptom.symptom_name for _, symptom in rows]


def get_patient_recommendation_text(session, patient_id: int, prediction_id: Optional[int] = None) -> str:
    recommendations = session.execute(
        select(Recommendation)
        .where(
            Recommendation.patient_id == patient_id,
            Recommendation.status == 'approved',
            *( [Recommendation.prediction_id == prediction_id] if prediction_id is not None else [] ),
        )
        .order_by(Recommendation.created_at.desc())
    ).scalars().all()
    if not recommendations:
        return 'No recommendations available.'
    parts = []
    for item in recommendations:
        medicine = f" ({item.medicine})" if item.medicine else ''
        parts.append(f"{item.recommendation}{medicine}")
    return '; '.join(parts)


def get_session():
    with get_db_session() as s:
        yield s


def add_notification(session, user_id: int, title: str, message: str) -> None:
    session.add(Notification(user_id=user_id, title=title, message=message))


def notify_providers(session, title: str, message: str) -> None:
    provider_ids = session.execute(
        select(User.id).where(User.role.in_(['doctor', 'provider']))
    ).scalars().all()
    for provider_id in provider_ids:
        add_notification(session, provider_id, title, message)


def get_user_from_token(authorization: str, session):
    token = authorization.split(' ', 1)[1] if authorization and authorization.startswith('Bearer ') else None
    return get_authenticated_user(token, session)


@app.get('/notifications')
def list_notifications(authorization: str = Header(None), session=Depends(get_session)):
    user = get_user_from_token(authorization, session)
    notifications = session.execute(
        select(Notification)
        .where(Notification.user_id == user.id)
        .order_by(Notification.created_at.desc())
        .limit(50)
    ).scalars().all()
    return {
        'notifications': [
            {
                'id': item.id,
                'title': item.title,
                'message': item.message,
                'created_at': item.created_at.isoformat() if item.created_at else None,
                'read_at': item.read_at.isoformat() if item.read_at else None,
                'is_read': item.read_at is not None,
            }
            for item in notifications
        ],
        'unread_count': sum(1 for item in notifications if item.read_at is None),
    }


@app.post('/notifications/{notification_id}/read')
def mark_notification_read(notification_id: int, authorization: str = Header(None), session=Depends(get_session)):
    user = get_user_from_token(authorization, session)
    notification = session.execute(
        select(Notification).where(Notification.id == notification_id, Notification.user_id == user.id)
    ).scalar_one_or_none()
    if not notification:
        raise HTTPException(status_code=404, detail='Notification not found')
    notification.read_at = notification.read_at or datetime.utcnow()
    session.commit()
    return {'status': 'read', 'id': notification.id}


@app.post('/notifications/read-all')
def mark_all_notifications_read(authorization: str = Header(None), session=Depends(get_session)):
    user = get_user_from_token(authorization, session)
    notifications = session.execute(
        select(Notification).where(Notification.user_id == user.id, Notification.read_at.is_(None))
    ).scalars().all()
    now = datetime.utcnow()
    for notification in notifications:
        notification.read_at = now
    session.commit()
    return {'status': 'read', 'count': len(notifications)}


@app.post('/register')
def register(user_in: UserCreate, session=Depends(get_session)):
    try:
        # Ensure password is at most 72 bytes (bcrypt limit). Truncate safely.
        def _truncate_to_n_bytes(s: str, n: int) -> str:
            b = s.encode('utf-8')
            if len(b) <= n:
                return s
            # iterate characters to avoid splitting multibyte chars
            encoded_len = 0
            out_chars = []
            for ch in s:
                ch_b = ch.encode('utf-8')
                if encoded_len + len(ch_b) > n:
                    break
                out_chars.append(ch)
                encoded_len += len(ch_b)
            return ''.join(out_chars)
        # perform truncation now
        user_in.password = _truncate_to_n_bytes(user_in.password, 72)
        print(f"[register] password_bytes={len(user_in.password.encode('utf-8'))}")
        user, token = create_user(user_in, session=session)
        return {"user": {"id": user['id'], "full_name": user['full_name'], "email": user['email'], "role": user['role'], "phone": user['phone']}, "confirmation_token": token}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except IntegrityError:
        raise HTTPException(status_code=400, detail='Email already registered')
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/confirm')
def confirm_email(token: str, session=Depends(get_session)):
    # Verify token and remove it
    stmt = select(EmailConfirmation).where(EmailConfirmation.token == token)
    conf = session.execute(stmt).scalar_one_or_none()
    if not conf:
        raise HTTPException(status_code=400, detail='Invalid or expired token')
    # Optionally mark user as confirmed (schema change required). We'll delete token.
    session.delete(conf)
    session.commit()
    return {"status": "confirmed"}


from sqlalchemy.exc import NoResultFound

@app.post('/login')
def login(payload: dict, session=Depends(get_session)):
    try:
        # payload: {"email":..., "password":...}
        email = payload.get('email')
        password = payload.get('password')
        if not email or not password:
            raise HTTPException(status_code=400, detail='email and password required')
        stmt = select(User).where(User.email == email)
        user = session.execute(stmt).scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=400, detail='invalid credentials')
        try:
            ok = pwd_ctx.verify(password, user.password_hash)
        except Exception:
            ok = False
        if not ok:
            raise HTTPException(status_code=400, detail='invalid credentials')

        import secrets
        token = secrets.token_urlsafe(32)
        api_token = ApiToken(user_id=user.id, token=token)
        session.add(api_token)
        session.commit()
        return {
            "user": {"id": user.id, "full_name": user.full_name, "email": user.email, "role": user.role},
            "token": token,
        }
    except HTTPException:
        # re-raise expected HTTP errors
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _chat_contact(user: User) -> dict:
    return {'id': user.id, 'full_name': user.full_name, 'email': user.email, 'role': user.role}


@app.get('/chat/contacts')
def chat_contacts(authorization: str = Header(None), session=Depends(get_session)):
    user = get_authenticated_user(
        authorization.split(' ', 1)[1] if authorization and authorization.startswith('Bearer ') else None,
        session,
    )
    target_role = 'doctor' if user.role == 'patient' else 'patient'
    contacts = session.execute(
        select(User).where(User.role == target_role).order_by(User.full_name)
    ).scalars().all()
    return {'contacts': [_chat_contact(contact) for contact in contacts]}


@app.get('/chat/messages/{other_user_id}')
def chat_messages(other_user_id: int, authorization: str = Header(None), session=Depends(get_session)):
    user = get_authenticated_user(
        authorization.split(' ', 1)[1] if authorization and authorization.startswith('Bearer ') else None,
        session,
    )
    other_user = session.get(User, other_user_id)
    if not other_user or other_user.id == user.id or {user.role, other_user.role} != {'patient', 'doctor'}:
        raise HTTPException(status_code=404, detail='Chat contact not found')

    messages = session.execute(
        select(ChatMessage)
        .where(
            ((ChatMessage.sender_id == user.id) & (ChatMessage.recipient_id == other_user.id))
            | ((ChatMessage.sender_id == other_user.id) & (ChatMessage.recipient_id == user.id))
        )
        .order_by(ChatMessage.created_at, ChatMessage.id)
    ).scalars().all()
    for message in messages:
        if message.recipient_id == user.id and message.read_at is None:
            message.read_at = datetime.utcnow()
    session.commit()
    return {
        'contact': _chat_contact(other_user),
        'messages': [
            {
                'id': message.id,
                'sender_id': message.sender_id,
                'recipient_id': message.recipient_id,
                'body': message.body,
                'created_at': message.created_at.isoformat() if message.created_at else None,
                'read_at': message.read_at.isoformat() if message.read_at else None,
            }
            for message in messages
        ],
    }


@app.post('/chat/messages')
def send_chat_message(payload: dict, authorization: str = Header(None), session=Depends(get_session)):
    user = get_authenticated_user(
        authorization.split(' ', 1)[1] if authorization and authorization.startswith('Bearer ') else None,
        session,
    )
    try:
        recipient_id = int(payload.get('recipient_id'))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail='recipient_id is required')
    body = str(payload.get('body') or '').strip()
    recipient = session.get(User, recipient_id)
    if not body or len(body) > 4000:
        raise HTTPException(status_code=400, detail='Message must contain 1 to 4000 characters')
    if not recipient or recipient.id == user.id or {user.role, recipient.role} != {'patient', 'doctor'}:
        raise HTTPException(status_code=404, detail='Chat contact not found')

    message = ChatMessage(sender_id=user.id, recipient_id=recipient.id, body=body)
    session.add(message)
    session.commit()
    session.refresh(message)
    return {
        'id': message.id,
        'sender_id': message.sender_id,
        'recipient_id': message.recipient_id,
        'body': message.body,
        'created_at': message.created_at.isoformat() if message.created_at else None,
    }


def get_authenticated_user(token: str, session):
    if not token:
        raise HTTPException(status_code=401, detail='Authorization token missing')
    stmt = select(ApiToken, User).join(User, ApiToken.user_id == User.id).where(ApiToken.token == token)
    result = session.execute(stmt).first()
    if not result:
        raise HTTPException(status_code=401, detail='Invalid authorization token')
    api_token, user = result
    return user


def get_or_create_patient_profile(session, user, create: bool = False) -> Optional[PatientProfile]:
    """Return the single profile for a patient, creating it only when requested."""
    profile = session.execute(
        select(PatientProfile).where(PatientProfile.user_id == user.id)
    ).scalar_one_or_none()
    if profile or not create:
        return profile

    profile = PatientProfile(user_id=user.id)
    session.add(profile)
    session.flush()
    return profile


def build_provider_analytics(session) -> dict:
    predictions = session.execute(
        select(DiseasePrediction).order_by(DiseasePrediction.prediction_date.desc())
    ).scalars().all()
    prediction_counts = Counter((item.predicted_disease or 'Unknown').strip() for item in predictions)
    confidence_values = [item.confidence for item in predictions if item.confidence is not None]
    total_prediction_count = len(predictions)
    prediction_statuses = [get_prediction_status(item) for item in predictions]
    approved_count = prediction_statuses.count('approved')
    pending_count = prediction_statuses.count('pending')
    rejected_count = prediction_statuses.count('rejected')

    prediction_trend_counts = Counter(
        item.prediction_date.strftime('%Y-%m')
        for item in predictions
        if item.prediction_date
    )

    symptom_rows = session.execute(
        select(Symptom.symptom_name, PatientSymptom.id)
        .join(PatientSymptom, PatientSymptom.symptom_id == Symptom.id)
    ).all()
    symptom_counts = Counter((name or 'Unknown').strip() for name, _ in symptom_rows)

    risk_records = session.execute(select(RiskAssessment)).scalars().all()
    high_risk_records = [item for item in risk_records if (item.risk_level or '').lower() == 'high']

    return {
        'disease_prediction_counts': [
            {
                'disease': disease,
                'count': count,
                'percentage': round((count / total_prediction_count) * 100, 1) if total_prediction_count else 0,
            }
            for disease, count in prediction_counts.most_common()
        ],
        'most_predicted_diseases': [
            {'disease': disease, 'count': count}
            for disease, count in prediction_counts.most_common(5)
        ],
        'average_prediction_confidence': round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0,
        'model_test_accuracy': None,
        'prediction_status': {
            'approved': approved_count,
            'pending': pending_count,
            'rejected': rejected_count,
        },
        'symptom_counts': [
            {'symptom': symptom, 'count': count}
            for symptom, count in symptom_counts.most_common()
        ],
        'prediction_trends': [
            {'period': period, 'count': count}
            for period, count in sorted(prediction_trend_counts.items())
        ],
        'total_patients_assessed': len({item.patient_id for item in predictions}),
        'most_common_disease': prediction_counts.most_common(1)[0][0] if prediction_counts else None,
        'most_common_symptom': symptom_counts.most_common(1)[0][0] if symptom_counts else None,
        'high_risk_cases': {
            'count': len({item.patient_id for item in high_risk_records}),
            'patient_ids': sorted({item.patient_id for item in high_risk_records}),
        },
    }


@app.get('/dashboard/patient')
def get_patient_dashboard(authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = get_or_create_patient_profile(session, user)
    patient_id = profile.id if profile else None

    medical_history = []
    symptoms = []
    predictions = []
    risk = None
    risk_history = []
    recommendations = []
    reports = []
    healthcare_advisory = None

    if patient_id:
        medical_history = [
            {
                'disease': mh.disease,
                'diagnosed_date': mh.diagnosed_date.isoformat() if mh.diagnosed_date else None,
                'treatment': mh.treatment,
                'status': mh.status,
            }
            for mh in session.execute(select(MedicalHistory).where(MedicalHistory.patient_id == patient_id)).scalars().all()
        ]

        symptoms = [
            {
                'id': patient_symptom.id,
                'symptom_id': symptom.id,
                'symptom_name': symptom.symptom_name,
                'severity': patient_symptom.severity,
                'entered_date': patient_symptom.entered_date.isoformat(),
            }
            for patient_symptom, symptom in session.execute(
                select(PatientSymptom, Symptom)
                .join(Symptom, PatientSymptom.symptom_id == Symptom.id)
                .where(PatientSymptom.patient_id == patient_id)
            ).all()
        ]

        predictions = [
            {
                'id': dp.id,
                'predicted_disease': dp.predicted_disease,
                'confidence': dp.confidence,
                'prediction_date': dp.prediction_date.isoformat(),
                'status': normalize_approval_status(dp.status or dp.provider_feedback),
                'provider_feedback': normalize_approval_status(dp.provider_feedback or dp.status),
            }
            for dp in session.execute(
                select(DiseasePrediction)
                .where(DiseasePrediction.patient_id == patient_id)
                .order_by(DiseasePrediction.prediction_date.desc())
            ).scalars().all()
        ]

        risk_history = [
            {
                'id': assessment.id,
                'risk_level': assessment.risk_level,
                'score': assessment.score,
                'remarks': assessment.remarks,
                'created_at': assessment.created_at.isoformat() if assessment.created_at else None,
            }
            for assessment in session.execute(
                select(RiskAssessment)
                .where(RiskAssessment.patient_id == patient_id)
                .order_by(RiskAssessment.created_at.desc())
            ).scalars().all()
        ]
        risk = risk_history[0] if risk_history else None
        risk_fields = {}
        if risk:
            risk_fields = {
                'risk_level': risk['risk_level'],
                'score': risk['score'],
                'remarks': risk['remarks'],
                'factors': ['age', 'symptom burden', 'chronic conditions'] if 'risk' in (risk['remarks'] or '').lower() else ['monitoring'],
                'warning': 'Consult a healthcare provider promptly and seek urgent care if symptoms worsen.' if (risk['risk_level'] or '').lower() == 'high' else 'Continue routine monitoring and maintain follow-up health habits.',
            }
        latest_prediction = predictions[0] if predictions else None
        healthcare_advisory = build_healthcare_advisory(
            predicted_disease=latest_prediction['predicted_disease'] if latest_prediction else '',
            risk_assessment=(risk['remarks'] if risk else ''),
            symptoms=[item['symptom_name'] for item in symptoms],
        ) if latest_prediction or risk or symptoms else None
        recommendations = [
            {
                'id': rec.id,
                'prediction_id': rec.prediction_id,
                'recommendation': rec.recommendation,
                'medicine': rec.medicine,
                'priority': rec.priority,
                'recommendation_type': rec.recommendation_type,
                'status': rec.status,
                'provider_comments': rec.provider_comments,
                'created_at': rec.created_at.isoformat(),
            }
            for rec in session.execute(
                select(Recommendation)
                .where(Recommendation.patient_id == patient_id, Recommendation.status == 'approved')
                .order_by(Recommendation.created_at.desc())
            ).scalars().all()
        ]
        reports = [
            serialize_report(report)
            for report in session.execute(
                select(Report)
                .where(Report.patient_id == patient_id)
                .order_by(Report.generated_at.desc())
            ).scalars().all()
        ]
    
    return {
        'user': {'id': user.id, 'full_name': user.full_name, 'email': user.email, 'role': user.role, 'phone': user.phone},
        'profile': {
            'age': profile.age if profile else None,
            'gender': profile.gender if profile else None,
            'blood_group': profile.blood_group if profile else None,
            'height': profile.height if profile else None,
            'weight': profile.weight if profile else None,
            'bmi': profile.bmi if profile else None,
            'emergency_contact': profile.emergency_contact if profile else None,
            'existing_conditions': profile.existing_conditions if profile else None,
            'allergies': profile.allergies if profile else None,
            'dob': profile.dob.isoformat() if profile and profile.dob else None,
            'profile_picture_url': profile.profile_picture_url if profile else None,
        },
        'medical_history': medical_history,
        'symptoms': symptoms,
        'predictions': predictions,
        'risk': risk_fields if risk_fields else None,
        'risk_history': risk_history,
        'recommendations': recommendations,
        'reports': reports,
        'healthcare_advisory': healthcare_advisory,
    }


@app.get('/patient/profile')
def get_patient_profile(authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = get_or_create_patient_profile(session, user)
    if not profile:
        raise HTTPException(status_code=404, detail='Patient profile not found')

    return {
        'user': {
            'id': user.id,
            'full_name': user.full_name,
            'email': user.email,
            'role': user.role,
            'phone': user.phone,
            'notification_preferences': user.notification_preferences,
            'profile_preferences': user.profile_preferences,
        },
        'profile': {
            'full_name': user.full_name,
            'phone': user.phone,
            'dob': profile.dob.isoformat() if profile.dob else None,
            'age': profile.age,
            'gender': profile.gender,
            'blood_group': profile.blood_group,
            'height': profile.height,
            'weight': profile.weight,
            'bmi': profile.bmi,
            'emergency_contact': profile.emergency_contact,
            'existing_conditions': profile.existing_conditions,
            'allergies': profile.allergies,
            'profile_picture_url': profile.profile_picture_url,
        },
    }


@app.put('/patient/profile')
def update_patient_profile(profile_in: PatientProfileUpdate, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = get_or_create_patient_profile(session, user, create=True)

    if profile_in.full_name is not None:
        user.full_name = profile_in.full_name
    if profile_in.phone is not None:
        user.phone = profile_in.phone
    if profile_in.dob is not None:
        try:
            profile.dob = datetime.fromisoformat(profile_in.dob).date()
        except ValueError:
            raise HTTPException(status_code=400, detail='Invalid dob format, expected YYYY-MM-DD')
    if profile_in.age is not None:
        profile.age = profile_in.age
    if profile_in.gender is not None:
        profile.gender = profile_in.gender
    if profile_in.blood_group is not None:
        profile.blood_group = profile_in.blood_group
    if profile_in.height is not None:
        profile.height = profile_in.height
    if profile_in.weight is not None:
        profile.weight = profile_in.weight
    if profile_in.emergency_contact is not None:
        profile.emergency_contact = profile_in.emergency_contact
    if profile_in.existing_conditions is not None:
        profile.existing_conditions = profile_in.existing_conditions
    if profile_in.allergies is not None:
        profile.allergies = profile_in.allergies
    if profile_in.profile_picture_url is not None:
        profile.profile_picture_url = profile_in.profile_picture_url

    session.add(user)
    session.add(profile)
    notify_providers(
        session,
        'Patient information updated',
        f'{user.full_name} updated their profile information.',
    )
    session.commit()

    return {'status': 'updated'}


@app.get('/patient/history')
def get_medical_history(authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = get_or_create_patient_profile(session, user)
    if not profile:
        return {'medical_history': []}

    histories = session.execute(select(MedicalHistory).where(MedicalHistory.patient_id == profile.id).order_by(MedicalHistory.diagnosed_date.desc())).scalars().all()
    return {'medical_history': [
        {
            'id': mh.id,
            'disease': mh.disease,
            'diagnosed_date': mh.diagnosed_date.isoformat() if mh.diagnosed_date else None,
            'treatment': mh.treatment,
            'status': mh.status,
            'surgery': mh.surgery,
            'medications': mh.medications,
            'allergies': mh.allergies,
            'family_history': mh.family_history,
            'ongoing_treatment': mh.ongoing_treatment,
        }
        for mh in histories
    ]}


@app.post('/patient/history')
def add_medical_history(entry: MedicalHistoryCreate, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = get_or_create_patient_profile(session, user, create=True)

    mh = MedicalHistory(
        patient_id=profile.id,
        disease=entry.disease,
        diagnosed_date=datetime.fromisoformat(entry.diagnosed_date).date() if entry.diagnosed_date else None,
        treatment=entry.treatment,
        status=entry.status,
        surgery=entry.surgery,
        medications=entry.medications,
        allergies=entry.allergies,
        family_history=entry.family_history,
        ongoing_treatment=entry.ongoing_treatment,
    )
    session.add(mh)
    session.commit()
    return {'status': 'created', 'id': mh.id}


@app.put('/patient/history/{history_id}')
def update_medical_history(history_id: int, entry: MedicalHistoryUpdate, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail='Patient profile not found')

    mh = session.execute(select(MedicalHistory).where(MedicalHistory.id == history_id, MedicalHistory.patient_id == profile.id)).scalar_one_or_none()
    if not mh:
        raise HTTPException(status_code=404, detail='Record not found')

    if entry.disease is not None:
        mh.disease = entry.disease
    if entry.diagnosed_date is not None:
        mh.diagnosed_date = datetime.fromisoformat(entry.diagnosed_date).date()
    if entry.treatment is not None:
        mh.treatment = entry.treatment
    if entry.status is not None:
        mh.status = entry.status
    if entry.surgery is not None:
        mh.surgery = entry.surgery
    if entry.medications is not None:
        mh.medications = entry.medications
    if entry.allergies is not None:
        mh.allergies = entry.allergies
    if entry.family_history is not None:
        mh.family_history = entry.family_history
    if entry.ongoing_treatment is not None:
        mh.ongoing_treatment = entry.ongoing_treatment

    session.add(mh)
    session.commit()
    return {'status': 'updated'}


@app.delete('/patient/history/{history_id}')
def delete_medical_history(history_id: int, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail='Patient profile not found')

    mh = session.execute(select(MedicalHistory).where(MedicalHistory.id == history_id, MedicalHistory.patient_id == profile.id)).scalar_one_or_none()
    if not mh:
        raise HTTPException(status_code=404, detail='Record not found')

    session.delete(mh)
    session.commit()
    return {'status': 'deleted'}


@app.get('/patient/symptoms')
def get_patient_symptoms(authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        return {'symptoms': []}

    symptoms = session.execute(
        select(PatientSymptom, Symptom)
        .join(Symptom, PatientSymptom.symptom_id == Symptom.id)
        .where(PatientSymptom.patient_id == profile.id)
        .order_by(PatientSymptom.entered_date.desc())
    ).all()

    return {'symptoms': [
        {
            'id': ps.id,
            'symptom_id': symptom.id,
            'symptom_name': symptom.symptom_name,
            'severity': ps.severity,
            'duration': ps.duration,
            'frequency': ps.frequency,
            'notes': ps.notes,
            'entered_date': ps.entered_date.isoformat(),
        }
        for ps, symptom in symptoms
    ]}


@app.get('/symptoms/search')
def search_symptoms(q: str, session=Depends(get_session)):
    stmt = select(Symptom).where(Symptom.symptom_name.ilike(f'%{q}%')).limit(20)
    results = session.execute(stmt).scalars().all()
    return {'results': [{'id': s.id, 'symptom_name': s.symptom_name} for s in results]}


@app.post('/patient/symptoms')
def add_patient_symptoms(payload: PatientSymptomCreate, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail='Patient profile not found')

    created_ids = []
    for entry in payload.symptoms:
        symptom = None
        if entry.symptom_id:
            symptom = session.execute(select(Symptom).where(Symptom.id == entry.symptom_id)).scalar_one_or_none()
        if not symptom and entry.symptom_name:
            symptom = session.execute(select(Symptom).where(Symptom.symptom_name == entry.symptom_name)).scalar_one_or_none()
            if not symptom:
                symptom = Symptom(symptom_name=entry.symptom_name)
                session.add(symptom)
                session.flush()
        if not symptom:
            continue
        ps = PatientSymptom(
            patient_id=profile.id,
            symptom_id=symptom.id,
            severity=entry.severity,
            duration=entry.duration,
            frequency=entry.frequency,
            notes=entry.notes,
        )
        session.add(ps)
        session.flush()
        created_ids.append(ps.id)

    if created_ids:
        symptom_word = 'entry' if len(created_ids) == 1 else 'entries'
        notify_providers(
            session,
            'New symptom assessment',
            f'{user.full_name} submitted {len(created_ids)} new symptom {symptom_word} for review.',
        )
    session.commit()
    return {'status': 'created', 'ids': created_ids}


@app.put('/patient/symptoms/{symptom_id}')
def update_patient_symptom(symptom_id: int, entry: PatientSymptomCreate, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail='Patient profile not found')

    ps = session.execute(select(PatientSymptom).where(PatientSymptom.id == symptom_id, PatientSymptom.patient_id == profile.id)).scalar_one_or_none()
    if not ps:
        raise HTTPException(status_code=404, detail='Record not found')

    entry_data = entry.symptoms[0] if entry.symptoms else None
    if not entry_data:
        raise HTTPException(status_code=400, detail='No symptom data provided')

    if entry_data.severity is not None:
        ps.severity = entry_data.severity
    if entry_data.duration is not None:
        ps.duration = entry_data.duration
    if entry_data.frequency is not None:
        ps.frequency = entry_data.frequency
    if entry_data.notes is not None:
        ps.notes = entry_data.notes

    session.add(ps)
    session.commit()
    return {'status': 'updated'}


@app.delete('/patient/symptoms/{symptom_id}')
def delete_patient_symptom(symptom_id: int, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail='Patient profile not found')

    ps = session.execute(select(PatientSymptom).where(PatientSymptom.id == symptom_id, PatientSymptom.patient_id == profile.id)).scalar_one_or_none()
    if not ps:
        raise HTTPException(status_code=404, detail='Record not found')

    session.delete(ps)
    session.commit()
    return {'status': 'deleted'}


@app.post('/patient/prediction')
def run_disease_prediction(payload: PredictionRequest, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = get_or_create_patient_profile(session, user, create=True)

    symptoms = []
    if payload.symptom_ids:
        symptom_objs = session.execute(select(Symptom).where(Symptom.id.in_(payload.symptom_ids))).scalars().all()
        symptoms = [s.symptom_name for s in symptom_objs]
    if payload.symptom_names:
        symptoms.extend(payload.symptom_names)

    if not symptoms:
        raise HTTPException(status_code=400, detail='No symptoms provided for prediction')

    try:
        predicted, confidence = predict_disease_from_model(symptoms, profile)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    dp = DiseasePrediction(
        patient_id=profile.id,
        predicted_disease=predicted,
        confidence=confidence,
        status='pending',
        provider_feedback='pending',
        provider_comments='Awaiting provider review.'
    )
    dp.prediction_date = datetime.utcnow()
    session.add(dp)
    session.flush()

    medical_histories = session.execute(
        select(MedicalHistory).where(MedicalHistory.patient_id == profile.id)
    ).scalars().all()
    medical_history_text = ' | '.join(item.disease for item in medical_histories) if medical_histories else None
    generate_recommendations_for_prediction(
        session=session,
        patient_id=profile.id,
        prediction_id=dp.id,
        predicted_disease=predicted,
        confidence=confidence,
        patient_profile=profile,
        risk_assessment=get_latest_patient_risk(session, profile.id),
        symptoms=symptoms,
        medical_history_text=medical_history_text,
        status='pending',
    )

    prediction_risk = get_latest_patient_risk(session, profile.id)
    prediction_risk_text = (
        prediction_risk.remarks
        if prediction_risk and prediction_risk.remarks
        else (f'{prediction_risk.risk_level} ({prediction_risk.score})' if prediction_risk else 'No risk assessment available')
    )

    report = Report(
        patient_id=profile.id,
        prediction_id=dp.id,
        report_name=f'MedAssist Report - Prediction {dp.id}',
        report_type='AI Prediction Summary',
        status='pending',
        provider_status='pending',
        generated_at=datetime.utcnow(),
        prediction_date=dp.prediction_date,
        symptoms=json.dumps(symptoms),
        predicted_disease=predicted,
        confidence_score=confidence,
        risk_assessment=prediction_risk_text,
        provider_comments='',
        recommendations='Pending provider approval',
    )
    report.report_url = build_report_download_url(build_report_payload(
        patient_id=profile.id,
        symptoms=symptoms,
        predicted_disease=predicted,
        confidence_score=confidence,
        risk_assessment=report.risk_assessment,
        provider_status=report.provider_status,
        provider_comments=report.provider_comments,
        recommendations=report.recommendations,
    ))
    session.add(report)

    notify_providers(
        session,
        'New disease prediction',
        f'{user.full_name} submitted a new AI disease prediction for review.',
    )
    generated_recommendations = session.execute(
        select(Recommendation)
        .where(Recommendation.patient_id == profile.id, Recommendation.prediction_id == dp.id)
        .order_by(Recommendation.created_at.asc())
    ).scalars().all()
    session.commit()

    return {
        'prediction_id': dp.id,
        'report_id': report.id,
        'predicted_disease': predicted,
        'confidence': confidence,
                'status': normalize_approval_status(dp.status or dp.provider_feedback),
                'provider_feedback': normalize_approval_status(dp.provider_feedback or dp.status),
        'prediction_date': dp.prediction_date.isoformat(),
        'recommendations': [
            {
                'id': item.id,
                'recommendation': item.recommendation,
                'medicine': item.medicine,
                'priority': item.priority,
                'recommendation_type': item.recommendation_type,
                'status': item.status,
            }
            for item in generated_recommendations
        ],
    }


@app.post('/patient/risk')
def run_risk_assessment(payload: RiskAssessmentRequest, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail='Patient profile not found')

    symptoms = get_patient_symptom_names(session, profile.id)
    medical_history = (
        ', '.join(
            [
                str(item.disease or '')
                for item in session.execute(select(MedicalHistory).where(MedicalHistory.patient_id == profile.id)).scalars().all()
            ]
        )
    )
    lifestyle = {
        'smoking': 'no',
        'alcohol': 'no',
        'exercise': 'low',
    }
    if payload and getattr(payload, 'notes', None):
        note_lower = (payload.notes or '').lower()
        if 'smoker' in note_lower or 'smoking' in note_lower:
            lifestyle['smoking'] = 'yes'
        if 'drinks alcohol' in note_lower or 'alcohol' in note_lower:
            lifestyle['alcohol'] = 'heavy'
        if 'exercise' in note_lower and 'low' in note_lower:
            lifestyle['exercise'] = 'low'

    risk = evaluate_patient_risk(
        profile=profile,
        symptoms=symptoms,
        predicted_disease=(
            session.execute(
                select(DiseasePrediction.predicted_disease)
                .where(DiseasePrediction.patient_id == profile.id)
                .order_by(DiseasePrediction.prediction_date.desc())
            ).scalars().first()
            or ''
        ),
        medical_history=medical_history,
        lifestyle=lifestyle,
    )

    ra = RiskAssessment(patient_id=profile.id)
    session.add(ra)
    ra.risk_level = risk['risk_level']
    ra.score = risk['score']
    ra.remarks = risk['remarks']
    session.commit()

    return {
        'risk_level': risk['risk_level'],
        'score': risk['score'],
        'remarks': risk['remarks'],
        'factors': risk['factors'],
        'warning': risk['warning'],
    }


@app.put('/patient/settings')
def update_patient_settings(settings: SettingsUpdate, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    if settings.new_password:
        if not settings.old_password or not pwd_ctx.verify(settings.old_password, user.password_hash):
            raise HTTPException(status_code=400, detail='Current password is incorrect')
        user.password_hash = pwd_ctx.hash(settings.new_password)

    if settings.notification_preferences is not None:
        user.notification_preferences = settings.notification_preferences
    if settings.profile_preferences is not None:
        user.profile_preferences = settings.profile_preferences

    session.add(user)
    session.commit()
    return {'status': 'updated'}


@app.get('/patient/recommendations')
def list_patient_recommendations(authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        return {'recommendations': []}

    recs = session.execute(
        select(Recommendation).where(Recommendation.patient_id == profile.id, Recommendation.status == 'approved').order_by(Recommendation.created_at.desc())
    ).scalars().all()
    return {'recommendations': [
        {
            'id': r.id,
            'prediction_id': r.prediction_id,
            'recommendation': r.recommendation,
            'medicine': r.medicine,
            'priority': r.priority,
            'recommendation_type': r.recommendation_type,
            'status': r.status,
            'provider_comments': r.provider_comments,
            'created_at': r.created_at.isoformat(),
        }
        for r in recs
    ]}


@app.get('/patient/reports')
def list_patient_reports(authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        return {'reports': []}

    reps = session.execute(select(Report).where(Report.patient_id == profile.id).order_by(Report.generated_at.desc())).scalars().all()
    return {'reports': [serialize_report(report) for report in reps]}


@app.get('/dashboard/provider')
def get_provider_dashboard(authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role not in ('doctor', 'provider'):
        raise HTTPException(status_code=403, detail='Forbidden')

    provider = session.execute(select(ProviderProfile).where(ProviderProfile.user_id == user.id)).scalar_one_or_none()
    if not provider:
        raise HTTPException(status_code=404, detail='Provider profile not found')

    # For now, provider can access all patients. In future this can be scoped by assignment.
    patients = session.execute(select(PatientProfile).order_by(PatientProfile.id.desc())).scalars().all()
    medical_histories = session.execute(select(MedicalHistory).order_by(MedicalHistory.diagnosed_date.desc())).scalars().all()
    patient_symptom_rows = session.execute(
        select(PatientSymptom, Symptom)
        .join(Symptom, PatientSymptom.symptom_id == Symptom.id)
        .order_by(PatientSymptom.entered_date.desc())
    ).all()

    latest_prediction_by_patient = {}
    for dp in session.execute(select(DiseasePrediction).order_by(DiseasePrediction.patient_id, DiseasePrediction.prediction_date.desc())).scalars().all():
        latest_prediction_by_patient.setdefault(dp.patient_id, dp)

    latest_risk_by_patient = {}
    for ra in session.execute(select(RiskAssessment).order_by(RiskAssessment.patient_id, RiskAssessment.created_at.desc())).scalars().all():
        latest_risk_by_patient.setdefault(ra.patient_id, ra)

    latest_history_by_patient = {}
    patient_conditions = {}
    for mh in medical_histories:
        latest_history_by_patient.setdefault(mh.patient_id, mh)
        patient_conditions.setdefault(mh.patient_id, []).append(mh.disease)

    patient_symptoms = []
    symptom_summary_by_patient = {}
    for ps, symptom in patient_symptom_rows:
        patient_symptoms.append({
            'id': ps.id,
            'patient_id': ps.patient_id,
            'symptom_id': symptom.id,
            'symptom_name': symptom.symptom_name,
            'severity': ps.severity,
            'duration': ps.duration,
            'frequency': ps.frequency,
            'notes': ps.notes,
            'entered_date': ps.entered_date.isoformat() if ps.entered_date else None,
        })
        symptom_summary_by_patient.setdefault(ps.patient_id, []).append(ps)

    patient_list = []
    for patient in patients:
        user_record = session.execute(select(User).where(User.id == patient.user_id)).scalar_one_or_none()
        latest_prediction = latest_prediction_by_patient.get(patient.id)
        latest_risk = latest_risk_by_patient.get(patient.id)
        latest_history = latest_history_by_patient.get(patient.id)
        symptom_entries = symptom_summary_by_patient.get(patient.id, [])
        last_visit = None
        if latest_history and latest_history.diagnosed_date:
            last_visit = latest_history.diagnosed_date.isoformat()
        elif symptom_entries:
            last_visit = symptom_entries[0].entered_date.isoformat() if symptom_entries[0].entered_date else None
        elif latest_prediction:
            last_visit = latest_prediction.prediction_date.isoformat() if latest_prediction.prediction_date else None

        patient_list.append({
            'id': patient.id,
            'patient_id': patient.id,
            'name': user_record.full_name if user_record else 'Unknown',
            'email': user_record.email if user_record else None,
            'phone': user_record.phone,
            'gender': patient.gender,
            'age': patient.age,
            'dob': patient.dob.isoformat() if patient.dob else None,
            'blood_group': patient.blood_group,
            'risk_level': latest_risk.risk_level if latest_risk else None,
            'latest_prediction': latest_prediction.predicted_disease if latest_prediction else None,
            'latest_prediction_confidence': latest_prediction.confidence if latest_prediction else None,
            'conditions': list({c for c in patient_conditions.get(patient.id, []) if c}),
            'last_visit': last_visit,
            'status': 'Active' if latest_history or symptom_entries or latest_prediction or latest_risk else 'New',
            'registered_at': user_record.created_at.isoformat() if user_record and user_record.created_at else None,
        })

    patient_history = [
        {
            'patient_id': mh.patient_id,
            'disease': mh.disease,
            'diagnosed_date': mh.diagnosed_date.isoformat() if mh.diagnosed_date else None,
            'treatment': mh.treatment,
            'status': mh.status,
        }
        for mh in session.execute(select(MedicalHistory).order_by(MedicalHistory.diagnosed_date.desc())).scalars().all()
    ]

    predictions = [
        {
            'id': dp.id,
            'patient_id': dp.patient_id,
            'predicted_disease': dp.predicted_disease,
            'confidence': dp.confidence,
            'prediction_date': dp.prediction_date.isoformat(),
            'status': normalize_approval_status(dp.status or dp.provider_feedback),
            'provider_feedback': normalize_approval_status(dp.provider_feedback or dp.status),
            'provider_comments': dp.provider_comments,
            'feedback_date': dp.feedback_date.isoformat() if dp.feedback_date else None,
        }
        for dp in session.execute(select(DiseasePrediction).order_by(DiseasePrediction.prediction_date.desc())).scalars().all()
    ]

    risks = [
        {
            'patient_id': ra.patient_id,
            'risk_level': ra.risk_level,
            'score': ra.score,
            'remarks': ra.remarks,
        }
        for ra in session.execute(select(RiskAssessment)).scalars().all()
    ]

    recommendations = [
        {
            'id': rec.id,
            'patient_id': rec.patient_id,
                'prediction_id': rec.prediction_id,
            'recommendation': rec.recommendation,
            'medicine': rec.medicine,
            'priority': rec.priority,
            'recommendation_type': rec.recommendation_type,
            'status': rec.status,
            'provider_comments': rec.provider_comments,
            'ai_generated': rec.ai_generated,
            'created_at': rec.created_at.isoformat(),
        }
        for rec in session.execute(select(Recommendation).order_by(Recommendation.created_at.desc())).scalars().all()
    ]

    reports = [
        serialize_report(report, include_patient_id=True)
        for report in session.execute(select(Report).order_by(Report.generated_at.desc())).scalars().all()
    ]
    analytics = build_provider_analytics(session)

    return {
        'user': {'id': user.id, 'full_name': user.full_name, 'email': user.email, 'role': user.role, 'phone': user.phone},
        'provider_profile': {
            'hospital_name': provider.hospital_name,
            'specialization': provider.specialization,
            'license_number': provider.license_number,
            'years_experience': provider.years_experience,
            'qualification': provider.qualification,
            'department': provider.department,
        },
        'patients': patient_list,
        'patient_history': patient_history,
        'symptoms': patient_symptoms,
        'predictions': predictions,
        'risks': risks,
        'recommendations': recommendations,
        'reports': reports,
        'analytics': analytics,
    }


@app.get('/provider/analytics')
def get_provider_analytics(authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role not in ('doctor', 'provider'):
        raise HTTPException(status_code=403, detail='Forbidden')
    return build_provider_analytics(session)


@app.post('/provider/recommendations')
def create_provider_recommendation(payload: ProviderRecommendationCreate, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role not in ('doctor', 'provider'):
        raise HTTPException(status_code=403, detail='Forbidden')

    patient = session.execute(select(PatientProfile).where(PatientProfile.id == payload.patient_id)).scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    prediction = None
    if payload.prediction_id is not None:
        prediction = session.execute(
            select(DiseasePrediction).where(DiseasePrediction.id == payload.prediction_id)
        ).scalar_one_or_none()
        if not prediction:
            raise HTTPException(status_code=404, detail='Prediction not found')
        if prediction.patient_id != payload.patient_id:
            raise HTTPException(status_code=400, detail='Prediction does not belong to patient')

    normalized_status = normalize_approval_status(payload.status)
    if normalized_status not in ('pending', 'approved', 'rejected'):
        raise HTTPException(status_code=400, detail='Status must be pending, approved, or rejected')

    recommendation = Recommendation(
        patient_id=payload.patient_id,
        recommendation=payload.recommendation,
        medicine=payload.medicine,
        priority=payload.priority,
        recommendation_type=payload.recommendation_type or 'Provider-added',
        status=normalized_status,
        ai_generated='no',
        provider_comments=payload.provider_comments,
        reviewed_at=datetime.utcnow() if normalized_status in ('approved', 'rejected') else None,
    )
    session.add(recommendation)
    patient_user_id = session.execute(select(PatientProfile.user_id).where(PatientProfile.id == payload.patient_id)).scalar_one()
    add_notification(
        session,
        patient_user_id,
        'Provider recommendation update',
        f'{user.full_name} added a recommendation for your care.' if normalized_status == 'approved' else f'{user.full_name} added a recommendation that is awaiting review.',
    )
    session.commit()

    return {
        'status': 'created',
        'id': recommendation.id,
        'patient_id': recommendation.patient_id,
        'recommendation': recommendation.recommendation,
        'medicine': recommendation.medicine,
        'priority': recommendation.priority,
        'recommendation_type': recommendation.recommendation_type,
        'status': recommendation.status,
        'provider_comments': recommendation.provider_comments,
        'created_at': recommendation.created_at.isoformat(),
    }


@app.post('/provider/recommendations/review')
def review_recommendation(payload: RecommendationReviewRequest, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role not in ('doctor', 'provider'):
        raise HTTPException(status_code=403, detail='Forbidden')

    recommendation_id = getattr(payload, 'recommendation_id', None)
    if recommendation_id is None:
        recommendation_id = getattr(payload, 'id', None)
    if recommendation_id is None:
        raise HTTPException(status_code=400, detail='recommendation_id is required')

    recommendation = session.execute(select(Recommendation).where(Recommendation.id == recommendation_id)).scalar_one_or_none()
    if not recommendation:
        raise HTTPException(status_code=404, detail='Recommendation not found')

    status_value = getattr(payload, 'status', None)
    normalized_status = normalize_approval_status(status_value)
    if status_value is not None and str(status_value).strip().lower() not in VALID_APPROVAL_STATUSES:
        raise HTTPException(status_code=400, detail='Status must be pending, approved, or rejected')

    recommendation_text = getattr(payload, 'recommendation', None)
    medicine_value = getattr(payload, 'medicine', None)
    priority_value = getattr(payload, 'priority', None)
    recommendation_type_value = getattr(payload, 'recommendation_type', None)
    comments_value = getattr(payload, 'provider_comments', None)
    if recommendation_text is not None:
        recommendation.recommendation = recommendation_text
    if medicine_value is not None:
        recommendation.medicine = medicine_value
    if priority_value is not None:
        recommendation.priority = priority_value
    if recommendation_type_value is not None:
        recommendation.recommendation_type = recommendation_type_value
    recommendation.status = normalized_status
    if comments_value is not None:
        recommendation.provider_comments = comments_value
    if normalized_status in ('approved', 'rejected'):
        recommendation.reviewed_at = datetime.utcnow()
        patient_user_id = session.execute(select(PatientProfile.user_id).where(PatientProfile.id == recommendation.patient_id)).scalar_one()
        add_notification(
            session,
            patient_user_id,
            'Recommendation reviewed',
            f'Your provider {user.full_name} {normalized_status} a recommendation.' + (f' Comment: {comments_value}' if comments_value else ''),
        )
    session.add(recommendation)
    session.commit()

    return {
        'status': 'updated',
        'id': recommendation.id,
        'recommendation': recommendation.recommendation,
        'medicine': recommendation.medicine,
        'priority': recommendation.priority,
        'recommendation_type': recommendation.recommendation_type,
        'status': recommendation.status,
        'provider_comments': recommendation.provider_comments,
        'reviewed_at': recommendation.reviewed_at.isoformat() if recommendation.reviewed_at else None,
    }


@app.post('/provider/reports')
def create_provider_report(payload: ProviderReportCreate, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role not in ('doctor', 'provider'):
        raise HTTPException(status_code=403, detail='Forbidden')

    patient = session.execute(select(PatientProfile).where(PatientProfile.id == payload.patient_id)).scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    report = Report(
        patient_id=payload.patient_id,
        prediction_id=payload.prediction_id,
        report_name=payload.report_name,
        report_type=payload.report_type or 'Summary',
        status=normalize_approval_status(payload.status),
        provider_status=normalize_approval_status(payload.provider_status or payload.status),
        report_url=payload.report_url,
        prediction_date=prediction.prediction_date if prediction else None,
        symptoms=json.dumps(payload.symptoms or []),
        predicted_disease=payload.predicted_disease,
        confidence_score=payload.confidence_score,
        risk_assessment=payload.risk_assessment,
        provider_comments=payload.provider_comments,
        recommendations=payload.recommendations,
    )
    session.add(report)
    patient_user_id = session.execute(select(PatientProfile.user_id).where(PatientProfile.id == payload.patient_id)).scalar_one()
    add_notification(
        session,
        patient_user_id,
        'Health report updated',
        f'{user.full_name} generated or updated your health report.',
    )
    session.commit()

    return {
        'status': 'created',
        'id': report.id,
        'patient_id': report.patient_id,
        'report_name': report.report_name,
        'report_type': report.report_type,
        'status': report.status,
        'report_url': report.report_url,
        'generated_at': report.generated_at.isoformat(),
    }


@app.post('/provider/prediction/feedback')
def submit_prediction_feedback(payload: PredictionFeedbackRequest, authorization: str = Header(None), session=Depends(get_session)):
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role not in ('doctor', 'provider'):
        raise HTTPException(status_code=403, detail='Forbidden')

    prediction = session.execute(select(DiseasePrediction).where(DiseasePrediction.id == payload.prediction_id)).scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=404, detail='Prediction not found')

    raw_feedback = (payload.feedback or payload.status or '').strip().lower()
    feedback = normalize_approval_status(raw_feedback, default='')
    if feedback not in ('approved', 'rejected'):
        raise HTTPException(status_code=400, detail='Feedback must be approved or rejected')

    prediction.provider_feedback = feedback
    prediction.status = feedback
    prediction.provider_comments = payload.comments or prediction.provider_comments or 'No comments added.'
    prediction.feedback_date = datetime.utcnow()
    session.add(prediction)
    session.commit()

    report_payload = None
    if feedback == 'approved':
        report = session.execute(
            select(Report).where(Report.prediction_id == prediction.id)
        ).scalar_one_or_none()
        if report is None:
            raise HTTPException(status_code=500, detail='Report record missing for prediction')
        try:
            symptoms = json.loads(report.symptoms) if report.symptoms else []
        except (TypeError, ValueError):
            symptoms = [item.strip() for item in str(report.symptoms).split(',') if item.strip()]
        if not symptoms:
            symptoms = get_patient_symptom_names(session, prediction.patient_id)
        prediction_recommendations = session.execute(
            select(Recommendation).where(
                Recommendation.patient_id == prediction.patient_id,
                Recommendation.prediction_id == prediction.id,
                Recommendation.ai_generated == 'yes',
            )
        ).scalars().all()
        for recommendation in prediction_recommendations:
            recommendation.status = 'approved'
            recommendation.reviewed_at = datetime.utcnow()
            session.add(recommendation)
        report_payload = build_report_payload(
            patient_id=prediction.patient_id,
            symptoms=symptoms,
            predicted_disease=prediction.predicted_disease,
            confidence_score=prediction.confidence,
            risk_assessment=report.risk_assessment or 'No risk assessment available',
            provider_status=feedback,
            provider_comments=prediction.provider_comments or 'No comments added.',
            recommendations=get_patient_recommendation_text(session, prediction.patient_id, prediction.id),
        )
        report.report_name = f"MedAssist Report - Prediction {prediction.id}"
        report.report_type = 'AI Prediction Summary'
        report.status = 'approved'
        report.report_url = build_report_download_url(report_payload)
        report.generated_at = datetime.utcnow()
        report.prediction_date = prediction.prediction_date
        report.symptoms = json.dumps(report_payload['symptoms'])
        report.predicted_disease = report_payload['predicted_disease']
        report.confidence_score = report_payload['confidence_score']
        report.risk_assessment = report_payload['risk_assessment']
        report.provider_status = 'approved'
        report.provider_comments = report_payload['provider_comments']
        report.recommendations = report_payload['recommendations']
        session.add(report)
        
        patient_user_id = session.execute(select(PatientProfile.user_id).where(PatientProfile.id == prediction.patient_id)).scalar_one()
        add_notification(
            session,
            patient_user_id,
            'Prediction approved and report ready',
            f'Your provider {user.full_name} approved your disease prediction and generated a health report with personalized recommendations.',
        )
        session.commit()
    else:
        report = session.execute(select(Report).where(Report.prediction_id == prediction.id)).scalar_one_or_none()
        prediction_recommendations = session.execute(
            select(Recommendation).where(
                Recommendation.patient_id == prediction.patient_id,
                Recommendation.prediction_id == prediction.id,
                Recommendation.ai_generated == 'yes',
            )
        ).scalars().all()
        for recommendation in prediction_recommendations:
            recommendation.status = 'rejected'
            recommendation.reviewed_at = datetime.utcnow()
            session.add(recommendation)
        if report:
            report.status = 'rejected'
            report.provider_status = 'rejected'
            report.provider_comments = prediction.provider_comments
            report.generated_at = datetime.utcnow()
            session.add(report)
        patient_user_id = session.execute(select(PatientProfile.user_id).where(PatientProfile.id == prediction.patient_id)).scalar_one()
        add_notification(
            session,
            patient_user_id,
            'Prediction reviewed',
            f'Your provider {user.full_name} reviewed your disease prediction.' + (f' Comment: {prediction.provider_comments}' if prediction.provider_comments else ''),
        )
        session.commit()

    return {
        'status': 'ok',
        'prediction_id': prediction.id,
        'provider_feedback': normalize_approval_status(prediction.provider_feedback or prediction.status),
        'status_value': normalize_approval_status(prediction.status or prediction.provider_feedback),
        'provider_comments': prediction.provider_comments,
        'feedback_date': prediction.feedback_date.isoformat(),
        'report': report_payload,
    }


@app.get('/patient/predictions/{prediction_id}/recommendations')
def get_prediction_recommendations(prediction_id: int, authorization: str = Header(None), session=Depends(get_session)):
    """Get recommendations for a specific prediction.
    
    MILESTONE 3: Endpoint for retrieving AI-generated recommendations associated with a prediction.
    """
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role != 'patient':
        raise HTTPException(status_code=403, detail='Forbidden')

    # Verify patient owns this prediction
    profile = session.execute(select(PatientProfile).where(PatientProfile.user_id == user.id)).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail='Patient profile not found')

    prediction = session.execute(
        select(DiseasePrediction).where(DiseasePrediction.id == prediction_id, DiseasePrediction.patient_id == profile.id)
    ).scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=404, detail='Prediction not found or does not belong to you')

    # Get all recommendations for this prediction
    recs = session.execute(
        select(Recommendation).where(Recommendation.prediction_id == prediction_id).order_by(Recommendation.created_at.desc())
    ).scalars().all()

    return {
        'prediction_id': prediction_id,
        'predicted_disease': prediction.predicted_disease,
        'confidence': prediction.confidence,
        'prediction_date': prediction.prediction_date.isoformat(),
        'status': prediction.status,
        'recommendations': [
            {
                'id': r.id,
                'recommendation': r.recommendation,
                'medicine': r.medicine,
                'priority': r.priority,
                'recommendation_type': r.recommendation_type,
                'status': r.status,
                'ai_generated': r.ai_generated,
                'provider_comments': r.provider_comments,
                'created_at': r.created_at.isoformat(),
                'reviewed_at': r.reviewed_at.isoformat() if r.reviewed_at else None,
            }
            for r in recs
        ]
    }


@app.get('/provider/predictions/{prediction_id}/recommendations')
def get_provider_prediction_recommendations(prediction_id: int, authorization: str = Header(None), session=Depends(get_session)):
    """Provider endpoint to view recommendations generated for a specific prediction.
    
    MILESTONE 3: Allows providers to view AI-generated recommendations for their approved predictions.
    """
    token = None
    if authorization and authorization.startswith('Bearer '):
        token = authorization.split(' ', 1)[1]
    user = get_authenticated_user(token, session)
    if user.role not in ('doctor', 'provider'):
        raise HTTPException(status_code=403, detail='Forbidden')

    prediction = session.execute(
        select(DiseasePrediction).where(DiseasePrediction.id == prediction_id)
    ).scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=404, detail='Prediction not found')

    # Get all recommendations for this prediction
    recs = session.execute(
        select(Recommendation).where(Recommendation.prediction_id == prediction_id).order_by(Recommendation.created_at.desc())
    ).scalars().all()

    patient_profile = session.execute(
        select(PatientProfile).where(PatientProfile.id == prediction.patient_id)
    ).scalar_one_or_none()

    return {
        'prediction_id': prediction_id,
        'patient_id': prediction.patient_id,
        'patient_name': session.execute(select(User.full_name).where(User.id == patient_profile.user_id)).scalar() if patient_profile else None,
        'predicted_disease': prediction.predicted_disease,
        'confidence': prediction.confidence,
        'prediction_date': prediction.prediction_date.isoformat(),
        'status': prediction.status,
        'provider_feedback': normalize_approval_status(prediction.provider_feedback or prediction.status),
        'recommendations': [
            {
                'id': r.id,
                'recommendation': r.recommendation,
                'medicine': r.medicine,
                'priority': r.priority,
                'recommendation_type': r.recommendation_type,
                'status': r.status,
                'ai_generated': r.ai_generated,
                'provider_comments': r.provider_comments,
                'created_at': r.created_at.isoformat(),
                'reviewed_at': r.reviewed_at.isoformat() if r.reviewed_at else None,
            }
            for r in recs
        ]
    }


@app.get('/')
def root():
    return {'status':'ok'}
