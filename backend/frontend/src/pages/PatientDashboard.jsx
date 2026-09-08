import React, { useEffect, useState, useCallback } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import DashboardLayout from '../components/DashboardLayout'
import Home from './patient/Home'
import MyProfile from './patient/MyProfile'
import MedicalHistory from './patient/MedicalHistory'
import SymptomEntry from './patient/SymptomEntry'
import DiseasePrediction from './patient/DiseasePrediction'
import RiskAssessment from './patient/RiskAssessment'
import Reports from './patient/Reports'
import Analytics from './patient/Analytics'
import Recommendations from './patient/Recommendations'
import Settings from './patient/Settings'
import Chat from './Chat'
import { fetchPatientDashboard } from '../api/dashboard'

const menu = [
  {to:'/dashboard/patient', label:'Dashboard'},
  {to:'/dashboard/patient/profile', label:'My Profile'},
  {to:'/dashboard/patient/history', label:'Medical History'},
  {to:'/dashboard/patient/symptoms', label:'Symptom Entry'},
  {to:'/dashboard/patient/prediction', label:'Disease Prediction'},
  {to:'/dashboard/patient/risk', label:'Risk Assessment'},
  {to:'/dashboard/patient/reports', label:'Reports'},
  {to:'/dashboard/patient/analytics', label:'Analytics'},
  {to:'/dashboard/patient/recommendations', label:'Recommendations'},
  {to:'/dashboard/patient/chat', label:'Chat'},
  {to:'/dashboard/patient/settings', label:'Settings'},
]

export default function PatientDashboard(){
  const [dashboardData, setDashboardData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const reloadDashboard = useCallback(async ({ showLoading = false } = {}) => {
    if (showLoading) setLoading(true)
    try {
      const payload = await fetchPatientDashboard()
      setDashboardData(payload)
      setError('')
    } catch (err) {
      setError(err.message || 'Unable to load dashboard data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    reloadDashboard({ showLoading: true })
    const intervalId = setInterval(() => {
      reloadDashboard()
    }, 15000)
    return () => clearInterval(intervalId)
  }, [reloadDashboard])

  if (loading) {
    return (
      <div className="container dashboard-page">
        <div className="dashboard-card">
          <h1>Loading patient dashboard...</h1>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container dashboard-page">
        <div className="dashboard-card">
          <h1>Patient Dashboard</h1>
          <div className="status error">{error}</div>
        </div>
      </div>
    )
  }

  return (
    <DashboardLayout menu={menu} title="Patient Dashboard" user={dashboardData?.user}>
      <Routes>
        <Route index element={<Home dashboardData={dashboardData} />} />
        <Route path="/" element={<Home dashboardData={dashboardData} />} />
        <Route path="profile" element={<MyProfile dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="history" element={<MedicalHistory dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="symptoms" element={<SymptomEntry dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="prediction" element={<DiseasePrediction dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="risk" element={<RiskAssessment dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="reports" element={<Reports dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="analytics" element={<Analytics dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="recommendations" element={<Recommendations dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="chat" element={<Chat />} />
        <Route path="settings" element={<Settings dashboardData={dashboardData} reloadDashboard={reloadDashboard} />} />
        <Route path="*" element={<Navigate to="/dashboard/patient" replace />} />
      </Routes>
    </DashboardLayout>
  )
}
