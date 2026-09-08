import React, { useEffect, useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import DashboardLayout from '../components/DashboardLayout'
import Home from './provider/Home'
import PatientList from './provider/PatientList'
import PatientHistory from './provider/PatientHistory'
import SymptomAnalysis from './provider/SymptomAnalysis'
import DiseasePrediction from './provider/DiseasePrediction'
import RiskAssessment from './provider/RiskAssessment'
import Reports from './provider/Reports'
import Analytics from './provider/Analytics'
import Recommendations from './provider/Recommendations'
import Settings from './provider/Settings'
import Profile from './provider/Profile'
import Chat from './Chat'
import { fetchProviderDashboard } from '../api/dashboard'

const menu = [
  {to:'/dashboard/provider', label:'Dashboard'},
  {to:'/dashboard/provider/patients', label:'Patient List'},
  {to:'/dashboard/provider/history', label:'Patient History'},
  {to:'/dashboard/provider/symptoms', label:'Symptom Analysis'},
  {to:'/dashboard/provider/prediction', label:'Disease Prediction'},
  {to:'/dashboard/provider/risk', label:'Risk Assessment'},
  {to:'/dashboard/provider/reports', label:'Reports'},
  {to:'/dashboard/provider/analytics', label:'Analytics'},
  {to:'/dashboard/provider/recommendations', label:'Recommendations'},
  {to:'/dashboard/provider/chat', label:'Chat'},
  {to:'/dashboard/provider/profile', label:'Profile'},
  {to:'/dashboard/provider/settings', label:'Settings'},
]

export default function ProviderDashboard(){
  const [dashboardData, setDashboardData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let canceled = false
    async function load(showLoading = false) {
      if (showLoading) setLoading(true)
      try {
        const payload = await fetchProviderDashboard()
        if (!canceled) setDashboardData(payload)
      } catch (err) {
        if (!canceled) setError(err.message || 'Unable to load provider dashboard data')
      } finally {
        if (!canceled) setLoading(false)
      }
    }
    load(true)
    const intervalId = setInterval(() => {
      load()
    }, 15000)
    return () => { canceled = true; clearInterval(intervalId) }
  }, [])

  if (loading) {
    return (
      <div className="container dashboard-page">
        <div className="dashboard-card">
          <h1>Loading provider dashboard...</h1>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container dashboard-page">
        <div className="dashboard-card">
          <h1>Provider Dashboard</h1>
          <div className="status error">{error}</div>
        </div>
      </div>
    )
  }

  return (
    <DashboardLayout menu={menu} title="Provider Dashboard" user={dashboardData?.user}>
      <Routes>
        <Route index element={<Home dashboardData={dashboardData} />} />
        <Route path="/" element={<Home dashboardData={dashboardData} />} />
        <Route path="patients" element={<PatientList dashboardData={dashboardData} />} />
        <Route path="history" element={<PatientHistory dashboardData={dashboardData} />} />
        <Route path="symptoms" element={<SymptomAnalysis dashboardData={dashboardData} />} />
        <Route path="prediction" element={<DiseasePrediction dashboardData={dashboardData} />} />
        <Route path="risk" element={<RiskAssessment dashboardData={dashboardData} />} />
        <Route path="reports" element={<Reports dashboardData={dashboardData} />} />
        <Route path="analytics" element={<Analytics dashboardData={dashboardData} />} />
        <Route path="recommendations" element={<Recommendations dashboardData={dashboardData} />} />
        <Route path="chat" element={<Chat />} />
        <Route path="profile" element={<Profile dashboardData={dashboardData} />} />
        <Route path="settings" element={<Settings dashboardData={dashboardData} />} />
        <Route path="*" element={<Navigate to="/dashboard/provider" replace />} />
      </Routes>
    </DashboardLayout>
  )
}
