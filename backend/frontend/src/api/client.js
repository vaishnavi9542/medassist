const fallbackApiBase = 'http://127.0.0.1:8001'
export const API_BASE = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE) || fallbackApiBase

export function getToken() {
  return window.localStorage.getItem('medassist_token')
}

export function setToken(token) {
  window.localStorage.setItem('medassist_token', token)
}

export function removeToken() {
  window.localStorage.removeItem('medassist_token')
}

export async function apiFetch(path, options = {}) {
  const token = getToken()
  const method = (options.method || 'GET').toUpperCase()
  const headers = {
    ...(options.headers || {}),
  }
  if (method !== 'GET' && method !== 'HEAD') {
    headers['Content-Type'] = 'application/json'
  }
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    method,
    headers,
  })
  if (!response.ok) {
    if (response.status === 401) {
      removeToken()
      if (typeof window !== 'undefined' && window.location.pathname.startsWith('/dashboard/')) {
        window.location.assign('/login')
      }
    }
    const text = await response.text()
    throw new Error(text || `HTTP ${response.status}`)
  }
  return response.json()
}

export async function login(payload) {
  const response = await fetch(`${API_BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `HTTP ${response.status}`)
  }
  return response.json()
}
