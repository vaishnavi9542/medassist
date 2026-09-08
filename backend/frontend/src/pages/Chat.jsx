import React, { useEffect, useState } from 'react'
import { apiFetch } from '../api/client'

function formatTime(value) {
  if (!value) return ''
  return new Date(value).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })
}

export default function Chat() {
  const [contacts, setContacts] = useState([])
  const [selectedId, setSelectedId] = useState('')
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(true)
  const [status, setStatus] = useState(null)

  async function loadContacts() {
    const response = await apiFetch('/chat/contacts')
    setContacts(response.contacts || [])
    setSelectedId((current) => current || String(response.contacts?.[0]?.id || ''))
  }

  async function loadMessages(contactId = selectedId) {
    if (!contactId) {
      setMessages([])
      return
    }
    const response = await apiFetch(`/chat/messages/${contactId}`)
    setMessages(response.messages || [])
  }

  useEffect(() => {
    let active = true
    setLoading(true)
    loadContacts()
      .catch((error) => active && setStatus({ type: 'error', message: error.message }))
      .finally(() => active && setLoading(false))
    return () => { active = false }
  }, [])

  useEffect(() => {
    loadMessages().catch((error) => setStatus({ type: 'error', message: error.message }))
    if (!selectedId) return undefined
    const intervalId = setInterval(() => {
      loadMessages().catch(() => {})
    }, 5000)
    return () => clearInterval(intervalId)
  }, [selectedId])

  async function handleSubmit(event) {
    event.preventDefault()
    const body = draft.trim()
    if (!body || !selectedId) return
    setStatus(null)
    try {
      await apiFetch('/chat/messages', {
        method: 'POST',
        body: JSON.stringify({ recipient_id: Number(selectedId), body }),
      })
      setDraft('')
      await loadMessages()
    } catch (error) {
      setStatus({ type: 'error', message: error.message })
    }
  }

  return (
    <div className="dashboard-card chat-page">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Secure care communication</span>
          <h1>Chat</h1>
          <p>Exchange messages with your healthcare team.</p>
        </div>
      </div>
      {status && <div className={`status ${status.type}`}>{status.message}</div>}
      {loading ? <p>Loading contacts...</p> : contacts.length === 0 ? (
        <div className="empty-state">No patient or healthcare provider contacts are available yet.</div>
      ) : (
        <div className="chat-layout">
          <aside className="chat-contacts">
            <h2>Contacts</h2>
            {contacts.map((contact) => (
              <button
                type="button"
                key={contact.id}
                className={`chat-contact ${String(contact.id) === String(selectedId) ? 'active' : ''}`}
                onClick={() => setSelectedId(String(contact.id))}
              >
                <strong>{contact.full_name}</strong>
                <span>{contact.role === 'doctor' ? 'Healthcare provider' : 'Patient'}</span>
              </button>
            ))}
          </aside>
          <section className="chat-thread">
            <div className="chat-thread-header">
              <h2>{contacts.find((contact) => String(contact.id) === String(selectedId))?.full_name || 'Conversation'}</h2>
            </div>
            <div className="chat-messages">
              {messages.length === 0 ? <p className="empty-state">Start the conversation.</p> : messages.map((message) => (
                <article key={message.id} className={`chat-bubble ${message.sender_id === Number(selectedId) ? 'incoming' : 'outgoing'}`}>
                  <p>{message.body}</p>
                  <small>{formatTime(message.created_at)}</small>
                </article>
              ))}
            </div>
            <form className="chat-compose" onSubmit={handleSubmit}>
              <textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Write a message..." maxLength={4000} rows={3} />
              <button className="btn primary" type="submit" disabled={!draft.trim() || !selectedId}>Send</button>
            </form>
          </section>
        </div>
      )}
    </div>
  )
}
