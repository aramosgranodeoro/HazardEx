import React, { useState, useEffect, useRef } from "react";
import ApiService from "../../ApiServices/ApiServices.js"; /*[cite: 5] */
import "./Sidebar.css";
import HazardEx from '../../assets/HazardEx.png' /*[cite: 5] */

export default function Sidebar({ onNewChat, onSelectConversation, activeThreadId }) {
  const [historyOpen, setHistoryOpen] = useState(false); /*[cite: 5] */
  const [ragOpen, setRagOpen] = useState(false); /*[cite: 5] */
  const [conversations, setConversations] = useState([]); /*[cite: 5] */
  const [ragDocs, setRagDocs] = useState([]); /*[cite: 5] */
  const [loadingHistory, setLoadingHistory] = useState(false); /*[cite: 5] */
  const [loadingRag, setLoadingRag] = useState(false); /*[cite: 5] */
  const ragFileInputRef = useRef(null); /*[cite: 5] */

  const loadConversations = async () => {
    setLoadingHistory(true); /*[cite: 5] */
    try {
      setConversations(await ApiService.listConversations()); /*[cite: 5] */
    } catch (err) {
      console.error("Error cargando historial:", err); /*[cite: 5] */
    } finally {
      setLoadingHistory(false); /*[cite: 5] */
    }
  };

  const loadRagDocs = async () => {
    setLoadingRag(true); /*[cite: 5] */
    try {
      setRagDocs(await ApiService.listRagDocuments()); /*[cite: 5] */
    } catch (err) {
      console.error("Error cargando documentos RAG:", err); /*[cite: 5] */
    } finally {
      setLoadingRag(false); /*[cite: 5] */
    }
  };

  useEffect(() => {
    loadConversations(); /*[cite: 5] */
  }, []); /*[cite: 5] */

  const toggleHistory = () => {
    const next = !historyOpen; /*[cite: 5] */
    setHistoryOpen(next); /*[cite: 5] */
    if (next) loadConversations(); /*[cite: 5] */
  };

  const toggleRag = () => {
    const next = !ragOpen; /*[cite: 5] */
    setRagOpen(next); /*[cite: 5] */
    if (next) loadRagDocs(); /*[cite: 5] */
  };

  const handleDeleteConversation = async (e, threadId) => {
    e.stopPropagation(); /*[cite: 5] */
    if (!window.confirm("¿Eliminar esta conversación?")) return; /*[cite: 5] */
    try {
      await ApiService.deleteConversation(threadId); /*[cite: 5] */
      setConversations((prev) => prev.filter((c) => c.thread_id !== threadId)); /*[cite: 5] */
      if (threadId === activeThreadId) onNewChat?.(); /*[cite: 5] */
    } catch (err) {
      console.error("Error eliminando conversación:", err); /*[cite: 5] */
    }
  };

  const handleRagFileChange = async (e) => {
    const file = e.target.files[0]; /*[cite: 5] */
    if (!file) return; /*[cite: 5] */
    e.target.value = ""; /*[cite: 5] */
    try {
      await ApiService.uploadRagDocument(file); /*[cite: 5] */
      loadRagDocs(); /*[cite: 5] */
    } catch (err) {
      alert(err?.response?.data?.detail || "Error al subir el documento"); /*[cite: 5] */
    }
  };

  const handleDeleteRag = async (filename) => {
    if (!window.confirm(`¿Eliminar "${filename}" del RAG?`)) return; /*[cite: 5] */
    try {
      await ApiService.deleteRagDocument(filename); /*[cite: 5] */
      setRagDocs((prev) => prev.filter((f) => f !== filename)); /*[cite: 5] */
    } catch (err) {
      console.error("Error eliminando documento:", err); /*[cite: 5] */
    }
  };

  return (
    <aside className="hx-sidebar">
      <div>
        <img src={HazardEx} alt="HazardEx Logo" className="hx-logo-image" />
      </div>

      <nav className="hx-sidebar-nav">
        {/* AQUÍ ESTÁ EL CAMBIO PRINCIPAL */}
        <button 
          type="button" 
          className="hx-nav-item" 
          onClick={() => {
            onNewChat?.();           // 1. Limpia el chat actual en App.jsx
            loadConversations();     // 2. Vuelve a pedirle el historial actualizado a la API
          }}
        >
          <span className="hx-nav-icon"><i className="fa-regular fa-square-plus"></i></span>
          <span className="hx-nav-label">Nuevo chat</span>
        </button>

        <button type="button" className={`hx-nav-item ${historyOpen ? "active" : ""}`} onClick={toggleHistory}>
          <span className="hx-nav-icon"><i className="fa-regular fa-comments"></i></span>
          <span className="hx-nav-label">Historial</span>
          <i className={`fa-solid fa-chevron-down hx-chevron ${historyOpen ? "open" : ""}`}></i>
        </button>
        {historyOpen && (
          <div className="hx-dropdown-panel">
            {loadingHistory && <div className="hx-dropdown-empty">Cargando...</div>}
            {!loadingHistory && conversations.length === 0 && (
              <div className="hx-dropdown-empty">Sin conversaciones</div>
            )}
            {conversations.map((c) => (
              <div
                key={c.thread_id}
                className={`hx-dropdown-item ${c.thread_id === activeThreadId ? "active" : ""}`}
                onClick={() => onSelectConversation?.(c.thread_id)}
              >
                <span className="hx-dropdown-item-title" title={c.title}>{c.title}</span>
                <button
                  type="button"
                  className="hx-dropdown-item-delete"
                  onClick={(e) => handleDeleteConversation(e, c.thread_id)}
                  title="Eliminar conversación"
                >
                  <i className="fa-solid fa-trash"></i>
                </button>
              </div>
            ))}
          </div>
        )}

        <button type="button" className={`hx-nav-item ${ragOpen ? "active" : ""}`} onClick={toggleRag}>
          <span className="hx-nav-icon"><i className="fa-solid fa-book"></i></span>
          <span className="hx-nav-label">Fuentes</span>
          <i className={`fa-solid fa-chevron-down hx-chevron ${ragOpen ? "open" : ""}`}></i>
        </button>
        {ragOpen && (
          <div className="hx-dropdown-panel">
            <input
              type="file"
              accept=".pdf,.docx,.txt,.md"
              ref={ragFileInputRef}
              style={{ display: "none" }}
              onChange={handleRagFileChange}
            />
            <button type="button" className="hx-dropdown-add-btn" onClick={() => ragFileInputRef.current?.click()}>
              <i className="fa-solid fa-plus"></i> Añadir fuente
            </button>
            {loadingRag && <div className="hx-dropdown-empty">Cargando...</div>}
            {!loadingRag && ragDocs.length === 0 && (
              <div className="hx-dropdown-empty">Sin documentos</div>
            )}
            {ragDocs.map((filename) => (
              <div key={filename} className="hx-dropdown-item">
                <span className="hx-dropdown-item-title" title={filename}>{filename}</span>
                <button
                  type="button"
                  className="hx-dropdown-item-delete"
                  onClick={() => handleDeleteRag(filename)}
                  title="Eliminar documento"
                >
                  <i className="fa-solid fa-trash"></i>
                </button>
              </div>
            ))}
          </div>
        )}
      </nav>
    </aside>
  );
}