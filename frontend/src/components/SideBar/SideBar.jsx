import React, { useState, useEffect, useRef } from "react";
import ApiService from "../../ApiService/ApiService.js";  
import "./Sidebar.css";
import HazardEx from '../../assets/HazardEx.png'  

export default function Sidebar({ onNewChat, onSelectConversation, activeThreadId }) {
  const [historyOpen, setHistoryOpen] = useState(false);  
  const [ragOpen, setRagOpen] = useState(false);  
  const [conversations, setConversations] = useState([]);  
  const [ragDocs, setRagDocs] = useState([]);  
  const [loadingHistory, setLoadingHistory] = useState(false);  
  const [loadingRag, setLoadingRag] = useState(false);  
  const ragFileInputRef = useRef(null);  

  const loadConversations = async () => {
    setLoadingHistory(true);  
    try {
      setConversations(await ApiService.listConversations());  
    } catch (err) {
      console.error("Error cargando historial:", err);  
    } finally {
      setLoadingHistory(false);  
    }
  };

  const loadRagDocs = async () => {
    setLoadingRag(true);  
    try {
      setRagDocs(await ApiService.listRagDocuments());  
    } catch (err) {
      console.error("Error cargando documentos RAG:", err);  
    } finally {
      setLoadingRag(false);  
    }
  };

  useEffect(() => {
    loadConversations();  
  }, []);  

  const toggleHistory = () => {
    const next = !historyOpen;  
    setHistoryOpen(next);  
    if (next) loadConversations();  
  };

  const toggleRag = () => {
    const next = !ragOpen;  
    setRagOpen(next);  
    if (next) loadRagDocs();  
  };

  const handleDeleteConversation = async (e, threadId) => {
    e.stopPropagation();  
    if (!window.confirm("¿Eliminar esta conversación?")) return;  
    try {
      await ApiService.deleteConversation(threadId);  
      setConversations((prev) => prev.filter((c) => c.thread_id !== threadId));  
      if (threadId === activeThreadId) onNewChat?.();  
    } catch (err) {
      console.error("Error eliminando conversación:", err);  
    }
  };

  const handleRagFileChange = async (e) => {
    const file = e.target.files[0];  
    if (!file) return;  
    e.target.value = "";  
    try {
      await ApiService.uploadRagDocument(file);  
      loadRagDocs();  
    } catch (err) {
      alert(err?.response?.data?.detail || "Error al subir el documento");  
    }
  };

  const handleDeleteRag = async (filename) => {
    if (!window.confirm(`¿Eliminar "${filename}" del RAG?`)) return;  
    try {
      await ApiService.deleteRagDocument(filename);  
      setRagDocs((prev) => prev.filter((f) => f !== filename));  
    } catch (err) {
      console.error("Error eliminando documento:", err);  
    }
  };

  return (
    <aside className="hx-sidebar">
     <div className="hx-sidebar-logo">
      <span className="hx-logo-text">Hazard<span className="hx-logo-accent">Ex</span></span>
    </div>

      <nav className="hx-sidebar-nav">
        <button 
          type="button" 
          className="hx-nav-item" 
          onClick={() => {
            onNewChat?.();           
            loadConversations();    
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