import React, { useState, useRef, useEffect } from "react";
import ApiService from "../../ApiServices/ApiServices.js";
import "./Chat.css";

const VLM_MODELS = [
  { id: "llava7b", label: "llava7b" },
  { id: "llava7b-es", label: "llava7b (ES)" },
  { id: "salamandra-vl", label: "salamandra-vl-7b" },
];

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [vlmModel, setVlmModel] = useState(VLM_MODELS[0].id);
  const [threadId, setThreadId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = ""; // permite volver a subir el mismo archivo si hace falta

    const previewUrl = URL.createObjectURL(file);
    const userMsgId = Date.now();

    // 1. Mostrar la imagen en el chat inmediatamente
    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        role: "user",
        image: previewUrl,
        fileName: file.name,
      },
    ]);

    // 2. Burbuja de espera
    setIsLoading(true);

    try {
      const data = await ApiService.analyze(file);
      // data: { thread_id, analysis }
      setThreadId(data.thread_id);

      setMessages((prev) => [
        ...prev,
        {
          id: userMsgId + 1,
          role: "assistant",
          text: data.analysis,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: userMsgId + 1,
          role: "assistant",
          text: "Error al analizar el archivo. Inténtalo de nuevo.",
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;

    const userMsgId = Date.now();
    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", text },
    ]);
    setInput("");
    setIsLoading(true);

    try {
      const data = await ApiService.query(threadId, text);
      setMessages((prev) => [
        ...prev,
        { id: userMsgId + 1, role: "assistant", text: data.response },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: userMsgId + 1,
          role: "assistant",
          text: "Error al enviar el mensaje.",
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <section className="hx-chat">
      <header className="hx-chat-header">
        <h1>AI Assistant</h1>
      </header>

      <div className="hx-chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`hx-msg-row ${msg.role}`}>
            <div className="hx-msg-bubble">
              {msg.role === "assistant" && (
                <span className="hx-bot-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M12 2 2 20h20L12 2z"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinejoin="round"
                    />
                  </svg>
                </span>
              )}
              <div className="hx-msg-content">
                {msg.role === "assistant" && (
                  <span className="hx-msg-sender">HazardEx: </span>
                )}

                {msg.image && (
                  <div className="hx-image-attachment">
                    <img
                      src={msg.image}
                      alt={msg.fileName || "adjunto"}
                      className="hx-chat-image"
                    />
                  </div>
                )}

                {msg.text && <span>{msg.text}</span>}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="hx-msg-row assistant">
            <div className="hx-msg-bubble hx-msg-loading">
              <span className="hx-bot-icon">
                <svg viewBox="0 0 24 24" fill="none">
                  <path
                    d="M12 2 2 20h20L12 2z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinejoin="round"
                  />
                </svg>
              </span>
              <div className="hx-msg-content">
                <span className="hx-typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <footer className="hx-chat-footer">
        <div className="hx-input-row">
          <input
            type="file"
            accept="image/*,video/*"
            ref={fileInputRef}
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <button
            type="button"
            className="hx-icon-btn"
            title="Adjuntar archivo"
            onClick={handleAttachClick}
            disabled={isLoading}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21.44 11.05 12.25 20.24a5.5 5.5 0 0 1-7.78-7.78l9.19-9.19a3.67 3.67 0 0 1 5.19 5.19l-9.2 9.19a1.83 1.83 0 0 1-2.6-2.6l8.49-8.48" />
            </svg>
          </button>
          <input
            type="text"
            className="hx-text-input"
            placeholder="Escribe tu pregunta o comando..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
        </div>

        <div className="hx-controls-row">
          <div className="hx-select-group">
            <label>VLM Model</label>
            <select
              className="hx-select"
              value={vlmModel}
              onChange={(e) => setVlmModel(e.target.value)}
            >
              {VLM_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            className="hx-send-btn"
            onClick={handleSend}
            disabled={isLoading}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="m22 2-7 20-4-9-9-4Z" />
              <path d="M22 2 11 13" />
            </svg>
          </button>
        </div>
      </footer>
    </section>
  );
}