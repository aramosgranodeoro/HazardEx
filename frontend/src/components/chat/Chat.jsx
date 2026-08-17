import React, { useState, useRef, useEffect } from "react";
import ApiService from "../../ApiServices/ApiServices.js";
import "./Chat.css";

const VLM_MODELS = [
  { id: "llava7b", label: "llava7b" },
  { id: "qwen3.5", label: "qwen3.5" },
];

export default function Chat({ threadId, setThreadId, messages, setMessages }) {
  const [input, setInput] = useState("");
  const [vlmModel, setVlmModel] = useState(VLM_MODELS[0].id);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleAttachClick = () => fileInputRef.current?.click();

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = "";

    const previewUrl = URL.createObjectURL(file);
    const userMsgId = Date.now();

    setMessages((prev) => [...prev, { id: userMsgId, role: "user", image: previewUrl, fileName: file.name }]);
    setIsLoading(true);

    try {
      const data = await ApiService.analyze(file);
      setThreadId(data.thread_id);
      setMessages((prev) => [...prev, { id: userMsgId + 1, role: "assistant", text: data.analysis }]);
    } catch (err) {
      setMessages((prev) => [...prev, { id: userMsgId + 1, role: "assistant", text: "Error al analizar el archivo. Inténtalo de nuevo.", isError: true }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;

    const userMsgId = Date.now();
    setMessages((prev) => [...prev, { id: userMsgId, role: "user", text }]);
    setInput("");
    setIsLoading(true);

    try {
      const data = await ApiService.query(threadId, text);
      if (!threadId) setThreadId(data.thread_id);
      setMessages((prev) => [...prev, { id: userMsgId + 1, role: "assistant", text: data.response }]);
    } catch (err) {
      setMessages((prev) => [...prev, { id: userMsgId + 1, role: "assistant", text: "Error al enviar el mensaje.", isError: true }]);
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
      <div className="hx-chat-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`hx-msg-row ${msg.role}`}>
            <div className="hx-msg-bubble">
              <div className="hx-msg-content">
                {msg.role === "assistant"}

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
            <i className="fa-solid fa-paperclip"></i>
          </button>
          <input
            type="text"
            className="hx-text-input"
            placeholder="Escribe un mensaje..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />

          {/* <div className="hx-select-group">
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
          </div> */}
          <button
            type="button"
            className="hx-send-btn"
            onClick={handleSend}
            disabled={isLoading}
          >
            <i class="fa-regular fa-paper-plane"></i>
          </button>
        </div>
      </footer>
    </section>
  );
}