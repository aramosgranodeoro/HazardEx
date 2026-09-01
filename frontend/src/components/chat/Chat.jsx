import React, { useState, useRef, useEffect } from "react";
import ApiService from "../../ApiService/ApiService.js"; 
import "./Chat.css";
import ReactMarkdown from 'react-markdown'; 
import remarkGfm from 'remark-gfm'; 

export default function Chat({ threadId, setThreadId, messages, setMessages }) {
  const [input, setInput] = useState(""); 
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

  const isVideo = file.type.startsWith("video");
  const userMsgId = Date.now();

  const previewUrl = isVideo ? null : URL.createObjectURL(file);
  setMessages((prev) => [...prev, { id: userMsgId, role: "user", image: previewUrl, videoUrl: null, fileName: file.name, isVideo, isLoadingMedia: isVideo }]);
  setIsLoading(true); 

  try {
    const data = await ApiService.analyze(file, threadId);
    if (!threadId) setThreadId(data.thread_id);

    if (isVideo) {
      setMessages((prev) => prev.map((m) =>
        m.id === userMsgId
          ? {
              ...m,
              image: ApiService.getMediaUrl(data.thread_id, `${data.media_id}_thumb`),
              videoUrl: ApiService.getMediaUrl(data.thread_id, data.media_id),
              isLoadingMedia: false,
            }
          : m
      ));
    }

    const annotatedImage = data.annotated_media_id
      ? ApiService.getMediaUrl(
          data.thread_id,
          data.annotated_media_id
        )
      : null;

   setMessages((prev) => {
  const newMessages = [
    ...prev,
    {
      id: userMsgId + 1,
      role: "assistant",
      text: data.analysis,
    }
  ];

  if (annotatedImage) {
    newMessages.push({
      id: userMsgId + 2,
      role: "assistant",
      annotatedImage,
    });
  }

  return newMessages;
});
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

 const handleAnnotatedImageError = (messageId) => {
  setMessages((prev) =>
    prev.filter((msg) => msg.id !== messageId)
  );
};

  return (
    <section className="hx-chat">
      <div className="hx-chat-messages">
        
        {/* LÓGICA DE BIENVENIDA O MENSAJES */}
        {messages.length === 0 ? (
          <div className="hx-welcome-container">
            <h2 className="hx-welcome-title">¿En qué puedo ayudarte hoy?</h2>
            <p className="hx-welcome-subtitle">Sube una imagen o vídeo para comenzar el análisis de amenazas.</p>
          </div>
        ) : (
          /* Renderizado normal de mensajes si la lista no está vacía */
          messages.map((msg) => (
            <div key={msg.id} className={`hx-msg-row ${msg.role}`}>
              <div className="hx-msg-bubble">
                <div className="hx-msg-content">
                  {msg.image && (
                    <div className="hx-image-attachment">
                      {msg.isVideo ? (
                        <video src={msg.videoUrl} poster={msg.image} controls className="hx-chat-video" />
                      ) : (
                        <img src={msg.image} alt={msg.fileName || "adjunto"} className="hx-chat-image" />
                      )}
                    </div>
                  )}

                  {msg.annotatedImage && (
                    <div className="hx-image-attachment">
                      <img
                        src={msg.annotatedImage}
                        alt="Detecciones"
                        className="hx-chat-image"
                        onError={() => handleAnnotatedImageError(msg.id)}
                      />
                    </div>
                  )}
                  {msg.text && (
                    msg.role === "assistant" ? (
                      <div className="hx-markdown">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            a: ({ node, ...props }) => (
                              <a {...props} target="_blank" rel="noopener noreferrer" className="hx-link" />
                            ),
                          }}
                        >
                          {msg.text}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <span>{msg.text}</span>
                    )
                  )}
                </div>
              </div>
            </div>
          ))
        )}

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

          <button
            type="button"
            className="hx-send-btn"
            onClick={handleSend}
            disabled={isLoading}
          >
            <i className="fa-regular fa-paper-plane"></i>
          </button>
        </div>
      </footer>
    </section>
  );
}