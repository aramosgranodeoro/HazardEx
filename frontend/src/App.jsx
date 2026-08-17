import React, { useState } from "react";
import Sidebar from "./components/Sidebar/Sidebar.jsx";
import Chat from "./components/Chat/Chat.jsx";
import ApiService from "./ApiServices/ApiServices.js";

export default function App() {
  const [threadId, setThreadId] = useState(null);
  const [messages, setMessages] = useState([]);

  const handleNewChat = () => {
    setThreadId(null);
    setMessages([]);
  };

  const handleSelectConversation = async (selectedThreadId) => {
    if (selectedThreadId === threadId) return;
    try {
      const data = await ApiService.getConversation(selectedThreadId);
      const loaded = [];

      if (data.has_media) {
        loaded.push({
          id: `media-${selectedThreadId}`,
          role: "user",
          image: ApiService.getMediaUrl(selectedThreadId),
          fileName: data.media_type === "video" ? "vídeo" : "imagen",
        });
      }

      data.messages.forEach((m, i) => loaded.push({ id: `${selectedThreadId}-${i}`, role: m.role, text: m.text }));

      setThreadId(selectedThreadId);
      setMessages(loaded);
    } catch (err) {
      alert("No se pudo cargar la conversación.");
    }
  };

  return (
    <div className="hx-app">
      <Sidebar
        onNewChat={handleNewChat}
        onSelectConversation={handleSelectConversation}
        activeThreadId={threadId}
      />
      <Chat threadId={threadId} setThreadId={setThreadId} messages={messages} setMessages={setMessages} />
    </div>
  );
}