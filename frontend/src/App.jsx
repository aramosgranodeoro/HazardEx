import React, { useState } from "react";
import Sidebar from "./components/Sidebar/Sidebar.jsx";
import Chat from "./components/Chat/Chat.jsx";
import ApiService from "./ApiService/ApiService.js";

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

    const loaded = data.items.map((item, i) => {
      if (item.type === "media") {
        return {
          id: `media-${selectedThreadId}-${item.media_id}`,
          role: "user",
          image: ApiService.getMediaUrl(selectedThreadId, item.media_id),
          fileName: item.media_type === "video" ? "vídeo" : "imagen",
        };
      }
      return {
        id: `msg-${selectedThreadId}-${i}`,
        role: item.role,
        text: item.text,
      };
    });

    setMessages(loaded);
    setThreadId(selectedThreadId);
  } catch (err) {
    console.error("Error cargando conversación:", err);
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