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
  try {
    const data = await ApiService.getConversation(selectedThreadId);

    const loaded = data.items.flatMap((item, i) => {
      if (item.type === "media") {
        const isVideo = item.media_type === "video";

        const mediaMessage = {
          id: `media-${selectedThreadId}-${item.media_id}`,
          role: "user",

          image: isVideo
            ? ApiService.getMediaUrl(
                selectedThreadId,
                `${item.media_id}_thumb`
              )
            : ApiService.getMediaUrl(
                selectedThreadId,
                item.media_id
              ),

          videoUrl: isVideo
            ? ApiService.getMediaUrl(
                selectedThreadId,
                item.media_id
              )
            : null,

          isVideo,
          fileName: isVideo ? "vídeo" : "imagen",
        };

        const annotatedMessage = {
          id: `annotated-${selectedThreadId}-${item.media_id}`,
          role: "assistant",
          annotatedImage: ApiService.getMediaUrl(
            selectedThreadId,
            `${item.media_id}_annotated`
          ),
          isAnnotated: true,
        };

        return [
          mediaMessage,
          annotatedMessage,
        ];
      }

      return [
        {
          id: `msg-${selectedThreadId}-${i}`,
          role: item.role,
          text: item.text,
        }
      ];
    });

    setThreadId(selectedThreadId);
    setMessages(loaded);

  } catch (err) {
    console.error(
      "Error cargando conversación:",
      err
    );
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