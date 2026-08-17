import axios from 'axios';

export const analyze = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    const response = await axios.post(
      "http://127.0.0.1:8000/analyze",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data; // { thread_id, analysis }
  } catch (error) {
    console.error("Error analyzing file:", error);
    throw error;
  }
};

export const query = async (threadId, question) => {
  try {
    const response = await axios.post(`http://127.0.0.1:8000/query`, {
      thread_id: threadId || null,
      question: question,
    });
    return response.data; // { thread_id: "...", response: "..." }
  } catch (error) {
    console.error("Error querying:", error);
    throw error;
  }
};

// ---------- Historial ----------
export const listConversations = async () => {
  const response = await axios.get(`http://127.0.0.1:8000/conversations`);
  return response.data.conversations; // [{ thread_id, title, media_type, created_at }]
};

export const getConversation = async (threadId) => {
  const response = await axios.get(`http://127.0.0.1:8000/conversation/${threadId}`);
  return response.data; // { thread_id, messages, has_media, media_type }
};

export const deleteConversation = async (threadId) => {
  const response = await axios.delete(`http://127.0.0.1:8000/conversation/${threadId}`);
  return response.data;
};

export const getMediaUrl = (threadId) => `${BASE_URL}/media/${threadId}`;

// ---------- RAG ----------
export const listRagDocuments = async () => {
  const response = await axios.get(`http://127.0.0.1:8000/rag`);
  return response.data.documents; // ["archivo.pdf", ...]
};

export const uploadRagDocument = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await axios.post(`http://127.0.0.1:8000/rag`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const deleteRagDocument = async (filename) => {
  const response = await axios.delete(`http://127.0.0.1:8000/rag`, { params: { filename } });
  return response.data;
};

const ApiService = {
  analyze,
  query,
  listConversations,
  getConversation,
  deleteConversation,
  getMediaUrl,
  listRagDocuments,
  uploadRagDocument,
  deleteRagDocument,
};

export default ApiService;