import axios from 'axios';
const apiUrl = import.meta.env.VITE_BASE_URL;

export const analyze = async (file, threadId) => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    if (threadId) {
      formData.append("thread_id", threadId);
    }
    const response = await axios.post(
      `${apiUrl}/analyze`,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data; // { thread_id, media_id, analysis, is_new_thread , annotated_media }
  } catch (error) {
    console.error("Error analyzing file:", error);
    throw error;
  }
};

export const query = async (threadId, question) => {
  try {
    const response = await axios.post(`${apiUrl}/query`, {
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
  try {
    const response = await axios.get(`${apiUrl}/conversations`);
    return response.data.conversations; // [{ thread_id, title, media_type, created_at }]
  } catch (error) {
    console.error("Error listing conversations:", error);
    throw error;
  }
};

export const getConversation = async (threadId) => {
  try {
    const response = await axios.get(`${apiUrl}/conversation/${threadId}`);
    return response.data; // { thread_id, items: [{ type: "media", media_id, media_type } | { type: "message", role, text }, ...] }
  } catch (error) {
    console.error("Error getting conversation:", error);
    throw error;
  }
};

export const deleteConversation = async (threadId) => {
  try {
    const response = await axios.delete(`${apiUrl}/conversation/${threadId}`);
    return response.data;
  } catch (error) {
    console.error("Error deleting conversation:", error);
    throw error;
  }
};

export const getMediaUrl = (threadId, mediaId) => `${apiUrl}/media/${threadId}/${mediaId}`;

// ---------- RAG ----------
export const listRagDocuments = async () => {
  try {
    const response = await axios.get(`${apiUrl}/rag`);
    return response.data.documents;
  } catch (error) {
    console.error("Error listing RAG documents:", error);
    throw error;
  }
};

export const uploadRagDocument = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    const response = await axios.post(`${apiUrl}/rag`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  } catch (error) {
    console.error("Error uploading RAG document:", error);
    throw error;
  }
};

export const deleteRagDocument = async (filename) => {
  try {
    const response = await axios.delete(`${apiUrl}/rag`, { params: { filename } });
    return response.data;
  } catch (error) {
    console.error("Error deleting RAG document:", error);
    throw error;
  }
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