import axios from 'axios';

// http://127.0.0.1:8000

// Subir un documento e indexarlo en la base de datos
export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await axios.post(`http://127.0.0.1:8000/upload`, formData);
  return response.data;
};

export const deleteDocument = async (fileName) => {
  const response = await axios.delete(`http://127.0.0.1:8000/delete/${fileName}`);
  return response.data;
};

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
      thread_id: threadId,
      question: question,
    });
    return response.data; // { response: "..." }
  } catch (error) {
    console.error("Error querying:", error);
    throw error;
  }
};

const ApiService = {
  uploadDocument,
  deleteDocument,
  analyze,
  query,
};

export default ApiService;