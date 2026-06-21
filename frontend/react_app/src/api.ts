const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL =
  configuredApiBaseUrl && !configuredApiBaseUrl.includes("your-backend-url")
    ? configuredApiBaseUrl
    : "/api/v1";

export type UploadedDocument = {
  document_id: string;
  original_filename: string;
  stored_filename: string;
  content_type: string;
  size_bytes: number;
  storage_path: string;
  status: string;
};

export type DocumentRecord = {
  document_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  storage_path: string;
  status: string;
  page_count: number | null;
  chunk_count: number;
};

export type SearchResult = {
  chunk_id: string;
  document_id: string;
  filename: string;
  text: string;
  score: number;
  page_number: number;
  source: string;
  heading: string | null;
};

export type ChatResponse = {
  answer: string;
  citations: Array<{ label: string }>;
  retrieved_count: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function uploadPdfs(files: FileList): Promise<UploadedDocument[]> {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  const data = await request<{ documents: UploadedDocument[] }>("/upload/pdfs", {
    method: "POST",
    body: formData
  });
  return data.documents;
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>("/documents");
}

export async function processDocument(documentId: string, chunkSize = 180, overlap = 40) {
  return request(`/documents/${documentId}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk_size: chunkSize, overlap })
  });
}

export async function searchDocuments(
  query: string,
  topK = 5,
  documentIds?: string[]
): Promise<SearchResult[]> {
  const data = await request<{ results: SearchResult[] }>("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK, document_ids: documentIds })
  });
  return data.results;
}

export async function chat(
  question: string,
  sessionId: string,
  topK = 5,
  documentIds?: string[]
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId, top_k: topK, document_ids: documentIds })
  });
}
