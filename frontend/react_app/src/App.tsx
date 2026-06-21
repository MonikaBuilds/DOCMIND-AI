import {
  AlertCircle,
  Bot,
  Check,
  FileText,
  Loader2,
  Menu,
  MessageSquareText,
  PanelRightOpen,
  Search,
  Send,
  Sparkles,
  Upload,
  User,
  X
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  ChatResponse,
  DocumentRecord,
  SearchResult,
  chat,
  listDocuments,
  processDocument,
  searchDocuments,
  uploadPdfs
} from "./api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  citations?: string[];
};

const sessionId = crypto.randomUUID();

export default function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Upload a document and ask anything about it. I will answer with clear references from your file."
    }
  ]);
  const [question, setQuestion] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isLibraryOpen, setIsLibraryOpen] = useState(false);
  const [isContextOpen, setIsContextOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const selectedDocument = useMemo(
    () => documents.find((document) => document.document_id === activeDocumentId) ?? documents[0],
    [activeDocumentId, documents]
  );

  const metrics = useMemo(
    () => ({
      documents: documents.length,
      processed: documents.filter((document) => document.status === "processed").length,
      pages: documents.reduce((sum, document) => sum + (document.page_count ?? 0), 0),
      chunks: documents.reduce((sum, document) => sum + document.chunk_count, 0)
    }),
    [documents]
  );

  useEffect(() => {
    refreshDocuments();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function refreshDocuments() {
    try {
      setDocuments(await listDocuments());
    } catch {
      setDocuments([]);
    }
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const files = event.target.files;
    if (!files?.length) return;
    setError(null);
    setIsUploading(true);
    try {
      const uploaded = await uploadPdfs(files);
      await refreshDocuments();
      setActiveDocumentId(uploaded[0]?.document_id ?? null);
      setIsLibraryOpen(false);
      setIsProcessing(true);
      for (const document of uploaded) {
        await processDocument(document.document_id);
      }
      await refreshDocuments();
    } catch (uploadError) {
      setError(getErrorMessage(uploadError));
    } finally {
      setIsUploading(false);
      setIsProcessing(false);
      event.target.value = "";
    }
  }

  async function handleProcess(documentId: string) {
    setError(null);
    setIsProcessing(true);
    try {
      await processDocument(documentId);
      await refreshDocuments();
      setActiveDocumentId(documentId);
      setIsContextOpen(false);
    } catch (processError) {
      setError(getErrorMessage(processError));
    } finally {
      setIsProcessing(false);
    }
  }

  async function handleChatSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion) return;
    setQuestion("");
    setError(null);
    setIsChatting(true);
    setMessages((current) => [...current, { role: "user", content: cleanQuestion }]);
    try {
      const isReady = await ensureDocumentReady();
      if (!isReady) return;
      const response: ChatResponse = await chat(cleanQuestion, sessionId);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: response.answer,
          citations: response.citations.map((citation) => citation.label)
        }
      ]);
    } catch (chatError) {
      setError(getErrorMessage(chatError));
    } finally {
      setIsChatting(false);
    }
  }

  async function ensureDocumentReady() {
    if (!documents.length) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: "Please upload a document first. Once it is ready, I can answer questions from it."
        }
      ]);
      return false;
    }

    if (documents.some((document) => document.status === "processed" && document.chunk_count > 0)) {
      return true;
    }

    const documentToPrepare = selectedDocument ?? documents[0];
    setIsProcessing(true);
    setMessages((current) => [
      ...current,
      {
        role: "assistant",
        content: "I need to prepare this document before answering. Starting that now."
      }
    ]);

    try {
      await processDocument(documentToPrepare.document_id);
      await refreshDocuments();
      return true;
    } catch (prepareError) {
      setError(getErrorMessage(prepareError));
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            "I could not prepare this file yet. Please check the app setup and try again."
        }
      ]);
      return false;
    } finally {
      setIsProcessing(false);
    }
  }

  async function handleSearchSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuery = searchQuery.trim();
    if (!cleanQuery) return;
    setError(null);
    setIsSearching(true);
    try {
      setSearchResults(await searchDocuments(cleanQuery));
    } catch (searchError) {
      setError(getErrorMessage(searchError));
    } finally {
      setIsSearching(false);
    }
  }

  function closeDrawers() {
    setIsLibraryOpen(false);
    setIsContextOpen(false);
  }

  return (
    <div className="app-shell">
      <div className={`mobile-overlay ${isLibraryOpen || isContextOpen ? "visible" : ""}`} onClick={closeDrawers} />

      <aside className={`left-rail ${isLibraryOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">DM</div>
          <div>
            <h1>DocMind AI</h1>
            <p>Document assistant</p>
          </div>
          <button className="icon-button close-drawer" onClick={() => setIsLibraryOpen(false)} aria-label="Close documents">
            <X size={18} />
          </button>
        </div>

        <label className="upload-zone">
          <input type="file" accept="application/pdf" multiple onChange={handleUpload} />
          <Upload size={18} />
          <span>{isUploading || isProcessing ? "Preparing..." : "Upload PDFs"}</span>
        </label>

        <div className="section-title">Documents</div>
        <div className="document-list">
          {documents.map((document) => (
            <button
              key={document.document_id}
              className={`document-item ${selectedDocument?.document_id === document.document_id ? "active" : ""}`}
              onClick={() => {
                setActiveDocumentId(document.document_id);
                setIsLibraryOpen(false);
              }}
            >
              <FileText size={18} />
              <span>
                <strong>{document.filename}</strong>
                <small>{getDocumentStatusLabel(document)}</small>
              </span>
            </button>
          ))}
          {!documents.length && (
            <div className="empty-list">
              <FileText size={18} />
              <span>No PDFs yet</span>
            </div>
          )}
        </div>
      </aside>

      <main className="chat-column">
        <header className="topbar">
          <button className="icon-button mobile-only" onClick={() => setIsLibraryOpen(true)} aria-label="Open documents">
            <Menu size={19} />
          </button>
          <div className="topbar-copy">
            <h2>DocMind AI</h2>
            <p>Ask questions, get summaries, and find answers across your files</p>
          </div>
          <div className="topbar-actions">
            <div className="status-pill">
              <Sparkles size={16} />
              Ready
            </div>
            <button className="icon-button mobile-context" onClick={() => setIsContextOpen(true)} aria-label="Open sources panel">
              <PanelRightOpen size={19} />
            </button>
          </div>
        </header>

        <div className="workspace">
          {error && (
            <div className="error-banner">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          {!!documents.length && metrics.processed === 0 && (
            <div className="setup-banner">
              <div>
                <strong>Prepare your file to start asking questions</strong>
                <span>Your document is uploaded. Preparing it makes answers and page references available.</span>
              </div>
              <button
                type="button"
                onClick={() => selectedDocument && handleProcess(selectedDocument.document_id)}
                disabled={isProcessing || !selectedDocument}
              >
                {isProcessing ? <Loader2 size={16} className="spin" /> : <Check size={16} />}
                Prepare file
              </button>
            </div>
          )}

          {!!documents.length && (
            <section className="metrics-grid" aria-label="Document metrics">
              <Metric label="Files" value={metrics.documents} />
              <Metric label="Ready" value={metrics.processed} />
              <Metric label="Pages" value={metrics.pages} />
              <Metric label="Sections" value={metrics.chunks} />
            </section>
          )}

          {!documents.length && (
            <section className="welcome-panel">
              <div className="welcome-mark">
                <Sparkles size={28} />
              </div>
              <h1>Your documents, instantly understood</h1>
              <p>
                Upload PDFs, research papers, notes, or reports. Ask questions, compare topics, and get reliable answers with page references.
              </p>
              <div className="welcome-actions">
                <label className="hero-upload">
                  <input type="file" accept="application/pdf" multiple onChange={handleUpload} />
                  <Upload size={18} />
                  {isUploading ? "Uploading..." : "Upload files"}
                </label>
                <button className="secondary-action" onClick={() => setIsContextOpen(true)}>
                  <PanelRightOpen size={18} />
                  View sources
                </button>
              </div>
            </section>
          )}

          <section className="messages" aria-live="polite">
            {messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                <div className="avatar">{message.role === "assistant" ? <Bot size={18} /> : <User size={18} />}</div>
                <div className="bubble">
                  <p>{message.content}</p>
                  {!!message.citations?.length && (
                    <div className="citation-row">
                      {message.citations.map((citation) => (
                        <span key={citation}>{citation}</span>
                      ))}
                    </div>
                  )}
                </div>
              </article>
            ))}
            {isChatting && (
              <article className="message assistant">
                <div className="avatar">
                  <Bot size={18} />
                </div>
                <div className="bubble pending">
                  <Loader2 size={18} className="spin" />
                  Reading your document
                </div>
              </article>
            )}
            <div ref={chatEndRef} />
          </section>

          <form className="composer" onSubmit={handleChatSubmit}>
            <MessageSquareText size={20} />
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about your documents..."
            />
            <button type="submit" disabled={isChatting || !question.trim()} aria-label="Send message">
              <Send size={18} />
            </button>
          </form>
        </div>
      </main>

      <aside className={`right-panel ${isContextOpen ? "open" : ""}`}>
        <div className="panel-header">
          <div>
            <PanelRightOpen size={18} />
            <h3>Sources</h3>
          </div>
          <button className="icon-button close-drawer" onClick={() => setIsContextOpen(false)} aria-label="Close sources">
            <X size={18} />
          </button>
        </div>

        {selectedDocument ? (
          <div className="document-card">
            <strong>{selectedDocument.filename}</strong>
            <small>{selectedDocument.storage_path}</small>
            <div className="document-stats">
              <span>{selectedDocument.page_count ?? 0} pages</span>
              <span>{selectedDocument.chunk_count} sections</span>
            </div>
            <button
              onClick={() => handleProcess(selectedDocument.document_id)}
              disabled={isProcessing}
              className="primary-action"
            >
              {isProcessing ? <Loader2 size={16} className="spin" /> : <Check size={16} />}
              Prepare file
            </button>
          </div>
        ) : (
          <div className="empty-panel">No document selected</div>
        )}

        <form className="search-box" onSubmit={handleSearchSubmit}>
          <Search size={18} />
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search inside documents"
          />
          <button type="submit" disabled={isSearching || !searchQuery.trim()} aria-label="Search">
            {isSearching ? <Loader2 size={16} className="spin" /> : <Search size={16} />}
          </button>
        </form>

        <div className="results">
          {searchResults.map((result) => (
            <div className="result-card" key={result.chunk_id}>
              <div className="result-meta">
                <span>{result.filename}</span>
                <span>p. {result.page_number}</span>
              </div>
              <p>{result.text}</p>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function getErrorMessage(error: unknown) {
  const rawMessage = error instanceof Error ? error.message : "Something went wrong.";

  try {
    const parsed = JSON.parse(rawMessage) as { detail?: string };
    if (parsed.detail) return friendlyError(parsed.detail);
  } catch {
    return friendlyError(rawMessage);
  }

  return friendlyError(rawMessage);
}

function friendlyError(message: string) {
  if (message.toLowerCase().includes("no processed chunks")) {
    return "Your document is uploaded but not ready yet. Click Prepare file, then ask your question again.";
  }

  if (message.toLowerCase().includes("api key")) {
    return "Document preparation is not available yet. Please check the app setup and try again.";
  }

  if (message.toLowerCase().includes("chroma")) {
    return "Document search is not ready yet. Please check the app setup and try again.";
  }

  return message;
}

function getDocumentStatusLabel(document: DocumentRecord) {
  if (document.status === "processed" && document.chunk_count > 0) {
    return `Ready / ${document.chunk_count} sections`;
  }

  if (document.status === "processing") return "Preparing...";
  if (document.status === "failed") return "Needs attention";
  return "Uploaded / needs preparation";
}
