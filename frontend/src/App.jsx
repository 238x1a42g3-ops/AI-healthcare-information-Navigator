import { useEffect, useRef, useState } from "react";

const initialMessages = [
  {
    role: "assistant",
    text: "Hello! I'm your healthcare document navigator. Upload a report and I'll help you understand what's in it.",
    welcome: true,
  },
];

async function readApiResponse(response, fallbackMessage) {
  const body = await response.text();
  let result = {};

  if (body.trim()) {
    try {
      result = JSON.parse(body);
    } catch {
      if (response.ok) {
        throw new Error("The server returned an invalid response. Check the backend terminal.");
      }
    }
  }

  if (!response.ok) {
    throw new Error(
      result.detail ||
        (body.trim()
          ? `Server error (${response.status}). Check the backend terminal.`
          : `The server returned an empty response (${response.status}). Check the backend terminal.`),
    );
  }

  if (!body.trim()) {
    throw new Error(fallbackMessage);
  }

  return result;
}

function Icon({ name, size = 20 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
  };

  const paths = {
    plus: <><path d="M12 5v14M5 12h14" /></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M8 13h8M8 17h6" /></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></>,
    shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" /><path d="m9 12 2 2 4-4" /></>,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 14 1.2 2.8L23 18l-2.8 1.2L19 22l-1.2-2.8L15 18l2.8-1.2L19 14Z" /></>,
    upload: <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m17 8-5-5-5 5M12 3v12" /></>,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

function App() {
  const [document, setDocument] = useState(null);
  const [messages, setMessages] = useState(initialMessages);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);
  const messageEnd = useRef(null);

  useEffect(() => {
    fetch("/api/health")
      .then(async (response) => {
        if (!response.ok) return;
        const status = await response.json();
        if (status.document_loaded && status.document) {
          setDocument({ name: status.document });
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    messageEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, thinking]);

  async function uploadPdf(file) {
    if (!file) return;
    setError("");
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/documents", {
        method: "POST",
        body: form,
      });
      const result = await readApiResponse(
        response,
        "The upload service returned an empty response. Check that the backend is running.",
      );
      setDocument({ name: result.document, chunks: result.chunks });
      setMessages(initialMessages);
    } catch (uploadError) {
      setError(uploadError.message || "Could not upload this PDF.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function sendQuestion(text = question) {
    const trimmed = text.trim();
    if (!trimmed || !document || thinking) return;
    setError("");
    setQuestion("");
    setMessages((current) => [...current, { role: "user", text: trimmed }]);
    setThinking(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      const result = await readApiResponse(
        response,
        "The chat service returned an empty response. Check that the backend is running.",
      );
      setMessages((current) => [
        ...current,
        { role: "assistant", text: result.answer, sources: result.sources || [] },
      ]);
    } catch (chatError) {
      setError(chatError.message || "The assistant could not answer right now.");
    } finally {
      setThinking(false);
    }
  }

  const suggestions = [
    "Can you summarize this report?",
    "What should I discuss with my doctor?",
    "Explain the key findings simply",
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="CareNotes home">
          <span className="brand-mark"><Icon name="plus" size={21} /></span>
          <span>care<span className="brand-light">notes</span></span>
        </a>

        <button className="new-chat" onClick={() => setMessages(initialMessages)}>
          <Icon name="plus" size={18} /> New conversation
        </button>

        <div className="sidebar-section">
          <p className="section-label">YOUR DOCUMENT</p>
          <button
            className={`document-card ${document ? "is-ready" : ""}`}
            onClick={() => fileInput.current?.click()}
            disabled={uploading}
          >
            <span className="document-icon"><Icon name="file" size={19} /></span>
            <span className="document-copy">
              <strong>{uploading ? "Reading your PDF…" : document?.name || "No document yet"}</strong>
              <small>{document ? `${document.chunks || "Indexed"} chunks · Ready` : "Upload a report to begin"}</small>
            </span>
            {document && <span className="ready-dot" />}
          </button>
        </div>

        <div className="sidebar-upload">
          <div className="upload-illustration"><Icon name="upload" size={21} /></div>
          <h3>Bring your report</h3>
          <p>Ask questions about lab results, visit summaries, and more.</p>
          <button
            className="outline-button"
            onClick={() => fileInput.current?.click()}
            disabled={uploading}
          >
            {uploading ? "Uploading…" : "Choose a PDF"}
          </button>
          <small>PDF files · Up to 25 MB</small>
        </div>

        <div className="sidebar-bottom">
          <span className="privacy-icon"><Icon name="shield" size={17} /></span>
          <p><strong>Know where your data goes.</strong><br />Your index is stored locally; questions and relevant passages are sent to Groq for answers.</p>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="mobile-brand">
            <span className="brand-mark"><Icon name="plus" size={18} /></span>
            <span>care<span className="brand-light">notes</span></span>
          </div>
          <div className="topbar-status">
            <span className={`status-dot ${document ? "online" : ""}`} />
            {document ? "Document ready" : "Ready when you are"}
          </div>
          <div className="avatar" aria-label="Healthcare navigator">C</div>
        </header>

        <section className="chat-area" aria-label="Conversation">
          <div className="conversation">
            {!document && messages.length === 1 ? (
              <div className="welcome">
                <div className="welcome-icon"><Icon name="spark" size={25} /></div>
                <p className="eyebrow">YOUR PERSONAL DOCUMENT GUIDE</p>
                <h1>Understand your<br /><span>healthcare documents.</span></h1>
                <p className="welcome-copy">
                  Upload a PDF report and ask questions in plain language. Get clear explanations grounded in your document.
                </p>
                <button
                  className="primary-button welcome-upload"
                  onClick={() => fileInput.current?.click()}
                  disabled={uploading}
                >
                  <Icon name="upload" size={18} /> {uploading ? "Uploading…" : "Upload a PDF"}
                </button>
                <div className="trust-note"><Icon name="shield" size={16} /> Local index · Questions and relevant passages sent to Groq</div>
              </div>
            ) : (
              <div className="message-list">
                <div className="chat-date"><span>YOUR CONVERSATION</span></div>
                {messages.map((message, index) => (
                  <article className={`message-row ${message.role}`} key={`${index}-${message.role}`}>
                    {message.role === "assistant" && <div className="assistant-avatar"><Icon name="spark" size={16} /></div>}
                    <div className="message-content">
                      {message.role === "assistant" && <span className="sender-label">CARENOTES <span>· AI ASSISTANT</span></span>}
                      <div className={`message-bubble ${message.role}`}>
                        {message.text}
                        {message.sources?.length > 0 && (
                          <details className="source-details">
                            <summary>Based on {message.sources.length} document {message.sources.length === 1 ? "passage" : "passages"}</summary>
                            <div className="source-list">
                              {message.sources.map((source, sourceIndex) => (
                                <blockquote key={sourceIndex}>{source}</blockquote>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    </div>
                  </article>
                ))}
                {thinking && (
                  <article className="message-row assistant">
                    <div className="assistant-avatar"><Icon name="spark" size={16} /></div>
                    <div className="message-content">
                      <span className="sender-label">CARENOTES <span>· AI ASSISTANT</span></span>
                      <div className="message-bubble assistant typing"><i /><i /><i /></div>
                    </div>
                  </article>
                )}
                <div ref={messageEnd} />
              </div>
            )}
          </div>
        </section>

        <div className="composer-wrap">
          {error && <div className="error-banner" role="alert">{error}</div>}
          {document && messages.length <= 1 && (
            <div className="suggestions">
              <p>NOT SURE WHERE TO START?</p>
              <div className="suggestion-list">
                {suggestions.map((suggestion) => (
                  <button key={suggestion} onClick={() => sendQuestion(suggestion)} disabled={thinking}>
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
          <form className="composer" onSubmit={(event) => { event.preventDefault(); sendQuestion(); }}>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  sendQuestion();
                }
              }}
              placeholder={document ? "Ask a question about your report…" : "Upload a PDF to start a conversation…"}
              aria-label="Ask a question about your report"
              rows="1"
              disabled={!document || thinking}
            />
            <button className="send-button" type="submit" disabled={!document || !question.trim() || thinking} aria-label="Send question">
              <Icon name="send" size={18} />
            </button>
          </form>
          <p className="disclaimer">For educational purposes only. Always consult a qualified healthcare professional for medical decisions.</p>
        </div>
      </main>

      <input
        ref={fileInput}
        className="visually-hidden"
        type="file"
        accept="application/pdf,.pdf"
        onChange={(event) => uploadPdf(event.target.files?.[0])}
      />
    </div>
  );
}

export default App;
