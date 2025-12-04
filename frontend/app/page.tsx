"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Send, Loader, Upload, FileText, Trash2 } from "lucide-react"

// ============================================================================
// TEACHER DOODLE COMPONENT
// ============================================================================
function TeacherDoodle() {
  return (
    <svg
      width="60"
      height="60"
      viewBox="0 0 60 60"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="flex-shrink-0"
    >
      {/* Head */}
      <circle cx="30" cy="18" r="8" fill="#f4a460" stroke="#d4860a" strokeWidth="1.5" />

      {/* Turban/Head wrap */}
      <path d="M 22 16 Q 22 10 30 10 Q 38 10 38 16" fill="#e67e22" stroke="#d4860a" strokeWidth="1.5" />
      <path d="M 22 16 Q 22 12 30 12 Q 38 12 38 16" fill="#f39c12" stroke="#d4860a" strokeWidth="1" />

      {/* Eyes */}
      <circle cx="27" cy="17" r="1.5" fill="#2c3e50" />
      <circle cx="33" cy="17" r="1.5" fill="#2c3e50" />

      {/* Smile */}
      <path d="M 27 20 Q 30 21.5 33 20" stroke="#2c3e50" strokeWidth="1.5" fill="none" strokeLinecap="round" />

      {/* Mustache */}
      <path d="M 30 19 Q 27 19.5 25 19" stroke="#8b4513" strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d="M 30 19 Q 33 19.5 35 19" stroke="#8b4513" strokeWidth="1.5" fill="none" strokeLinecap="round" />

      {/* Body - Traditional Kurta */}
      <path
        d="M 22 26 L 20 42 Q 20 44 22 44 L 38 44 Q 40 44 40 42 L 38 26 Z"
        fill="#c0392b"
        stroke="#a93226"
        strokeWidth="1.5"
      />

      {/* Kurta pattern */}
      <line x1="30" y1="26" x2="30" y2="44" stroke="#a93226" strokeWidth="1" opacity="0.5" />
      <circle cx="30" cy="32" r="1.5" fill="#f39c12" />
      <circle cx="30" cy="38" r="1.5" fill="#f39c12" />

      {/* Arms */}
      <line x1="22" y1="28" x2="12" y2="32" stroke="#f4a460" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="38" y1="28" x2="48" y2="32" stroke="#f4a460" strokeWidth="2.5" strokeLinecap="round" />

      {/* Hands */}
      <circle cx="12" cy="32" r="2" fill="#f4a460" stroke="#d4860a" strokeWidth="1" />
      <circle cx="48" cy="32" r="2" fill="#f4a460" stroke="#d4860a" strokeWidth="1" />

      {/* Legs */}
      <line x1="25" y1="44" x2="25" y2="54" stroke="#2c3e50" strokeWidth="2" strokeLinecap="round" />
      <line x1="35" y1="44" x2="35" y2="54" stroke="#2c3e50" strokeWidth="2" strokeLinecap="round" />

      {/* Feet */}
      <ellipse cx="25" cy="55" rx="2.5" ry="1.5" fill="#2c3e50" />
      <ellipse cx="35" cy="55" rx="2.5" ry="1.5" fill="#2c3e50" />

      {/* Pointer stick */}
      <line x1="48" y1="32" x2="52" y2="20" stroke="#8b4513" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="52" cy="19" r="1.5" fill="#f39c12" />
    </svg>
  )
}

// ============================================================================
// HEADER COMPONENT
// ============================================================================
function Header() {
  return (
    <header className="border-b border-border bg-card px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <TeacherDoodle />
          <div>
            <h1
              className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-orange-400"
              style={{ fontFamily: "'Playfair Display', serif" }}
            >
              <span className="text-amber-400">M</span>asterji
            </h1>
            <p className="text-sm text-muted-foreground">Your AI Tutor - Learn Smarter</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span>Ready to help</span>
        </div>
      </div>
    </header>
  )
}

// ============================================================================
// CHAT INTERFACE COMPONENT
// ============================================================================
interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: string
}

interface ChatInterfaceProps {
  selectedDocument: string | null
}

function ChatInterface({ selectedDocument }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Hello! I'm masterji, your AI tutor. Upload a document and ask me anything about it. I'll help you understand the content better!",
      timestamp: new Date().toLocaleTimeString(),
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!input.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Math.random().toString(36).substr(2, 9),
      role: "user",
      content: input,
      timestamp: new Date().toLocaleTimeString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: Math.random().toString(36).substr(2, 9),
        role: "assistant",
        content: `I understand your question about "${input}". Based on the document you've uploaded, I can help you with this. In a real implementation, I would retrieve relevant information from your document and provide a detailed answer.`,
        timestamp: new Date().toLocaleTimeString(),
      }
      setMessages((prev) => [...prev, assistantMessage])
      setIsLoading(false)
    }, 1000)
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-md lg:max-w-2xl px-4 py-3 rounded-lg ${
                message.role === "user"
                  ? "bg-primary text-primary-foreground rounded-br-none"
                  : "bg-card border border-border text-foreground rounded-bl-none"
              }`}
            >
              <p className="text-sm">{message.content}</p>
              <p className={`text-xs mt-1 ${message.role === "user" ? "opacity-70" : "text-muted-foreground"}`}>
                {message.timestamp}
              </p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-card border border-border text-foreground px-4 py-3 rounded-lg rounded-bl-none">
              <div className="flex items-center gap-2">
                <Loader className="w-4 h-4 animate-spin" />
                <span className="text-sm">masterji is thinking...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card p-4">
        {!selectedDocument && (
          <div className="mb-3 p-3 bg-muted/50 rounded-lg border border-border">
            <p className="text-sm text-muted-foreground">
              💡 Tip: Upload a document first to get started with your questions!
            </p>
          </div>
        )}
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            placeholder="Ask me anything about your document..."
            disabled={isLoading}
            className="flex-1 px-4 py-2 rounded-lg border border-input bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !input.trim()}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// DOCUMENT SIDEBAR COMPONENT
// ============================================================================
interface Document {
  id: string
  name: string
  type: string
  uploadedAt: string
}

interface DocumentSidebarProps {
  documents: Document[]
  selectedDoc: string | null
  onSelectDoc: (id: string | null) => void
  onDocumentUpload: (file: File) => void
}

function DocumentSidebar({ documents, selectedDoc, onSelectDoc, onDocumentUpload }: DocumentSidebarProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const acceptedFormats = ".pdf,.xlsx,.xls,.doc,.docx,.html"

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      onDocumentUpload(files[0])
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      onDocumentUpload(files[0])
    }
  }

  const getFileIcon = (type: string) => {
    if (type.includes("pdf")) return "📄"
    if (type.includes("sheet") || type.includes("excel")) return "📊"
    if (type.includes("word") || type.includes("document")) return "📝"
    if (type.includes("html")) return "🌐"
    return "📎"
  }

  return (
    <aside className="w-80 border-r border-border bg-sidebar flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-sidebar-border">
        <h2 className="text-lg font-semibold text-sidebar-foreground mb-4">Documents</h2>

        {/* Upload Area */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
            isDragging
              ? "border-sidebar-primary bg-sidebar-primary/10"
              : "border-sidebar-border hover:border-sidebar-primary/50"
          }`}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="w-5 h-5 mx-auto mb-2 text-sidebar-muted-foreground" />
          <p className="text-xs font-medium text-sidebar-foreground">Drop files here</p>
          <p className="text-xs text-sidebar-muted-foreground">or click to browse</p>
          <input
            ref={fileInputRef}
            type="file"
            accept={acceptedFormats}
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      </div>

      {/* Documents List */}
      <div className="flex-1 overflow-y-auto p-4">
        {documents.length === 0 ? (
          <div className="text-center py-8">
            <FileText className="w-8 h-8 mx-auto mb-2 text-sidebar-muted-foreground opacity-50" />
            <p className="text-sm text-sidebar-muted-foreground">No documents yet</p>
            <p className="text-xs text-sidebar-muted-foreground mt-1">Upload a file to get started</p>
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.id}
                onClick={() => onSelectDoc(doc.id)}
                className={`p-3 rounded-lg cursor-pointer transition-colors group ${
                  selectedDoc === doc.id
                    ? "bg-sidebar-primary text-sidebar-primary-foreground"
                    : "bg-sidebar-accent hover:bg-sidebar-accent/80 text-sidebar-foreground"
                }`}
              >
                <div className="flex items-start gap-2">
                  <span className="text-lg mt-0.5">{getFileIcon(doc.type)}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{doc.name}</p>
                    <p className="text-xs opacity-75">{doc.uploadedAt}</p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      // Handle delete
                    }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}

// ============================================================================
// MAIN PAGE COMPONENT
// ============================================================================
export default function Home() {
  const [documents, setDocuments] = useState<Array<{ id: string; name: string; type: string; uploadedAt: string }>>([])
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null)

  const handleDocumentUpload = (file: File) => {
    const newDoc = {
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      type: file.type,
      uploadedAt: new Date().toLocaleString(),
    }
    setDocuments([...documents, newDoc])
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <DocumentSidebar
        documents={documents}
        selectedDoc={selectedDoc}
        onSelectDoc={setSelectedDoc}
        onDocumentUpload={handleDocumentUpload}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <Header />
        <ChatInterface selectedDocument={selectedDoc} />
      </div>
    </div>
  )
}
