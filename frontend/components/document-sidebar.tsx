"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload, FileText, Trash2 } from "lucide-react"

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

export function DocumentSidebar({ documents, selectedDoc, onSelectDoc, onDocumentUpload }: DocumentSidebarProps) {
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
