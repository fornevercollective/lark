import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Separator } from './ui/separator';
import { ScrollArea } from './ui/scroll-area';
import { 
  Bold, 
  Italic, 
  Code, 
  List, 
  ListOrdered, 
  Quote, 
  Heading1, 
  Heading2, 
  Heading3,
  Link,
  Image,
  Table,
  Save,
  Download,
  Upload,
  Eye,
  EyeOff,
  Split,
  FileText,
  Copy
} from 'lucide-react';

interface MarkdownFile {
  id: string;
  name: string;
  content: string;
  createdAt: Date;
  updatedAt: Date;
}

type ViewMode = 'split' | 'edit' | 'preview';

export function MarkdownEditor() {
  const [content, setContent] = useState(`# Welcome to Custom Markdown Editor

## Features
- **Live Preview** - See your markdown rendered in real-time
- **Syntax Highlighting** - Code blocks with proper highlighting
- **File Management** - Save, load, and export your documents
- **Rich Toolbar** - Quick formatting tools
- **Export Options** - HTML, PDF, and more

## Code Example

\`\`\`python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
\`\`\`

## Table Example

| Framework | Performance | Ease of Use | Community |
|-----------|-------------|-------------|-----------|
| PyTorch   | High        | Good        | Large     |
| TinyGrad  | Medium      | Excellent   | Growing   |
| JAX       | Very High   | Medium      | Medium    |

## Todo List
- [x] Create markdown editor
- [x] Add live preview
- [ ] Implement export functionality
- [ ] Add collaborative features

---

*Happy coding!* 🚀`);
  
  const [files, setFiles] = useState<MarkdownFile[]>([]);
  const [currentFile, setCurrentFile] = useState<MarkdownFile | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('split');
  const [fileName, setFileName] = useState('');
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);

  // Enhanced markdown rendering with more features
  const renderMarkdown = (markdown: string) => {
    let rendered = markdown
      // Headers
      .replace(/^# (.*$)/gm, '<h1 class="text-3xl font-medium mb-4 mt-6 text-foreground border-b border-border pb-2">$1</h1>')
      .replace(/^## (.*$)/gm, '<h2 class="text-2xl font-medium mb-3 mt-5 text-foreground">$1</h2>')
      .replace(/^### (.*$)/gm, '<h3 class="text-xl font-medium mb-2 mt-4 text-foreground">$1</h3>')
      .replace(/^#### (.*$)/gm, '<h4 class="text-lg font-medium mb-2 mt-3 text-foreground">$1</h4>')
      
      // Bold and italic
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-medium">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
      
      // Code blocks with syntax highlighting
      .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const language = lang || 'text';
        return `<div class="my-4">
          <div class="bg-muted text-muted-foreground text-xs px-3 py-1 rounded-t-md border border-b-0">${language}</div>
          <pre class="bg-card border rounded-b-md p-4 overflow-x-auto"><code class="font-mono text-sm text-foreground">${code.trim()}</code></pre>
        </div>`;
      })
      
      // Inline code
      .replace(/`(.*?)`/g, '<code class="bg-muted text-muted-foreground px-1.5 py-0.5 rounded font-mono text-sm">$1</code>')
      
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-primary hover:underline">$1</a>')
      
      // Images
      .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="max-w-full h-auto rounded-md my-4" />')
      
      // Tables
      .replace(/\|(.+)\|\n\|[-\s|:]+\|\n((?:\|.+\|\n)*)/g, (match, header, rows) => {
        const headerCells = header.split('|').map(cell => cell.trim()).filter(Boolean);
        const headerRow = headerCells.map(cell => `<th class="border border-border px-3 py-2 bg-muted font-medium text-left">${cell}</th>`).join('');
        
        const bodyRows = rows.trim().split('\n').map(row => {
          const cells = row.split('|').map(cell => cell.trim()).filter(Boolean);
          const rowCells = cells.map(cell => `<td class="border border-border px-3 py-2">${cell}</td>`).join('');
          return `<tr>${rowCells}</tr>`;
        }).join('');
        
        return `<div class="my-4 overflow-x-auto">
          <table class="w-full border-collapse border border-border rounded-md">
            <thead><tr>${headerRow}</tr></thead>
            <tbody>${bodyRows}</tbody>
          </table>
        </div>`;
      })
      
      // Lists
      .replace(/^\- \[(x| )\] (.+)$/gm, (match, checked, text) => {
        const isChecked = checked === 'x';
        return `<div class="flex items-center space-x-2 my-1">
          <input type="checkbox" ${isChecked ? 'checked' : ''} disabled class="rounded" />
          <span class="${isChecked ? 'line-through text-muted-foreground' : ''}">${text}</span>
        </div>`;
      })
      .replace(/^\- (.+)$/gm, '<li class="ml-4 my-1">• $1</li>')
      .replace(/^\d+\. (.+)$/gm, '<li class="ml-4 my-1 list-decimal">$1</li>')
      
      // Blockquotes
      .replace(/^> (.+)$/gm, '<blockquote class="border-l-4 border-primary pl-4 py-2 my-4 bg-muted/50 italic">$1</blockquote>')
      
      // Horizontal rules
      .replace(/^---$/gm, '<hr class="border-border my-6" />')
      
      // Line breaks
      .replace(/\n/g, '<br>');
    
    return rendered;
  };

  // Insert text at cursor position
  const insertText = (before: string, after: string = '') => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = content.substring(start, end);
    
    const newText = content.substring(0, start) + before + selectedText + after + content.substring(end);
    setContent(newText);
    
    // Set cursor position
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + before.length, start + before.length + selectedText.length);
    }, 0);
  };

  // Toolbar actions
  const toolbarActions = {
    bold: () => insertText('**', '**'),
    italic: () => insertText('*', '*'),
    code: () => insertText('`', '`'),
    heading1: () => insertText('# '),
    heading2: () => insertText('## '),
    heading3: () => insertText('### '),
    list: () => insertText('- '),
    orderedList: () => insertText('1. '),
    quote: () => insertText('> '),
    link: () => insertText('[', '](url)'),
    image: () => insertText('![alt](', ')'),
    codeBlock: () => insertText('```\n', '\n```'),
    table: () => insertText('| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |\n')
  };

  // File management
  const saveFile = () => {
    if (!fileName.trim()) {
      alert('Please enter a filename');
      return;
    }

    const file: MarkdownFile = {
      id: currentFile?.id || Date.now().toString(),
      name: fileName,
      content,
      createdAt: currentFile?.createdAt || new Date(),
      updatedAt: new Date()
    };

    setFiles(prev => {
      const existing = prev.findIndex(f => f.id === file.id);
      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = file;
        return updated;
      }
      return [...prev, file];
    });

    setCurrentFile(file);
    alert(`File "${fileName}" saved successfully!`);
  };

  const loadFile = (file: MarkdownFile) => {
    setContent(file.content);
    setFileName(file.name);
    setCurrentFile(file);
  };

  const exportHtml = () => {
    const html = `
<!DOCTYPE html>
<html>
<head>
    <title>${fileName || 'Markdown Document'}</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.6; }
        pre { background: #f5f5f5; padding: 1rem; border-radius: 4px; overflow-x: auto; }
        code { background: #f5f5f5; padding: 0.2rem 0.4rem; border-radius: 3px; }
        blockquote { border-left: 4px solid #ccc; margin: 0; padding-left: 1rem; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
        th { background: #f5f5f5; }
    </style>
</head>
<body>
${renderMarkdown(content)}
</body>
</html>`;
    
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileName || 'document'}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(content);
    alert('Content copied to clipboard!');
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 border-b border-border bg-card">
        <div className="flex items-center space-x-1">
          <Button variant="outline" size="sm" onClick={toolbarActions.bold}>
            <Bold className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toolbarActions.italic}>
            <Italic className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toolbarActions.code}>
            <Code className="h-4 w-4" />
          </Button>
          
          <Separator orientation="vertical" className="h-6" />
          
          <Button variant="outline" size="sm" onClick={toolbarActions.heading1}>
            <Heading1 className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toolbarActions.heading2}>
            <Heading2 className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toolbarActions.heading3}>
            <Heading3 className="h-4 w-4" />
          </Button>
          
          <Separator orientation="vertical" className="h-6" />
          
          <Button variant="outline" size="sm" onClick={toolbarActions.list}>
            <List className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toolbarActions.orderedList}>
            <ListOrdered className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toolbarActions.quote}>
            <Quote className="h-4 w-4" />
          </Button>
          
          <Separator orientation="vertical" className="h-6" />
          
          <Button variant="outline" size="sm" onClick={toolbarActions.link}>
            <Link className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toolbarActions.image}>
            <Image className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={toolbarActions.table}>
            <Table className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 border rounded-md">
            <Button 
              variant={viewMode === 'edit' ? 'default' : 'ghost'} 
              size="sm"
              onClick={() => setViewMode('edit')}
            >
              <FileText className="h-4 w-4" />
            </Button>
            <Button 
              variant={viewMode === 'split' ? 'default' : 'ghost'} 
              size="sm"
              onClick={() => setViewMode('split')}
            >
              <Split className="h-4 w-4" />
            </Button>
            <Button 
              variant={viewMode === 'preview' ? 'default' : 'ghost'} 
              size="sm"
              onClick={() => setViewMode('preview')}
            >
              <Eye className="h-4 w-4" />
            </Button>
          </div>
          
          <Separator orientation="vertical" className="h-6" />
          
          <Button variant="outline" size="sm" onClick={copyToClipboard}>
            <Copy className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={exportHtml}>
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* File Management */}
      <div className="flex items-center space-x-2 p-4 bg-muted/30 border-b border-border">
        <input
          type="text"
          value={fileName}
          onChange={(e) => setFileName(e.target.value)}
          placeholder="Enter filename..."
          className="px-3 py-1 border border-border rounded-md bg-background text-sm"
        />
        <Button size="sm" onClick={saveFile}>
          <Save className="h-4 w-4 mr-1" />
          Save
        </Button>
        
        {files.length > 0 && (
          <select
            onChange={(e) => {
              const file = files.find(f => f.id === e.target.value);
              if (file) loadFile(file);
            }}
            className="px-3 py-1 border border-border rounded-md bg-background text-sm"
          >
            <option value="">Load file...</option>
            {files.map(file => (
              <option key={file.id} value={file.id}>
                {file.name} ({file.updatedAt.toLocaleDateString()})
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Editor/Preview */}
      <div className="flex-1 flex overflow-hidden">
        {/* Editor */}
        {(viewMode === 'edit' || viewMode === 'split') && (
          <div className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} flex flex-col`}>
            <div className="p-2 bg-muted/30 border-b border-border text-sm text-muted-foreground">
              Editor
            </div>
            <textarea
              ref={textareaRef}
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="flex-1 p-4 bg-background border-0 resize-none focus:outline-none font-mono text-sm"
              placeholder="Start writing your markdown..."
            />
          </div>
        )}

        {/* Divider */}
        {viewMode === 'split' && (
          <div className="w-px bg-border" />
        )}

        {/* Preview */}
        {(viewMode === 'preview' || viewMode === 'split') && (
          <div className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} flex flex-col`}>
            <div className="p-2 bg-muted/30 border-b border-border text-sm text-muted-foreground">
              Preview
            </div>
            <ScrollArea className="flex-1">
              <div 
                ref={previewRef}
                className="p-4 prose prose-neutral max-w-none"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
              />
            </ScrollArea>
          </div>
        )}
      </div>
    </div>
  );
}