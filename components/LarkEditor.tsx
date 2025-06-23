import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from './ui/button';
import { Separator } from './ui/separator';
import { ScrollArea } from './ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Label } from './ui/label';
import { LarkAssetManager, LarkAsset } from './LarkAssetManager';

type LarkMode = 'editor' | 'terminal' | 'ai' | 'files' | 'templates' | 'settings';

interface LarkFile {
  id: string;
  name: string;
  content: string;
  type: 'markdown' | 'python' | 'text';
  createdAt: Date;
  updatedAt: Date;
}

interface TerminalLine {
  id: number;
  content: string;
  type: 'command' | 'output' | 'success' | 'error' | 'python' | 'ai';
}

interface AIConversation {
  id: string;
  title: string;
  messages: Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
  }>;
  createdAt: Date;
}

interface LarkTemplate {
  id: string;
  name: string;
  description: string;
  content: string;
  category: 'ai' | 'python' | 'docs' | 'project';
}

interface TextSelection {
  start: number;
  end: number;
  text: string;
  element?: HTMLElement;
}

interface AudioSettings {
  waveformEnabled: boolean;
  sensitivity: number;
  animationSpeed: number;
  phraseRotationEnabled: boolean;
  phraseRotationSpeed: number;
  autoActivityDetection: boolean;
  visualFeedbackEnabled: boolean;
  waveformStyle: 'minimal' | 'standard' | 'detailed';
  waveformColor: 'default' | 'accent' | 'primary';
}

type ViewMode = 'split' | 'edit' | 'preview';

interface LarkEditorProps {
  mode: LarkMode;
  audioSettings?: AudioSettings;
  onAudioSettingsChange?: (settings: Partial<AudioSettings>) => void;
}

export function LarkEditor({ 
  mode, 
  audioSettings = {
    waveformEnabled: true,
    sensitivity: 0.8,
    animationSpeed: 1.0,
    phraseRotationEnabled: true,
    phraseRotationSpeed: 3,
    autoActivityDetection: true,
    visualFeedbackEnabled: true,
    waveformStyle: 'minimal',
    waveformColor: 'default'
  },
  onAudioSettingsChange
}: LarkEditorProps) {
  // Core state
  const [content, setContent] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('split');
  const [files, setFiles] = useState<LarkFile[]>([]);
  const [currentFile, setCurrentFile] = useState<LarkFile | null>(null);
  const [fileName, setFileName] = useState('');
  const [assetPanelOpen, setAssetPanelOpen] = useState(false);

  // Selection state
  const [selectedText, setSelectedText] = useState<TextSelection | null>(null);
  const [showSelectionTools, setShowSelectionTools] = useState(false);
  const [selectionToolsPosition, setSelectionToolsPosition] = useState({ x: 0, y: 0 });

  // Assets state
  const [assets, setAssets] = useState<LarkAsset[]>([]);

  // Terminal state
  const [terminalLines, setTerminalLines] = useState<TerminalLine[]>([
    { id: 0, content: '.\ lark dev environment ready', type: 'output' }
  ]);
  const [terminalInput, setTerminalInput] = useState('');
  const [multiLineInput, setMultiLineInput] = useState('');
  const [isMultiLine, setIsMultiLine] = useState(false);

  // AI state
  const [conversations, setConversations] = useState<AIConversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<AIConversation | null>(null);
  const [aiInput, setAiInput] = useState('');
  const [isAiThinking, setIsAiThinking] = useState(false);

  // Templates
  const [templates] = useState<LarkTemplate[]>([
    {
      id: '1',
      name: 'pytorch.model',
      description: 'basic neural network template',
      category: 'python',
      content: `# PyTorch Model Template

\`\`\`python
import torch
import torch.nn as nn
import torch.optim as optim

class LarkModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LarkModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model
model = LarkModel(input_size=784, hidden_size=128, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
\`\`\`
`
    },
    {
      id: '2',
      name: 'research.notes',
      description: 'ai research documentation',
      category: 'docs',
      content: `# research notes

## experiment overview
- date: ${new Date().toISOString().split('T')[0]}
- objective: 
- hypothesis: 

## dataset
- source: 
- size: 
- features: 

## model architecture
- type: 
- parameters: 
- loss function: 

## results
| metric | value |
|--------|-------|
| accuracy | % |
| loss | |
| f1 score | |

## next steps
- [ ] 
- [ ] 
- [ ] 

## code
\`\`\`python
# experiment code
\`\`\`
`
    },
    {
      id: '3',
      name: 'tinygrad.setup',
      description: 'minimal framework quickstart',
      category: 'python',
      content: `# tinygrad quickstart

\`\`\`python
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import numpy as np

# create tensors
x = Tensor.randn(32, 784)  # batch of 32 samples
y = Tensor.randn(32, 10)   # 10 classes

# simple linear layer
linear = Linear(784, 10)
output = linear(x)

# compute loss
loss = ((output - y) ** 2).mean()
print(f"loss: {loss.numpy()}")

# backward pass
loss.backward()
print("gradients computed")
\`\`\`

## key features
- automatic differentiation
- minimal dependencies
- gpu support
- pytorch-like api
`
    },
    {
      id: '4',
      name: 'ai.prompt',
      description: 'structured conversation template',
      category: 'ai',
      content: `# ai conversation template

## context
you are an ai assistant helping with [specific task]

## task
[describe the specific task or question]

## requirements
- provide code examples when relevant
- explain complex concepts clearly
- include best practices
- suggest next steps

## example input/output
input: [example question]
output: [expected response format]

---

*use this template for better ai conversations*
`
    }
  ]);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lineNumbersRef = useRef<HTMLDivElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<HTMLDivElement>(null);
  const terminalInputRef = useRef<HTMLInputElement>(null);

  // Scroll synchronization
  const isScrollingFromTextarea = useRef(false);
  const isScrollingFromLineNumbers = useRef(false);

  // Sync scroll between textarea and line numbers
  useEffect(() => {
    const textarea = textareaRef.current;
    const lineNumbers = lineNumbersRef.current;

    if (!textarea || !lineNumbers) return;

    const handleTextareaScroll = () => {
      if (isScrollingFromLineNumbers.current) return;
      
      isScrollingFromTextarea.current = true;
      lineNumbers.scrollTop = textarea.scrollTop;
      
      // Reset flag after a brief delay
      setTimeout(() => {
        isScrollingFromTextarea.current = false;
      }, 10);
    };

    const handleLineNumbersScroll = () => {
      if (isScrollingFromTextarea.current) return;
      
      isScrollingFromLineNumbers.current = true;
      textarea.scrollTop = lineNumbers.scrollTop;
      
      // Reset flag after a brief delay
      setTimeout(() => {
        isScrollingFromLineNumbers.current = false;
      }, 10);
    };

    textarea.addEventListener('scroll', handleTextareaScroll);
    lineNumbers.addEventListener('scroll', handleLineNumbersScroll);

    return () => {
      textarea.removeEventListener('scroll', handleTextareaScroll);
      lineNumbers.removeEventListener('scroll', handleLineNumbersScroll);
    };
  }, [viewMode]); // Re-run when view mode changes

  // Load content on mount
  useEffect(() => {
    const savedContent = localStorage.getItem('lark-current-content');
    if (savedContent) {
      setContent(savedContent);
    }

    const savedFiles = localStorage.getItem('lark-files');
    if (savedFiles) {
      setFiles(JSON.parse(savedFiles));
    }

    const savedConversations = localStorage.getItem('lark-conversations');
    if (savedConversations) {
      setConversations(JSON.parse(savedConversations));
    }

    const savedAssets = localStorage.getItem('lark-assets');
    if (savedAssets) {
      setAssets(JSON.parse(savedAssets));
    }
  }, []);

  // Save content on change
  useEffect(() => {
    if (content) {
      localStorage.setItem('lark-current-content', content);
    }
  }, [content]);

  // Save assets on change
  useEffect(() => {
    localStorage.setItem('lark-assets', JSON.stringify(assets));
  }, [assets]);

  // Text selection handling
  const handleTextSelection = useCallback((e: React.MouseEvent) => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      const text = selection.toString();
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      
      setSelectedText({
        start: range.startOffset,
        end: range.endOffset,
        text: text,
        element: range.commonAncestorContainer.parentElement || undefined
      });
      
      setSelectionToolsPosition({
        x: rect.left + rect.width / 2,
        y: rect.top - 40
      });
      
      setShowSelectionTools(true);
    } else {
      setShowSelectionTools(false);
      setSelectedText(null);
    }
  }, []);

  // Hide selection tools when clicking elsewhere
  useEffect(() => {
    const handleClickOutside = () => {
      if (!window.getSelection()?.toString()) {
        setShowSelectionTools(false);
        setSelectedText(null);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Enhanced markdown rendering with line numbers
  const renderMarkdown = (markdown: string) => {
    const lines = markdown.split('\n');
    let rendered = markdown
      .replace(/^# (.*$)/gm, '<h1 class="text-2xl font-mono mb-4 mt-6 text-foreground border-b border-border pb-2">$1</h1>')
      .replace(/^## (.*$)/gm, '<h2 class="text-xl font-mono mb-3 mt-5 text-foreground">$1</h2>')
      .replace(/^### (.*$)/gm, '<h3 class="text-lg font-mono mb-2 mt-4 text-foreground">$1</h3>')
      .replace(/^#### (.*$)/gm, '<h4 class="text-base font-mono mb-2 mt-3 text-foreground">$1</h4>')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-medium">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
      .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const language = lang || 'text';
        return `<div class="my-4">
          <div class="bg-muted text-muted-foreground text-xs font-mono px-3 py-1 rounded-t-md border border-b-0 flex items-center justify-between">
            <span>${language}</span>
            <button onclick="navigator.clipboard.writeText(\`${code.trim().replace(/`/g, '\\`')}\`)" class="hover:bg-muted-foreground/10 px-2 py-1 rounded text-xs">copy</button>
          </div>
          <pre class="bg-card border rounded-b-md p-4 overflow-x-auto"><code class="font-mono text-sm text-foreground">${code.trim()}</code></pre>
        </div>`;
      })
      .replace(/`(.*?)`/g, '<code class="bg-muted text-muted-foreground px-1.5 py-0.5 rounded font-mono text-sm">$1</code>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-foreground hover:underline">$1</a>')
      .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" class="max-w-full h-auto rounded-md my-4" />')
      .replace(/\|(.+)\|\n\|[-\s|:]+\|\n((?:\|.+\|\n)*)/g, (match, header, rows) => {
        const headerCells = header.split('|').map(cell => cell.trim()).filter(Boolean);
        const headerRow = headerCells.map(cell => `<th class="border border-border px-3 py-2 bg-muted font-mono text-left text-sm">${cell}</th>`).join('');
        const bodyRows = rows.trim().split('\n').map(row => {
          const cells = row.split('|').map(cell => cell.trim()).filter(Boolean);
          const rowCells = cells.map(cell => `<td class="border border-border px-3 py-2 font-mono text-sm">${cell}</td>`).join('');
          return `<tr>${rowCells}</tr>`;
        }).join('');
        return `<div class="my-4 overflow-x-auto">
          <table class="w-full border-collapse border border-border rounded-md">
            <thead><tr>${headerRow}</tr></thead>
            <tbody>${bodyRows}</tbody>
          </table>
        </div>`;
      })
      .replace(/^\- \[(x| )\] (.+)$/gm, (match, checked, text) => {
        const isChecked = checked === 'x';
        return `<div class="flex items-center space-x-2 my-1 font-mono text-sm">
          <input type="checkbox" ${isChecked ? 'checked' : ''} disabled class="rounded" />
          <span class="${isChecked ? 'line-through text-muted-foreground' : ''}">${text}</span>
        </div>`;
      })
      .replace(/^\- (.+)$/gm, '<li class="ml-4 my-1 font-mono text-sm">• $1</li>')
      .replace(/^\d+\. (.+)$/gm, '<li class="ml-4 my-1 list-decimal font-mono text-sm">$1</li>')
      .replace(/^> (.+)$/gm, '<blockquote class="border-l-4 border-primary pl-4 py-2 my-4 bg-muted/50 italic font-mono text-sm">$1</blockquote>')
      .replace(/^---$/gm, '<hr class="border-border my-6" />')
      .replace(/\n/g, '<br>');
    
    return rendered;
  };

  // Terminal functionality
  const addTerminalLine = (content: string, type: TerminalLine['type'] = 'output') => {
    setTerminalLines(prev => [...prev, { 
      id: Date.now(), 
      content, 
      type 
    }]);
  };

  const handleTerminalCommand = (command: string) => {
    addTerminalLine(`❯ ${command}`, 'command');
    const cmd = command.trim().toLowerCase();
    const args = command.trim().split(' ');

    if (cmd === 'clear') {
      setTerminalLines([]);
    } else if (cmd === 'help') {
      addTerminalLine('.\ available commands:', 'output');
      addTerminalLine('  python/py - enter python mode', 'output');
      addTerminalLine('  ai <msg> - query ai assistant', 'output');
      addTerminalLine('  save <name> - save current content', 'output');
      addTerminalLine('  load <name> - load file', 'output');
      addTerminalLine('  ls - list files', 'output');
      addTerminalLine('  clear - clear terminal', 'output');
    } else if (cmd === 'python' || cmd === 'py') {
      setIsMultiLine(true);
      addTerminalLine('.\ python mode activated', 'success');
      addTerminalLine('  ctrl+enter to execute, esc to exit', 'output');
    } else if (cmd.startsWith('ai ')) {
      const query = command.slice(3);
      simulateAiResponse(query);
    } else if (cmd === 'ls') {
      addTerminalLine(`.\ files (${files.length}):`, 'output');
      files.forEach(file => {
        addTerminalLine(`  ${file.name} (${file.type})`, 'output');
      });
    } else if (cmd.startsWith('save ')) {
      const name = args[1];
      if (name && content) {
        saveFile(name);
        addTerminalLine(`.\ saved "${name}"`, 'success');
      } else {
        addTerminalLine('.\ usage: save <filename>', 'error');
      }
    } else {
      addTerminalLine(`.\ command not found: ${cmd}`, 'error');
      addTerminalLine('  type "help" for available commands', 'output');
    }
  };

  const simulateAiResponse = (query: string) => {
    addTerminalLine(`.\ processing "${query}"...`, 'ai');
    
    setTimeout(() => {
      let response = '';
      if (query.includes('pytorch') || query.includes('torch')) {
        response = 'pytorch is a deep learning framework\n\nquick example:\nimport torch\nx = torch.randn(3, 3)\nprint(x)';
      } else if (query.includes('tinygrad')) {
        response = 'tinygrad is a lightweight deep learning framework\ngreat for learning and experimentation';
      } else if (query.includes('code') || query.includes('python')) {
        response = 'i can help with python code\nwhat specific functionality do you need?';
      } else {
        response = `regarding "${query}":\n\nthis is a simulated ai response\nin production, this would connect to a language model`;
      }
      
      addTerminalLine(`.\ ${response}`, 'ai');
    }, 1000);
  };

  // File operations
  const saveFile = (name?: string) => {
    const filename = name || fileName;
    if (!filename.trim()) {
      alert('please enter a filename');
      return;
    }

    const file: LarkFile = {
      id: currentFile?.id || Date.now().toString(),
      name: filename,
      content,
      type: 'markdown',
      createdAt: currentFile?.createdAt || new Date(),
      updatedAt: new Date()
    };

    const updatedFiles = files.filter(f => f.id !== file.id);
    updatedFiles.push(file);
    setFiles(updatedFiles);
    setCurrentFile(file);
    setFileName(filename);
    
    localStorage.setItem('lark-files', JSON.stringify(updatedFiles));
  };

  const loadFile = (file: LarkFile) => {
    setContent(file.content);
    setFileName(file.name);
    setCurrentFile(file);
  };

  const loadTemplate = (template: LarkTemplate) => {
    setContent(template.content);
    setFileName(`${template.name}.md`);
  };

  const printDocument = () => {
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      const html = `
<!DOCTYPE html>
<html>
<head>
    <title>${fileName || 'lark document'}</title>
    <style>
        body { font-family: 'SF Mono', Monaco, monospace; max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.6; }
        pre { background: #f5f5f5; padding: 1rem; border-radius: 4px; overflow: visible; }
        code { background: #f5f5f5; padding: 0.2rem 0.4rem; border-radius: 3px; font-family: inherit; }
        blockquote { border-left: 4px solid #ccc; margin: 0; padding-left: 1rem; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
        th { background: #f5f5f5; }
        @media print {
          body { font-size: 12px; }
          pre { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
${renderMarkdown(content)}
</body>
</html>`;
      printWindow.document.write(html);
      printWindow.document.close();
      printWindow.print();
    }
  };

  const exportHtml = () => {
    const html = `
<!DOCTYPE html>
<html>
<head>
    <title>${fileName || 'lark document'}</title>
    <style>
        body { font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.6; }
        pre { background: #f5f5f5; padding: 1rem; border-radius: 4px; overflow-x: auto; }
        code { background: #f5f5f5; padding: 0.2rem 0.4rem; border-radius: 3px; font-family: inherit; }
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

  // Asset handling
  const handleAssetSelect = (asset: LarkAsset) => {
    if (asset.type === 'image') {
      const imageMarkdown = `![${asset.name}](${asset.url})`;
      insertText(imageMarkdown);
    }
    // Close asset panel after selection
    setAssetPanelOpen(false);
  };

  const handleAssetUpload = (newAssets: LarkAsset[]) => {
    setAssets(prev => [...prev, ...newAssets]);
  };

  // Toolbar actions
  const insertText = (before: string, after: string = '') => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = content.substring(start, end);
    
    const newText = content.substring(0, start) + before + selectedText + after + content.substring(end);
    setContent(newText);
    
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + before.length, start + before.length + selectedText.length);
    }, 0);
  };

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
    table: () => insertText('| header 1 | header 2 |\n|----------|----------|\n| cell 1   | cell 2   |\n')
  };

  // Selection toolbar actions
  const applySelectionFormatting = (format: string) => {
    if (!selectedText) return;

    const textarea = textareaRef.current;
    if (!textarea) return;

    let before = '';
    let after = '';

    switch (format) {
      case 'bold':
        before = '**';
        after = '**';
        break;
      case 'italic':
        before = '*';
        after = '*';
        break;
      case 'code':
        before = '`';
        after = '`';
        break;
      case 'link':
        before = '[';
        after = '](url)';
        break;
    }

    // Find the position in the textarea content
    const textContent = textarea.value;
    const selectionStart = textContent.indexOf(selectedText.text);
    
    if (selectionStart !== -1) {
      const newText = textContent.substring(0, selectionStart) + 
                      before + selectedText.text + after + 
                      textContent.substring(selectionStart + selectedText.text.length);
      
      setContent(newText);
      setShowSelectionTools(false);
      setSelectedText(null);
    }
  };

  // Generate line numbers
  const getLineNumbers = () => {
    const lines = content.split('\n');
    return lines.map((_, index) => index + 1);
  };

  // Audio settings handlers
  const updateAudioSetting = (key: keyof AudioSettings, value: any) => {
    if (onAudioSettingsChange) {
      onAudioSettingsChange({ [key]: value });
    }
  };

  // Render different modes
  const renderEditor = () => (
    <div className="flex flex-col h-full bg-background relative">
      {/* Enhanced Toolbar */}
      <div className="flex items-center justify-between p-3 border-b border-border bg-background">
        <div className="flex items-center space-x-1">
          <span className="text-sm font-mono text-muted-foreground mr-3">.\ editor</span>
          <Button variant="ghost" size="sm" onClick={toolbarActions.bold} className="h-7 px-2">
            <span className="font-mono text-xs">B</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={toolbarActions.italic} className="h-7 px-2">
            <span className="font-mono text-xs">I</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={toolbarActions.code} className="h-7 px-2">
            <span className="font-mono text-xs">C</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={toolbarActions.heading1} className="h-7 px-2">
            <span className="font-mono text-xs">H1</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={toolbarActions.list} className="h-7 px-2">
            <span className="font-mono text-xs">L</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={toolbarActions.link} className="h-7 px-2">
            <span className="font-mono text-xs">lk</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={toolbarActions.image} className="h-7 px-2">
            <span className="font-mono text-xs">im</span>
          </Button>
        </div>

        <div className="flex items-center space-x-2">
          <div className="flex items-center border rounded-md">
            <Button 
              variant={viewMode === 'edit' ? 'default' : 'ghost'} 
              size="sm"
              onClick={() => setViewMode('edit')}
              className="h-7 px-2"
            >
              <span className="font-mono text-xs">edit</span>
            </Button>
            <Button 
              variant={viewMode === 'split' ? 'default' : 'ghost'} 
              size="sm"
              onClick={() => setViewMode('split')}
              className="h-7 px-2"
            >
              <span className="font-mono text-xs">split</span>
            </Button>
            <Button 
              variant={viewMode === 'preview' ? 'default' : 'ghost'} 
              size="sm"
              onClick={() => setViewMode('preview')}
              className="h-7 px-2"
            >
              <span className="font-mono text-xs">preview</span>
            </Button>
          </div>

          {/* Asset Panel Toggle */}
          <Button 
            variant={assetPanelOpen ? 'default' : 'ghost'} 
            size="sm" 
            onClick={() => setAssetPanelOpen(!assetPanelOpen)} 
            className="h-7 px-2"
          >
            <span className="font-mono text-xs">as</span>
          </Button>
          
          <Button variant="ghost" size="sm" onClick={() => navigator.clipboard.writeText(content)} className="h-7 px-2">
            <span className="font-mono text-xs">copy</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={printDocument} className="h-7 px-2">
            <span className="font-mono text-xs">print</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={exportHtml} className="h-7 px-2">
            <span className="font-mono text-xs">save</span>
          </Button>
        </div>
      </div>

      {/* File Management */}
      <div className="flex items-center space-x-2 p-3 bg-background border-b border-border">
        <span className="text-sm font-mono text-muted-foreground mr-2">file:</span>
        <Input
          value={fileName}
          onChange={(e) => setFileName(e.target.value)}
          placeholder="filename.md"
          className="w-48 h-7 font-mono text-sm"
        />
        <Button size="sm" onClick={() => saveFile()} className="h-7 px-3 font-mono text-sm">
          save
        </Button>
        
        {files.length > 0 && (
          <select
            onChange={(e) => {
              const file = files.find(f => f.id === e.target.value);
              if (file) loadFile(file);
            }}
            className="px-2 py-1 border border-border rounded-md bg-background text-sm font-mono h-7"
          >
            <option value="">load...</option>
            {files.map(file => (
              <option key={file.id} value={file.id}>
                {file.name}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Main Editor Area */}
      <div className="flex-1 flex overflow-hidden relative">
        {(viewMode === 'edit' || viewMode === 'split') && (
          <div className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} flex relative`}>
            {/* Editor */}
            <div className="flex-1 flex flex-col">
              <textarea
                ref={textareaRef}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                onMouseUp={handleTextSelection}
                className="flex-1 border-0 resize-none font-mono text-sm bg-background text-foreground p-4 focus:outline-none overflow-y-auto subtle-scrollbar"
                placeholder=".\ start writing..."
                style={{ lineHeight: '1.5rem' }}
              />
            </div>

            {/* Line Numbers - Right Side with Synchronized Scrolling */}
            <div 
              ref={lineNumbersRef}
              className="w-16 bg-auto border-l border-border flex flex-col py-4 text-left pl-2 relative overflow-y-auto overflow-x-hidden"
              style={{ 
                maxHeight: '100%',
                scrollbarWidth: 'none', /* Firefox */
                msOverflowStyle: 'none'  /* Internet Explorer 10+ */
              }}
            >
              {/* Hide scrollbar for webkit browsers */}
              <style jsx>{`
                div::-webkit-scrollbar {
                  display: none;
                }
              `}</style>
              
              {getLineNumbers().map((lineNum) => (
                <div 
                  key={lineNum} 
                  className="relative h-6 flex items-center group animate-pulse-dot flex-shrink-0"
                  style={{ 
                    color: 'rgba(70, 70, 70, 70.1)',
                    fontSize: '12px',
                    fontFamily: 'monospace',
                    lineHeight: '1.5rem'
                  }}
                >
                  {/* Animated dots */}
                  <div className="absolute left-0 flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity animate-terminal-line">
                    <div className="w-1 h-1 bg-current rounded-full animate-pulse"></div>
                    <div className="w-1 h-1 bg-current rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-1 h-1 bg-current rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                  
                  {/* Line number */}
                  <span className="ml-8 warp-section-highlight animate-underline">
                    {lineNum}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {viewMode === 'split' && <div className="w-px bg-border" />}

        {(viewMode === 'preview' || viewMode === 'split') && (
          <div className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} flex flex-col`}>
            <div 
              ref={previewRef}
              className="flex-1 p-4 font-mono text-sm overflow-y-auto bg-background subtle-scrollbar"
              onMouseUp={handleTextSelection}
              dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }}
            />
          </div>
        )}

        {/* Asset Manager Popup - Appears in open area */}
        {assetPanelOpen && (
          <div className="absolute inset-x-4 bottom-4 top-20 bg-card/95 backdrop-blur-sm border border-border rounded-lg shadow-xl z-50 terminal-section-fade">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="font-mono text-sm animate-underline">asset manager</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setAssetPanelOpen(false)}
                className="h-6 w-6 p-0 animate-border-grow"
              >
                <span className="font-mono text-xs">×</span>
              </Button>
            </div>
            <div className="flex-1 p-4 overflow-hidden">
              <LarkAssetManager
                onAssetSelect={handleAssetSelect}
                onAssetUpload={handleAssetUpload}
                onAssetsChange={setAssets}
                initialAssets={assets}
                height="h-full"
                acceptedTypes={['image/*', 'video/*', 'audio/*', '.pdf', '.md', '.txt']}
              />
            </div>
          </div>
        )}
      </div>

      {/* Selection Tools */}
      {showSelectionTools && selectedText && (
        <div 
          className="fixed z-50 bg-card border border-border rounded-lg shadow-lg p-2 flex items-center space-x-1 terminal-section-fade"
          style={{ 
            left: `${selectionToolsPosition.x}px`, 
            top: `${selectionToolsPosition.y}px`,
            transform: 'translateX(-50%)'
          }}
        >
          <Button
            variant="ghost"
            size="sm"
            onClick={() => applySelectionFormatting('bold')}
            className="h-6 px-2 font-mono text-xs"
          >
            B
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => applySelectionFormatting('italic')}
            className="h-6 px-2 font-mono text-xs"
          >
            I
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => applySelectionFormatting('code')}
            className="h-6 px-2 font-mono text-xs"
          >
            C
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => applySelectionFormatting('link')}
            className="h-6 px-2 font-mono text-xs"
          >
            lk
          </Button>
        </div>
      )}
    </div>
  );

  const renderAI = () => (
    <div className="flex h-full bg-background">
      <div className="w-64 border-r border-border p-4">
        <div className="mb-4">
          <span className="font-mono text-sm text-muted-foreground">.\ conversations</span>
        </div>
        <div className="space-y-1">
          {conversations.length === 0 ? (
            <p className="text-sm font-mono text-muted-foreground">no conversations</p>
          ) : (
            conversations.map(conv => (
              <div key={conv.id} className="p-2 hover:bg-muted rounded cursor-pointer">
                <div className="font-mono text-sm">{conv.title}</div>
                <div className="text-xs font-mono text-muted-foreground">
                  {conv.messages.length} messages
                </div>
              </div>
            ))
          )}
        </div>
      </div>
      
      <div className="flex-1 flex flex-col">
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="max-w-2xl mx-auto space-y-6">
            <div className="text-center">
              <h2 className="text-lg font-mono mb-2">.\ ai assistant</h2>
              <p className="text-sm font-mono text-muted-foreground">
                ask about ai, machine learning, or code help
              </p>
            </div>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="p-4 border border-border rounded hover:bg-muted/50 cursor-pointer" 
                   onClick={() => setAiInput('explain pytorch basics')}>
                <div className="font-mono text-sm mb-1">pytorch help</div>
                <div className="text-xs font-mono text-muted-foreground">
                  concepts and code examples
                </div>
              </div>
              
              <div className="p-4 border border-border rounded hover:bg-muted/50 cursor-pointer" 
                   onClick={() => setAiInput('review my code')}>
                <div className="font-mono text-sm mb-1">code review</div>
                <div className="text-xs font-mono text-muted-foreground">
                  bug fixes and improvements
                </div>
              </div>
              
              <div className="p-4 border border-border rounded hover:bg-muted/50 cursor-pointer" 
                   onClick={() => setAiInput('ml project ideas')}>
                <div className="font-mono text-sm mb-1">project ideas</div>
                <div className="text-xs font-mono text-muted-foreground">
                  creative suggestions
                </div>
              </div>
              
              <div className="p-4 border border-border rounded hover:bg-muted/50 cursor-pointer" 
                   onClick={() => setAiInput('optimize performance')}>
                <div className="font-mono text-sm mb-1">optimization</div>
                <div className="text-xs font-mono text-muted-foreground">
                  speed and accuracy tips
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="p-4 border-t border-border">
          <div className="flex space-x-2">
            <Input
              value={aiInput}
              onChange={(e) => setAiInput(e.target.value)}
              placeholder="ask anything..."
              className="flex-1 font-mono text-sm"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (aiInput.trim()) {
                    simulateAiResponse(aiInput);
                    setAiInput('');
                  }
                }
              }}
            />
            <Button onClick={() => {
              if (aiInput.trim()) {
                simulateAiResponse(aiInput);
                setAiInput('');
              }
            }} className="font-mono text-sm">
              send
            </Button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderFiles = () => (
    <div className="p-6 bg-background">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <span className="font-mono text-muted-foreground">.\ files</span>
          <Button variant="ghost" size="sm" className="font-mono text-sm">
            import
          </Button>
        </div>
        
        <div className="space-y-2">
          {files.map(file => (
            <div key={file.id} className="flex items-center justify-between p-3 border border-border rounded hover:bg-muted/50">
              <div className="flex items-center space-x-3">
                <span className="font-mono text-sm">{file.name}</span>
                <span className="text-xs font-mono text-muted-foreground">
                  {file.type}
                </span>
                <span className="text-xs font-mono text-muted-foreground">
                  {file.updatedAt.toLocaleDateString()}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <Button size="sm" onClick={() => loadFile(file)} className="font-mono text-sm h-7">
                  load
                </Button>
                <Button size="sm" variant="ghost" onClick={() => {
                  const updatedFiles = files.filter(f => f.id !== file.id);
                  setFiles(updatedFiles);
                  localStorage.setItem('lark-files', JSON.stringify(updatedFiles));
                }} className="font-mono text-sm h-7">
                  delete
                </Button>
              </div>
            </div>
          ))}
          
          {files.length === 0 && (
            <div className="text-center py-12">
              <p className="font-mono text-muted-foreground">no files yet</p>
              <p className="font-mono text-sm text-muted-foreground mt-1">
                create your first file in the editor
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderTemplates = () => (
    <div className="p-6 bg-background">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <span className="font-mono text-muted-foreground">.\ templates</span>
          <span className="font-mono text-sm text-muted-foreground">
            {templates.length} available
          </span>
        </div>
        
        <div className="space-y-3">
          {templates.map(template => (
            <div key={template.id} className="flex items-center justify-between p-4 border border-border rounded hover:bg-muted/50">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-1">
                  <span className="font-mono text-sm">{template.name}</span>
                  <span className="text-xs font-mono text-muted-foreground px-2 py-1 bg-muted rounded">
                    {template.category}
                  </span>
                </div>
                <p className="font-mono text-sm text-muted-foreground">
                  {template.description}
                </p>
              </div>
              <Button size="sm" onClick={() => loadTemplate(template)} className="font-mono text-sm">
                use
              </Button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderSettings = () => (
    <div className="p-6 bg-background">
      <div className="max-w-2xl mx-auto">
        <div className="mb-6">
          <span className="font-mono text-muted-foreground">.\ settings</span>
        </div>
        
        <Tabs defaultValue="appearance" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="appearance">
              <span className="font-mono text-xs">view</span>
            </TabsTrigger>
            <TabsTrigger value="audio">
              <span className="font-mono text-xs">audio</span>
            </TabsTrigger>
            <TabsTrigger value="ai">
              <span className="font-mono text-xs">ai</span>
            </TabsTrigger>
            <TabsTrigger value="data">
              <span className="font-mono text-xs">data</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="appearance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="font-mono text-sm">appearance</CardTitle>
                <CardDescription className="font-mono text-xs">
                  customize the visual interface
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">theme</Label>
                  <Select defaultValue="light">
                    <SelectTrigger className="w-32 font-mono text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">light</SelectItem>
                      <SelectItem value="dark">dark</SelectItem>
                      <SelectItem value="auto">auto</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">font size</Label>
                  <Select defaultValue="14px">
                    <SelectTrigger className="w-32 font-mono text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="12px">12px</SelectItem>
                      <SelectItem value="14px">14px</SelectItem>
                      <SelectItem value="16px">16px</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="audio" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="font-mono text-sm">waveform display</CardTitle>
                <CardDescription className="font-mono text-xs">
                  control visual audio feedback
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">enable waveform</Label>
                  <Switch
                    checked={audioSettings.waveformEnabled}
                    onCheckedChange={(checked) => updateAudioSetting('waveformEnabled', checked)}
                  />
                </div>

                <div className="space-y-2">
                  <Label className="font-mono text-sm">sensitivity: {audioSettings.sensitivity.toFixed(1)}</Label>
                  <Slider
                    value={[audioSettings.sensitivity]}
                    onValueChange={([value]) => updateAudioSetting('sensitivity', value)}
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <Label className="font-mono text-sm">animation speed: {audioSettings.animationSpeed.toFixed(1)}x</Label>
                  <Slider
                    value={[audioSettings.animationSpeed]}
                    onValueChange={([value]) => updateAudioSetting('animationSpeed', value)}
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    className="w-full"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">waveform style</Label>
                  <Select
                    value={audioSettings.waveformStyle}
                    onValueChange={(value: 'minimal' | 'standard' | 'detailed') => 
                      updateAudioSetting('waveformStyle', value)
                    }
                  >
                    <SelectTrigger className="w-32 font-mono text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="minimal">minimal</SelectItem>
                      <SelectItem value="standard">standard</SelectItem>
                      <SelectItem value="detailed">detailed</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">waveform color</Label>
                  <Select
                    value={audioSettings.waveformColor}
                    onValueChange={(value: 'default' | 'accent' | 'primary') => 
                      updateAudioSetting('waveformColor', value)
                    }
                  >
                    <SelectTrigger className="w-32 font-mono text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">default</SelectItem>
                      <SelectItem value="accent">accent</SelectItem>
                      <SelectItem value="primary">primary</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">visual feedback</Label>
                  <Switch
                    checked={audioSettings.visualFeedbackEnabled}
                    onCheckedChange={(checked) => updateAudioSetting('visualFeedbackEnabled', checked)}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="font-mono text-sm">text rotation</CardTitle>
                <CardDescription className="font-mono text-xs">
                  configure idle phrase display
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">enable phrases</Label>
                  <Switch
                    checked={audioSettings.phraseRotationEnabled}
                    onCheckedChange={(checked) => updateAudioSetting('phraseRotationEnabled', checked)}
                  />
                </div>

                <div className="space-y-2">
                  <Label className="font-mono text-sm">rotation speed: {audioSettings.phraseRotationSpeed}s</Label>
                  <Slider
                    value={[audioSettings.phraseRotationSpeed]}
                    onValueChange={([value]) => updateAudioSetting('phraseRotationSpeed', value)}
                    min={1}
                    max={10}
                    step={1}
                    className="w-full"
                    disabled={!audioSettings.phraseRotationEnabled}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="font-mono text-sm">activity detection</CardTitle>
                <CardDescription className="font-mono text-xs">
                  automatic audio state management
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">auto detection</Label>
                  <Switch
                    checked={audioSettings.autoActivityDetection}
                    onCheckedChange={(checked) => updateAudioSetting('autoActivityDetection', checked)}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="ai" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="font-mono text-sm">ai configuration</CardTitle>
                <CardDescription className="font-mono text-xs">
                  ai model and api settings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="font-mono text-sm">model</Label>
                  <Select defaultValue="simulated">
                    <SelectTrigger className="w-48 font-mono text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="simulated">simulated (demo)</SelectItem>
                      <SelectItem value="gpt-4">gpt-4</SelectItem>
                      <SelectItem value="claude">claude</SelectItem>
                      <SelectItem value="local">local model</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="font-mono text-sm">api key</Label>
                  <Input 
                    type="password" 
                    placeholder="enter api key..." 
                    className="font-mono text-sm" 
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="data" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="font-mono text-sm">data management</CardTitle>
                <CardDescription className="font-mono text-xs">
                  import, export, and reset data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex space-x-2">
                  <Button variant="ghost" size="sm" className="font-mono text-sm">
                    export data
                  </Button>
                  <Button variant="ghost" size="sm" className="font-mono text-sm">
                    import data
                  </Button>
                </div>
                <Separator />
                <Button variant="ghost" size="sm" className="w-full font-mono text-sm text-destructive">
                  clear all data
                </Button>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );

  // Render based on mode
  switch (mode) {
    case 'editor':
      return renderEditor();
    case 'ai':
      return renderAI();
    case 'files':
      return renderFiles();
    case 'templates':
      return renderTemplates();
    case 'settings':
      return renderSettings();
    default:
      return renderEditor();
  }
}