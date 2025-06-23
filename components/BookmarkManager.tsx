import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ScrollArea } from './ui/scroll-area';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Textarea } from './ui/textarea';
import { Separator } from './ui/separator';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

interface CodeSnippet {
  id: string;
  language: string;
  code: string;
  lineStart?: number;
  lineEnd?: number;
}

interface BookmarkEntry {
  id: string;
  url: string;
  title: string;
  description: string;
  summary: string;
  tags: string[];
  project?: string;
  content: string;
  codeSnippets: CodeSnippet[];
  createdAt: Date;
  updatedAt: Date;
  metadata: {
    favicon?: string;
    author?: string;
    publishDate?: string;
    domain: string;
    status: 'archived' | 'processing' | 'failed';
  };
  notes: string;
  archived: boolean;
  screenshot?: string;
}

interface Project {
  id: string;
  name: string;
  description: string;
  bookmarks: string[]; // bookmark IDs
  createdAt: Date;
}

// Helper function to ensure dates are Date objects
const ensureDate = (date: any): Date => {
  if (date instanceof Date) {
    return date;
  }
  if (typeof date === 'string' || typeof date === 'number') {
    return new Date(date);
  }
  return new Date();
};

// Helper function to convert bookmark objects with proper dates
const normalizeBookmark = (bookmark: any): BookmarkEntry => {
  return {
    ...bookmark,
    createdAt: ensureDate(bookmark.createdAt),
    updatedAt: ensureDate(bookmark.updatedAt)
  };
};

// Helper function to convert project objects with proper dates
const normalizeProject = (project: any): Project => {
  return {
    ...project,
    createdAt: ensureDate(project.createdAt)
  };
};

export function BookmarkManager() {
  const [bookmarks, setBookmarks] = useState<BookmarkEntry[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedBookmark, setSelectedBookmark] = useState<BookmarkEntry | null>(null);
  const [activeView, setActiveView] = useState<'grid' | 'list' | 'graph'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedProject, setSelectedProject] = useState<string>('all');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newBookmarkUrl, setNewBookmarkUrl] = useState('');
  const [newBookmarkTags, setNewBookmarkTags] = useState('');
  const [newBookmarkProject, setNewBookmarkProject] = useState('');

  // Demo data initialization
  useEffect(() => {
    const savedBookmarks = localStorage.getItem('codex-bookmarks');
    const savedProjects = localStorage.getItem('codex-projects');

    if (!savedBookmarks) {
      // Initialize with demo bookmarks
      const demoBookmarks: BookmarkEntry[] = [
        {
          id: 'bm1',
          url: 'https://github.com/tinygrad/tinygrad',
          title: 'tinygrad - A simple and powerful deep learning framework',
          description: 'A neural network framework for training and inference, inspired by PyTorch',
          summary: 'TinyGrad is a minimal neural network framework written in Python, designed for learning and experimentation. Features automatic differentiation, GPU support, and a PyTorch-like API.',
          tags: ['ml', 'framework', 'python', 'deep-learning'],
          project: 'local-llm',
          content: 'TinyGrad repository content...',
          codeSnippets: [
            {
              id: 'cs1',
              language: 'python',
              code: `from tinygrad.tensor import Tensor
x = Tensor.randn(3, 3)
y = x.relu()
print(y.numpy())`,
              lineStart: 1,
              lineEnd: 4
            }
          ],
          createdAt: new Date('2024-01-15'),
          updatedAt: new Date('2024-01-15'),
          metadata: {
            domain: 'github.com',
            status: 'archived' as const,
            author: 'George Hotz'
          },
          notes: 'Excellent for learning how neural networks work under the hood. Clean codebase.',
          archived: false
        },
        {
          id: 'bm2',
          url: 'https://pytorch.org/tutorials/beginner/basics/intro.html',
          title: 'PyTorch Tutorials - Learn the Basics',
          description: 'Official PyTorch tutorials covering tensors, datasets, models, and training',
          summary: 'Comprehensive tutorial series covering PyTorch fundamentals including tensor operations, autograd, neural networks, and training loops. Perfect for beginners.',
          tags: ['pytorch', 'tutorial', 'deep-learning', 'beginner'],
          project: 'local-llm',
          content: 'PyTorch tutorial content...',
          codeSnippets: [
            {
              id: 'cs2',
              language: 'python',
              code: `import torch
x = torch.rand(5, 3)
print(x)`
            }
          ],
          createdAt: new Date('2024-01-10'),
          updatedAt: new Date('2024-01-10'),
          metadata: {
            domain: 'pytorch.org',
            status: 'archived' as const
          },
          notes: 'Official tutorials - very comprehensive',
          archived: false
        },
        {
          id: 'bm3',
          url: 'https://huggingface.co/transformers/',
          title: 'Transformers by Hugging Face',
          description: 'State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX',
          summary: 'Library providing thousands of pretrained models for Natural Language Processing, Computer Vision, and Audio tasks. Easy-to-use APIs for training and inference.',
          tags: ['transformers', 'nlp', 'huggingface', 'pretrained'],
          project: 'local-llm',
          content: 'Hugging Face transformers documentation...',
          codeSnippets: [
            {
              id: 'cs3',
              language: 'python',
              code: `from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")`
            }
          ],
          createdAt: new Date('2024-01-20'),
          updatedAt: new Date('2024-01-20'),
          metadata: {
            domain: 'huggingface.co',
            status: 'archived' as const
          },
          notes: 'Essential for working with pretrained language models',
          archived: false
        }
      ];

      const demoProjects: Project[] = [
        {
          id: 'proj1',
          name: 'local-llm',
          description: 'Building a local language model from scratch',
          bookmarks: ['bm1', 'bm2', 'bm3'],
          createdAt: new Date('2024-01-01')
        }
      ];

      setBookmarks(demoBookmarks);
      setProjects(demoProjects);
      localStorage.setItem('codex-bookmarks', JSON.stringify(demoBookmarks));
      localStorage.setItem('codex-projects', JSON.stringify(demoProjects));
    } else {
      // Parse and normalize saved data
      const parsedBookmarks = JSON.parse(savedBookmarks).map(normalizeBookmark);
      setBookmarks(parsedBookmarks);
      
      const parsedProjects = savedProjects 
        ? JSON.parse(savedProjects).map(normalizeProject) 
        : [];
      setProjects(parsedProjects);
    }
  }, []);

  // Save to localStorage when bookmarks change
  useEffect(() => {
    if (bookmarks.length > 0) {
      localStorage.setItem('codex-bookmarks', JSON.stringify(bookmarks));
    }
  }, [bookmarks]);

  useEffect(() => {
    if (projects.length > 0) {
      localStorage.setItem('codex-projects', JSON.stringify(projects));
    }
  }, [projects]);

  // Simulate bookmark processing (triage)
  const processBookmark = async (url: string, tags: string[], project?: string) => {
    const newBookmark: BookmarkEntry = {
      id: `bm${Date.now()}`,
      url,
      title: 'Processing...',
      description: 'Extracting content and metadata...',
      summary: '',
      tags,
      project,
      content: '',
      codeSnippets: [],
      createdAt: new Date(),
      updatedAt: new Date(),
      metadata: {
        domain: new URL(url).hostname,
        status: 'processing' as const
      },
      notes: '',
      archived: false
    };

    setBookmarks(prev => [newBookmark, ...prev]);

    // Simulate processing delay
    setTimeout(() => {
      setBookmarks(prev => prev.map(bm => 
        bm.id === newBookmark.id 
          ? {
              ...bm,
              title: `Page from ${new URL(url).hostname}`,
              description: 'Archived page content',
              summary: 'AI-generated summary of the page content would appear here. This is a simulation of the content extraction and summarization process.',
              content: 'Full page content would be stored here...',
              metadata: {
                ...bm.metadata,
                status: 'archived' as const
              },
              updatedAt: new Date()
            }
          : bm
      ));
    }, 2000);
  };

  // Filter bookmarks based on search and filters
  const filteredBookmarks = bookmarks.filter(bookmark => {
    const matchesSearch = !searchQuery || 
      bookmark.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      bookmark.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      bookmark.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesProject = selectedProject === 'all' || bookmark.project === selectedProject;
    
    const matchesTags = selectedTags.length === 0 || 
      selectedTags.every(tag => bookmark.tags.includes(tag));

    return matchesSearch && matchesProject && matchesTags;
  });

  // Get all unique tags
  const allTags = Array.from(new Set(bookmarks.flatMap(bm => bm.tags))).sort();

  const handleAddBookmark = () => {
    if (!newBookmarkUrl.trim()) return;

    const tags = newBookmarkTags.split(',').map(tag => tag.trim()).filter(Boolean);
    processBookmark(newBookmarkUrl, tags, newBookmarkProject || undefined);
    
    setNewBookmarkUrl('');
    setNewBookmarkTags('');
    setNewBookmarkProject('');
    setShowAddForm(false);
  };

  // Safe date formatting function
  const formatDate = (date: Date | string | number): string => {
    try {
      const dateObj = ensureDate(date);
      return dateObj.toLocaleDateString();
    } catch (error) {
      console.warn('Date formatting error:', error);
      return 'Invalid date';
    }
  };

  const renderBookmarkCard = (bookmark: BookmarkEntry) => (
    <Card 
      key={bookmark.id} 
      className={`cursor-pointer hover:bg-muted/50 transition-colors animate-terminal-fade ${
        activeView === 'grid' ? 'h-64 flex flex-col' : ''
      }`}
      onClick={() => setSelectedBookmark(bookmark)}
    >
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-start justify-between space-x-2">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-sm font-mono line-clamp-2 animate-underline leading-tight">
              {bookmark.title}
            </CardTitle>
            <CardDescription className="text-xs font-mono mt-1 truncate">
              {bookmark.metadata.domain}
            </CardDescription>
          </div>
          <Badge 
            variant={bookmark.metadata.status === 'archived' ? 'default' : 'secondary'}
            className="text-xs font-mono flex-shrink-0"
          >
            {bookmark.metadata.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className={`pt-0 ${activeView === 'grid' ? 'flex-1 flex flex-col overflow-hidden' : ''}`}>
        <p className={`text-sm font-mono text-muted-foreground mb-3 ${
          activeView === 'grid' ? 'line-clamp-3 flex-1' : 'line-clamp-2'
        }`}>
          {bookmark.description}
        </p>
        <div className="flex flex-wrap gap-1 mb-3 min-h-[1.5rem] overflow-hidden">
          {bookmark.tags.slice(0, activeView === 'grid' ? 2 : 3).map(tag => (
            <Badge 
              key={tag} 
              variant="outline" 
              className="text-xs font-mono truncate max-w-[5rem] flex-shrink-0"
              title={tag}
            >
              {tag}
            </Badge>
          ))}
          {bookmark.tags.length > (activeView === 'grid' ? 2 : 3) && (
            <Badge variant="outline" className="text-xs font-mono flex-shrink-0">
              +{bookmark.tags.length - (activeView === 'grid' ? 2 : 3)}
            </Badge>
          )}
        </div>
        <div className="flex items-center justify-between text-xs font-mono text-muted-foreground mt-auto">
          <span className="truncate flex-1 mr-2">{bookmark.project || 'No project'}</span>
          <span className="flex-shrink-0">{formatDate(bookmark.createdAt)}</span>
        </div>
      </CardContent>
    </Card>
  );

  const renderDetailView = () => {
    if (!selectedBookmark) return null;

    return (
      <Dialog open={!!selectedBookmark} onOpenChange={() => setSelectedBookmark(null)}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="font-mono text-sm animate-underline">
              {selectedBookmark.title}
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            {/* Metadata */}
            <div className="flex items-center space-x-4 text-sm font-mono">
              <Badge variant="outline">{selectedBookmark.metadata.domain}</Badge>
              <Badge variant={selectedBookmark.metadata.status === 'archived' ? 'default' : 'secondary'}>
                {selectedBookmark.metadata.status}
              </Badge>
              <span className="text-muted-foreground">
                {formatDate(selectedBookmark.createdAt)}
              </span>
            </div>

            {/* URL */}
            <div className="p-3 bg-muted rounded-md">
              <p className="text-sm font-mono break-all">{selectedBookmark.url}</p>
            </div>

            {/* Summary */}
            <div>
              <h4 className="font-mono text-sm mb-2 animate-underline">AI Summary</h4>
              <p className="text-sm font-mono text-muted-foreground">
                {selectedBookmark.summary || 'No summary available'}
              </p>
            </div>

            {/* Tags */}
            <div>
              <h4 className="font-mono text-sm mb-2 animate-underline">Tags</h4>
              <div className="flex flex-wrap gap-1">
                {selectedBookmark.tags.map(tag => (
                  <Badge key={tag} variant="outline" className="text-xs font-mono">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Code Snippets */}
            {selectedBookmark.codeSnippets.length > 0 && (
              <div>
                <h4 className="font-mono text-sm mb-2 animate-underline">Extracted Code</h4>
                <div className="space-y-3">
                  {selectedBookmark.codeSnippets.map(snippet => (
                    <div key={snippet.id} className="border rounded-md">
                      <div className="bg-muted px-3 py-1 flex items-center justify-between border-b">
                        <span className="text-xs font-mono">{snippet.language}</span>
                        <Button 
                          size="sm" 
                          variant="ghost" 
                          className="h-6 px-2 font-mono text-xs"
                          onClick={() => navigator.clipboard.writeText(snippet.code)}
                        >
                          copy
                        </Button>
                      </div>
                      <pre className="p-3 overflow-x-auto">
                        <code className="text-sm font-mono">{snippet.code}</code>
                      </pre>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Notes */}
            <div>
              <h4 className="font-mono text-sm mb-2 animate-underline">Notes</h4>
              <Textarea
                value={selectedBookmark.notes}
                onChange={(e) => {
                  const updatedBookmark = { ...selectedBookmark, notes: e.target.value, updatedAt: new Date() };
                  setSelectedBookmark(updatedBookmark);
                  setBookmarks(prev => prev.map(bm => 
                    bm.id === selectedBookmark.id ? updatedBookmark : bm
                  ));
                }}
                placeholder="Add your notes about this bookmark..."
                className="font-mono text-sm"
                rows={4}
              />
            </div>

            {/* Actions */}
            <div className="flex space-x-2">
              <Button 
                size="sm" 
                className="font-mono text-sm"
                onClick={() => window.open(selectedBookmark.url, '_blank')}
              >
                Open Original
              </Button>
              <Button 
                size="sm" 
                variant="outline" 
                className="font-mono text-sm"
                onClick={() => {
                  const exportData = {
                    title: selectedBookmark.title,
                    url: selectedBookmark.url,
                    notes: selectedBookmark.notes,
                    tags: selectedBookmark.tags,
                    summary: selectedBookmark.summary
                  };
                  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `${selectedBookmark.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                Export Notes
              </Button>
              <Button size="sm" variant="outline" className="font-mono text-sm">
                Clone to Project
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    );
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="font-mono text-lg animate-underline">.\ codex</h1>
            <p className="text-sm font-mono text-muted-foreground">
              Actionable Knowledge Environment
            </p>
          </div>
          <Button 
            onClick={() => setShowAddForm(true)} 
            className="font-mono text-sm animate-border-grow"
          >
            cx add
          </Button>
        </div>

        {/* Search and Filters */}
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <Input
              placeholder="Search bookmarks, tags, content..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="font-mono text-sm"
            />
          </div>
          
          <Select value={selectedProject} onValueChange={setSelectedProject}>
            <SelectTrigger className="w-48 font-mono text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Projects</SelectItem>
              {projects.map(project => (
                <SelectItem key={project.id} value={project.id}>
                  {project.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            variant={activeView === 'grid' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveView('grid')}
            className="font-mono text-xs px-2"
          >
            grid
          </Button>
          <Button
            variant={activeView === 'list' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveView('list')}
            className="font-mono text-xs px-2"
          >
            list
          </Button>
        </div>

        {/* Tag Filter */}
        {allTags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-3">
            {allTags.slice(0, 10).map(tag => (
              <Badge
                key={tag}
                variant={selectedTags.includes(tag) ? 'default' : 'outline'}
                className="text-xs font-mono cursor-pointer hover:bg-muted animate-border-grow"
                onClick={() => {
                  setSelectedTags(prev => 
                    prev.includes(tag) 
                      ? prev.filter(t => t !== tag)
                      : [...prev, tag]
                  );
                }}
              >
                {tag}
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 p-4 overflow-y-auto subtle-scrollbar">
        {filteredBookmarks.length === 0 ? (
          <div className="text-center py-12">
            <p className="font-mono text-muted-foreground mb-4">
              {bookmarks.length === 0 ? 'No bookmarks yet' : 'No bookmarks match your filters'}
            </p>
            <Button onClick={() => setShowAddForm(true)} className="font-mono text-sm">
              Add Your First Bookmark
            </Button>
          </div>
        ) : (
          <div className={
            activeView === 'grid' 
              ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4'
              : 'space-y-3'
          }>
            {filteredBookmarks.map(renderBookmarkCard)}
          </div>
        )}
      </div>

      {/* Add Bookmark Dialog */}
      <Dialog open={showAddForm} onOpenChange={setShowAddForm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="font-mono text-sm animate-underline">
              cx add - New Bookmark
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-mono mb-2">URL</label>
              <Input
                value={newBookmarkUrl}
                onChange={(e) => setNewBookmarkUrl(e.target.value)}
                placeholder="https://example.com"
                className="font-mono text-sm"
              />
            </div>
            <div>
              <label className="block text-sm font-mono mb-2">Tags</label>
              <Input
                value={newBookmarkTags}
                onChange={(e) => setNewBookmarkTags(e.target.value)}
                placeholder="ml, framework, python"
                className="font-mono text-sm"
              />
            </div>
            <div>
              <label className="block text-sm font-mono mb-2">Project (optional)</label>
              <Select value={newBookmarkProject} onValueChange={setNewBookmarkProject}>
                <SelectTrigger className="font-mono text-sm">
                  <SelectValue placeholder="Select project..." />
                </SelectTrigger>
                <SelectContent>
                  {projects.map(project => (
                    <SelectItem key={project.id} value={project.id}>
                      {project.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex space-x-2">
              <Button onClick={handleAddBookmark} className="font-mono text-sm">
                Process & Archive
              </Button>
              <Button variant="outline" onClick={() => setShowAddForm(false)} className="font-mono text-sm">
                Cancel
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Detail View */}
      {renderDetailView()}
    </div>
  );
}