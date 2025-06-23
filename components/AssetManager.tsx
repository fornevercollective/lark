import React, { useState, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface AssetFile {
  id: string;
  name: string;
  type: 'image' | 'document' | 'code' | 'data';
  size: number;
  url?: string;
  content?: string;
  thumbnail?: string;
  createdAt: Date;
  updatedAt: Date;
  tags: string[];
  source: 'upload' | 'ai-generated' | 'terminal' | 'editor';
}

interface AssetManagerProps {
  isOpen: boolean;
  onClose: () => void;
  height?: number;
}

export interface AssetManagerRef {
  addAsset: (asset: Partial<AssetFile>) => void;
  addImageFromUrl: (url: string, name: string, source?: AssetFile['source']) => void;
  addTextContent: (content: string, name: string, type?: 'document' | 'code', source?: AssetFile['source']) => void;
}

type ViewMode = 'grid' | 'list';
type SortBy = 'name' | 'date' | 'size' | 'type';
type FilterBy = 'all' | 'images' | 'documents' | 'code' | 'data';

export const AssetManager = forwardRef<AssetManagerRef, AssetManagerProps>(
  ({ isOpen, onClose, height = 240 }, ref) => {
    const [assets, setAssets] = useState<AssetFile[]>([
      // Sample data
      {
        id: '1',
        name: 'pytorch_model_diagram.png',
        type: 'image',
        size: 245760,
        url: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=400&h=300&fit=crop',
        thumbnail: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=150&h=100&fit=crop',
        createdAt: new Date('2024-01-15'),
        updatedAt: new Date('2024-01-15'),
        tags: ['pytorch', 'neural-network', 'diagram'],
        source: 'ai-generated'
      },
      {
        id: '2',
        name: 'training_loss_chart.png',
        type: 'image',
        size: 128000,
        url: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&h=300&fit=crop',
        thumbnail: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=150&h=100&fit=crop',
        createdAt: new Date('2024-01-14'),
        updatedAt: new Date('2024-01-14'),
        tags: ['chart', 'loss', 'training'],
        source: 'terminal'
      },
      {
        id: '3',
        name: 'research_notes.md',
        type: 'document',
        size: 8192,
        content: '# Research Notes\n\nExperiment results...',
        createdAt: new Date('2024-01-13'),
        updatedAt: new Date('2024-01-16'),
        tags: ['research', 'notes', 'markdown'],
        source: 'editor'
      },
      {
        id: '4',
        name: 'model_architecture.py',
        type: 'code',
        size: 4096,
        content: 'import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.Sequential(...)',
        createdAt: new Date('2024-01-12'),
        updatedAt: new Date('2024-01-15'),
        tags: ['python', 'pytorch', 'model'],
        source: 'editor'
      },
      {
        id: '5',
        name: 'dataset_analysis.png',
        type: 'image',
        size: 189440,
        url: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&h=300&fit=crop',
        thumbnail: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=150&h=100&fit=crop',
        createdAt: new Date('2024-01-11'),
        updatedAt: new Date('2024-01-11'),
        tags: ['dataset', 'analysis', 'visualization'],
        source: 'ai-generated'
      }
    ]);

    const [viewMode, setViewMode] = useState<ViewMode>('grid');
    const [sortBy, setSortBy] = useState<SortBy>('date');
    const [filterBy, setFilterBy] = useState<FilterBy>('all');
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
    const [dragOver, setDragOver] = useState(false);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Expose methods to parent components
    useImperativeHandle(ref, () => ({
      addAsset: (asset: Partial<AssetFile>) => {
        const newAsset: AssetFile = {
          id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
          name: asset.name || 'Untitled',
          type: asset.type || 'document',
          size: asset.size || (asset.content ? asset.content.length : 0),
          url: asset.url,
          content: asset.content,
          thumbnail: asset.thumbnail,
          createdAt: new Date(),
          updatedAt: new Date(),
          tags: asset.tags || [],
          source: asset.source || 'editor',
          ...asset
        };
        setAssets(prev => [newAsset, ...prev]);
      },
      
      addImageFromUrl: (url: string, name: string, source: AssetFile['source'] = 'ai-generated') => {
        const newAsset: AssetFile = {
          id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
          name,
          type: 'image',
          size: 0, // Unknown size for external URLs
          url,
          thumbnail: url,
          createdAt: new Date(),
          updatedAt: new Date(),
          tags: [source, 'generated'],
          source
        };
        setAssets(prev => [newAsset, ...prev]);
      },
      
      addTextContent: (content: string, name: string, type: 'document' | 'code' = 'document', source: AssetFile['source'] = 'editor') => {
        const newAsset: AssetFile = {
          id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
          name,
          type,
          size: content.length,
          content,
          createdAt: new Date(),
          updatedAt: new Date(),
          tags: [type, source],
          source
        };
        setAssets(prev => [newAsset, ...prev]);
      }
    }));

    // Filter and sort assets
    const filteredAssets = assets
      .filter(asset => {
        if (filterBy !== 'all' && asset.type !== filterBy) return false;
        if (searchTerm) {
          const searchLower = searchTerm.toLowerCase();
          return asset.name.toLowerCase().includes(searchLower) ||
                 asset.tags.some(tag => tag.toLowerCase().includes(searchLower));
        }
        return true;
      })
      .sort((a, b) => {
        switch (sortBy) {
          case 'name':
            return a.name.localeCompare(b.name);
          case 'date':
            return b.updatedAt.getTime() - a.updatedAt.getTime();
          case 'size':
            return b.size - a.size;
          case 'type':
            return a.type.localeCompare(b.type);
          default:
            return 0;
        }
      });

    // File upload handling
    const handleFileUpload = useCallback((files: FileList) => {
      Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
          const newAsset: AssetFile = {
            id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
            name: file.name,
            type: file.type.startsWith('image/') ? 'image' : 
                  file.name.endsWith('.py') ? 'code' :
                  file.name.endsWith('.js') ? 'code' :
                  file.name.endsWith('.ts') ? 'code' :
                  file.name.endsWith('.tsx') ? 'code' :
                  file.name.endsWith('.jsx') ? 'code' : 'document',
            size: file.size,
            url: file.type.startsWith('image/') ? e.target?.result as string : undefined,
            thumbnail: file.type.startsWith('image/') ? e.target?.result as string : undefined,
            content: !file.type.startsWith('image/') ? e.target?.result as string : undefined,
            createdAt: new Date(),
            updatedAt: new Date(),
            tags: [file.type.split('/')[0], 'upload'],
            source: 'upload'
          };
          setAssets(prev => [newAsset, ...prev]);
        };
        
        if (file.type.startsWith('image/')) {
          reader.readAsDataURL(file);
        } else {
          reader.readAsText(file);
        }
      });
    }, []);

    // Drag and drop handlers
    const handleDragOver = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFileUpload(files);
      }
    }, [handleFileUpload]);

    // Asset actions
    const toggleAssetSelection = (assetId: string) => {
      setSelectedAssets(prev => 
        prev.includes(assetId) 
          ? prev.filter(id => id !== assetId)
          : [...prev, assetId]
      );
    };

    const deleteSelectedAssets = () => {
      setAssets(prev => prev.filter(asset => !selectedAssets.includes(asset.id)));
      setSelectedAssets([]);
    };

    const exportSelectedAssets = () => {
      const selectedAssetFiles = assets.filter(asset => selectedAssets.includes(asset.id));
      selectedAssetFiles.forEach(asset => {
        if (asset.type === 'image' && asset.url) {
          // Download image
          const a = document.createElement('a');
          a.href = asset.url;
          a.download = asset.name;
          a.click();
        } else if (asset.content) {
          // Download text content
          const blob = new Blob([asset.content], { type: 'text/plain' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = asset.name;
          a.click();
          URL.revokeObjectURL(url);
        }
      });
    };

    const formatFileSize = (bytes: number) => {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    };

    const getTypeIcon = (type: string) => {
      switch (type) {
        case 'image': return '🖼️';
        case 'document': return '📄';
        case 'code': return '💻';
        case 'data': return '📊';
        default: return '📁';
      }
    };

    const getSourceBadgeColor = (source: string) => {
      switch (source) {
        case 'ai-generated': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
        case 'terminal': return 'bg-green-500/20 text-green-400 border-green-500/30';
        case 'editor': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
        case 'upload': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
        default: return 'bg-muted text-muted-foreground';
      }
    };

    if (!isOpen) return null;

    return (
      <div 
        className="border-t border-border bg-card terminal-section-fade"
        style={{ height: `${height}px` }}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-3 border-b border-border">
            <div className="flex items-center space-x-4">
              <h3 className="font-medium font-mono animate-underline">.\ assets</h3>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono text-muted-foreground">
                  {filteredAssets.length} items
                </span>
                {selectedAssets.length > 0 && (
                  <Badge variant="outline" className="text-xs font-mono">
                    {selectedAssets.length} selected
                  </Badge>
                )}
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              {/* Search */}
              <Input
                placeholder="search assets..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-40 h-7 text-xs font-mono"
              />
              
              {/* Filter */}
              <select
                value={filterBy}
                onChange={(e) => setFilterBy(e.target.value as FilterBy)}
                className="px-2 py-1 text-xs font-mono bg-background border border-border rounded h-7"
              >
                <option value="all">all</option>
                <option value="images">images</option>
                <option value="documents">docs</option>
                <option value="code">code</option>
                <option value="data">data</option>
              </select>
              
              {/* Sort */}
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as SortBy)}
                className="px-2 py-1 text-xs font-mono bg-background border border-border rounded h-7"
              >
                <option value="date">date</option>
                <option value="name">name</option>
                <option value="size">size</option>
                <option value="type">type</option>
              </select>
              
              {/* View Mode */}
              <div className="flex border rounded">
                <Button
                  variant={viewMode === 'grid' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('grid')}
                  className="h-7 px-2 text-xs font-mono"
                >
                  <span className="font-mono text-xs">▦</span>
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('list')}
                  className="h-7 px-2 text-xs font-mono"
                >
                  <span className="font-mono text-xs">☰</span>
                </Button>
              </div>
              
              {/* Actions */}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                className="h-7 px-2 text-xs font-mono warp-section-highlight"
              >
                <span className="font-mono text-xs mr-1">+</span>
                upload
              </Button>
              
              {selectedAssets.length > 0 && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={exportSelectedAssets}
                    className="h-7 px-2 text-xs font-mono warp-section-highlight"
                  >
                    <span className="font-mono text-xs mr-1">↓</span>
                    export
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={deleteSelectedAssets}
                    className="h-7 px-2 text-xs font-mono text-destructive hover:bg-destructive/10"
                  >
                    <span className="font-mono text-xs">×</span>
                  </Button>
                </>
              )}
              
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-7 w-7 p-0 animate-border-grow"
              >
                <span className="font-mono text-xs">×</span>
              </Button>
            </div>
          </div>

          {/* Content */}
          <div 
            className={`flex-1 relative ${dragOver ? 'bg-muted/50' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <ScrollArea className="h-full">
              {filteredAssets.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-32 text-center">
                  <div className="text-2xl mb-2">📂</div>
                  <p className="text-sm font-mono text-muted-foreground">no assets found</p>
                  <p className="text-xs font-mono text-muted-foreground mt-1">
                    drag files here or click upload
                  </p>
                </div>
              ) : viewMode === 'grid' ? (
                <div className="p-3 grid grid-cols-6 gap-3">
                  {filteredAssets.map((asset) => (
                    <div
                      key={asset.id}
                      className={`relative group cursor-pointer border rounded-lg p-2 hover:bg-muted/50 transition-colors warp-section-highlight ${
                        selectedAssets.includes(asset.id) ? 'ring-2 ring-primary' : ''
                      }`}
                      onClick={() => toggleAssetSelection(asset.id)}
                    >
                      <div className="aspect-square mb-2 rounded overflow-hidden bg-muted/30 flex items-center justify-center">
                        {asset.type === 'image' ? (
                          <ImageWithFallback
                            src={asset.thumbnail || asset.url || ''}
                            alt={asset.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <span className="text-2xl">{getTypeIcon(asset.type)}</span>
                        )}
                      </div>
                      
                      <div className="space-y-1">
                        <p className="text-xs font-mono truncate" title={asset.name}>
                          {asset.name}
                        </p>
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-mono text-muted-foreground">
                            {formatFileSize(asset.size)}
                          </span>
                          <Badge 
                            variant="outline" 
                            className={`text-xs px-1 py-0 ${getSourceBadgeColor(asset.source)}`}
                          >
                            {asset.source === 'ai-generated' ? 'ai' : asset.source.slice(0, 2)}
                          </Badge>
                        </div>
                      </div>
                      
                      {selectedAssets.includes(asset.id) && (
                        <div className="absolute top-1 right-1 w-4 h-4 bg-primary rounded-full flex items-center justify-center">
                          <span className="text-xs text-primary-foreground">✓</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-3 space-y-1">
                  {filteredAssets.map((asset) => (
                    <div
                      key={asset.id}
                      className={`flex items-center space-x-3 p-2 rounded hover:bg-muted/50 cursor-pointer warp-section-highlight ${
                        selectedAssets.includes(asset.id) ? 'bg-muted' : ''
                      }`}
                      onClick={() => toggleAssetSelection(asset.id)}
                    >
                      <div className="w-8 h-8 rounded overflow-hidden bg-muted/30 flex items-center justify-center flex-shrink-0">
                        {asset.type === 'image' ? (
                          <ImageWithFallback
                            src={asset.thumbnail || asset.url || ''}
                            alt={asset.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <span className="text-sm">{getTypeIcon(asset.type)}</span>
                        )}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-mono truncate">{asset.name}</p>
                        <div className="flex items-center space-x-2 mt-1">
                          <span className="text-xs font-mono text-muted-foreground">
                            {formatFileSize(asset.size)}
                          </span>
                          <span className="text-xs font-mono text-muted-foreground">•</span>
                          <span className="text-xs font-mono text-muted-foreground">
                            {asset.updatedAt.toLocaleDateString()}
                          </span>
                          <Badge 
                            variant="outline" 
                            className={`text-xs px-1 py-0 ${getSourceBadgeColor(asset.source)}`}
                          >
                            {asset.source}
                          </Badge>
                        </div>
                      </div>
                      
                      {selectedAssets.includes(asset.id) && (
                        <div className="w-4 h-4 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                          <span className="text-xs text-primary-foreground">✓</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>

            {/* Drag overlay */}
            {dragOver && (
              <div className="absolute inset-0 bg-primary/10 border-2 border-dashed border-primary rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <div className="text-3xl mb-2">📁</div>
                  <p className="font-mono text-sm">drop files here</p>
                </div>
              </div>
            )}
          </div>

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) {
                handleFileUpload(e.target.files);
              }
            }}
          />
        </div>
      </div>
    );
  }
);

AssetManager.displayName = 'AssetManager';