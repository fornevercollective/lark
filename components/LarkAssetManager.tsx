import { useState, useRef, useCallback } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Progress } from './ui/progress';

// Asset Management Types
export interface LarkAsset {
  id: string;
  name: string;
  type: 'image' | 'video' | 'audio' | 'document' | 'other';
  url: string;
  thumbnail?: string;
  size: number;
  dimensions?: { width: number; height: number };
  createdAt: Date;
  lastUsed?: Date;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface LarkAssetManagerProps {
  // Event handlers
  onAssetSelect?: (asset: LarkAsset) => void;
  onAssetDelete?: (assetId: string) => void;
  onAssetUpload?: (assets: LarkAsset[]) => void;
  onAssetsChange?: (assets: LarkAsset[]) => void;
  
  // Configuration
  selectedAssets?: string[];
  maxFileSize?: number; // in MB
  acceptedTypes?: string[];
  height?: string;
  
  // UI Options
  showSearch?: boolean;
  showFilters?: boolean;
  showDetails?: boolean;
  showBulkActions?: boolean;
  enableDragDrop?: boolean;
  
  // Initial data
  initialAssets?: LarkAsset[];
}

export function LarkAssetManager({
  onAssetSelect,
  onAssetDelete,
  onAssetUpload,
  onAssetsChange,
  selectedAssets = [],
  maxFileSize = 50,
  acceptedTypes = ['image/*', 'video/*', 'audio/*', '.pdf'],
  height = 'h-full',
  showSearch = true,
  showFilters = true,
  showDetails = true,
  showBulkActions = true,
  enableDragDrop = true,
  initialAssets = []
}: LarkAssetManagerProps) {
  // State management
  const [assets, setAssets] = useState<LarkAsset[]>(initialAssets);
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>(selectedAssets);
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'size' | 'type'>('date');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [dragActive, setDragActive] = useState(false);
  const [selectedAssetForDetails, setSelectedAssetForDetails] = useState<LarkAsset | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Asset management functions
  const updateAssets = (newAssets: LarkAsset[]) => {
    setAssets(newAssets);
    onAssetsChange?.(newAssets);
  };

  const addAssets = (newAssets: LarkAsset[]) => {
    const updatedAssets = [...assets, ...newAssets];
    updateAssets(updatedAssets);
    onAssetUpload?.(newAssets);
  };

  const removeAsset = (assetId: string) => {
    const updatedAssets = assets.filter(asset => asset.id !== assetId);
    updateAssets(updatedAssets);
    setSelectedAssetIds(prev => prev.filter(id => id !== assetId));
    onAssetDelete?.(assetId);
  };

  // File processing
  const processFile = async (file: File): Promise<LarkAsset> => {
    const asset: LarkAsset = {
      id: `asset-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name: file.name,
      type: getFileType(file),
      url: URL.createObjectURL(file),
      size: file.size,
      createdAt: new Date(),
      metadata: {
        originalFile: file.name,
        mimeType: file.type
      }
    };

    // Generate thumbnail for images
    if (asset.type === 'image') {
      asset.thumbnail = asset.url;
      
      // Get image dimensions
      try {
        const dimensions = await getImageDimensions(asset.url);
        asset.dimensions = dimensions;
      } catch (error) {
        console.warn('Failed to get image dimensions:', error);
      }
    }

    return asset;
  };

  const getImageDimensions = (url: string): Promise<{ width: number; height: number }> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.width, height: img.height });
      img.onerror = reject;
      img.src = url;
    });
  };

  const getFileType = (file: File): LarkAsset['type'] => {
    if (file.type.startsWith('image/')) return 'image';
    if (file.type.startsWith('video/')) return 'video';
    if (file.type.startsWith('audio/')) return 'audio';
    if (file.type === 'application/pdf' || file.name.match(/\.(doc|docx)$/i)) return 'document';
    return 'other';
  };

  // File upload handling
  const handleFileUpload = useCallback(async (files: FileList) => {
    const validFiles = Array.from(files).filter(file => {
      if (file.size > maxFileSize * 1024 * 1024) {
        console.warn(`File ${file.name} exceeds size limit (${maxFileSize}MB)`);
        return false;
      }
      return true;
    });

    const newAssets: LarkAsset[] = [];

    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i];
      const progressId = `upload-${Date.now()}-${i}`;
      
      // Simulate upload progress
      setUploadProgress(prev => ({ ...prev, [progressId]: 0 }));
      
      for (let progress = 0; progress <= 100; progress += 20) {
        await new Promise(resolve => setTimeout(resolve, 100));
        setUploadProgress(prev => ({ ...prev, [progressId]: progress }));
      }

      const asset = await processFile(file);
      newAssets.push(asset);

      // Remove progress tracking
      setUploadProgress(prev => {
        const { [progressId]: removed, ...rest } = prev;
        return rest;
      });
    }

    addAssets(newAssets);
  }, [maxFileSize, assets]);

  // Utility functions
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getFileTypeAbbr = (type: LarkAsset['type']): string => {
    switch (type) {
      case 'image': return 'img';
      case 'video': return 'vid';
      case 'audio': return 'aud';
      case 'document': return 'doc';
      default: return 'fil';
    }
  };

  // Filter and sort assets
  const filteredAssets = assets
    .filter(asset => {
      const matchesSearch = asset.name.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesType = typeFilter === 'all' || asset.type === typeFilter;
      return matchesSearch && matchesType;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name': return a.name.localeCompare(b.name);
        case 'size': return b.size - a.size;
        case 'type': return a.type.localeCompare(b.type);
        case 'date':
        default: return b.createdAt.getTime() - a.createdAt.getTime();
      }
    });

  // Selection handlers
  const toggleSelection = (assetId: string) => {
    const newSelection = selectedAssetIds.includes(assetId)
      ? selectedAssetIds.filter(id => id !== assetId)
      : [...selectedAssetIds, assetId];
    
    setSelectedAssetIds(newSelection);
  };

  const selectAll = () => setSelectedAssetIds(filteredAssets.map(a => a.id));
  const clearSelection = () => setSelectedAssetIds([]);

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (enableDragDrop) setDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    
    if (enableDragDrop && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  return (
    <div className={`${height} flex flex-col font-mono bg-card`}>
      {/* Header Controls */}
      <div className="p-3 border-b border-border space-y-2">
        {/* Search and Filters */}
        {(showSearch || showFilters) && (
          <div className="flex items-center gap-2">
            {showSearch && (
              <Input
                placeholder="search assets..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="flex-1 h-7 font-mono text-xs"
              />
            )}
            
            {showFilters && (
              <>
                <Select value={typeFilter} onValueChange={setTypeFilter}>
                  <SelectTrigger className="w-20 h-7 font-mono text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">all</SelectItem>
                    <SelectItem value="image">img</SelectItem>
                    <SelectItem value="video">vid</SelectItem>
                    <SelectItem value="audio">aud</SelectItem>
                    <SelectItem value="document">doc</SelectItem>
                  </SelectContent>
                </Select>

                <Select value={sortBy} onValueChange={(v: any) => setSortBy(v)}>
                  <SelectTrigger className="w-16 h-7 font-mono text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="date">date</SelectItem>
                    <SelectItem value="name">name</SelectItem>
                    <SelectItem value="size">size</SelectItem>
                    <SelectItem value="type">type</SelectItem>
                  </SelectContent>
                </Select>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
                  className="h-7 px-2 font-mono text-xs animate-accent-line"
                >
                  {viewMode === 'grid' ? 'ls' : 'gr'}
                </Button>
              </>
            )}
          </div>
        )}

        {/* Action Bar */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="font-mono text-xs h-5">
              {filteredAssets.length} assets
            </Badge>
            
            {selectedAssetIds.length > 0 && (
              <Badge variant="outline" className="font-mono text-xs h-5">
                {selectedAssetIds.length} selected
              </Badge>
            )}
          </div>

          {showBulkActions && (
            <div className="flex items-center gap-1">
              {selectedAssetIds.length > 0 && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => selectedAssetIds.forEach(removeAsset)}
                    className="h-6 px-2 font-mono text-xs animate-accent-line"
                  >
                    rm
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearSelection}
                    className="h-6 px-2 font-mono text-xs animate-accent-line"
                  >
                    cl
                  </Button>
                </>
              )}
              
              <Button
                variant="ghost"
                size="sm"
                onClick={selectedAssetIds.length === filteredAssets.length ? clearSelection : selectAll}
                className="h-6 px-2 font-mono text-xs animate-accent-line"
              >
                sa
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Upload Progress */}
      {Object.keys(uploadProgress).length > 0 && (
        <div className="p-2 border-b border-border space-y-1">
          {Object.entries(uploadProgress).map(([id, progress]) => (
            <div key={id} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">uploading...</span>
                <span className="text-muted-foreground">{progress}%</span>
              </div>
              <Progress value={progress} className="h-1" />
            </div>
          ))}
        </div>
      )}

      {/* Main Content */}
      <div 
        className="flex-1 overflow-y-auto p-3"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {filteredAssets.length === 0 ? (
          /* Empty State */
          <div 
            className={`
              h-full border-2 border-dashed rounded-lg flex flex-col items-center justify-center
              cursor-pointer transition-colors animate-border-grow
              ${dragActive ? 'border-foreground bg-muted/20' : 'border-border'}
            `}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="text-center space-y-2">
              <div className="font-mono text-2xl text-muted-foreground">+</div>
              <p className="font-mono text-xs text-muted-foreground">
                {enableDragDrop ? 'drag & drop files or click to upload' : 'click to upload files'}
              </p>
              <p className="font-mono text-xs text-muted-foreground opacity-70">
                max {maxFileSize}MB
              </p>
            </div>
          </div>
        ) : (
          /* Asset Grid/List */
          <div className={
            viewMode === 'grid' 
              ? 'grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2'
              : 'space-y-1'
          }>
            {/* Upload Button */}
            {viewMode === 'grid' && (
              <div
                className="aspect-square border-2 border-dashed border-border rounded-lg flex items-center justify-center cursor-pointer animate-border-grow"
                onClick={() => fileInputRef.current?.click()}
              >
                <span className="font-mono text-xs text-muted-foreground">+</span>
              </div>
            )}

            {/* Assets */}
            {filteredAssets.map((asset) => (
              <div
                key={asset.id}
                className={`
                  group relative cursor-pointer transition-all animate-border-grow
                  ${selectedAssetIds.includes(asset.id) ? 'ring-1 ring-foreground' : ''}
                  ${viewMode === 'grid' ? 'aspect-square' : 'flex items-center p-2 rounded-lg hover:bg-muted/50'}
                `}
                onClick={() => {
                  toggleSelection(asset.id);
                  onAssetSelect?.(asset);
                }}
                onDoubleClick={() => showDetails && setSelectedAssetForDetails(asset)}
              >
                {viewMode === 'grid' ? (
                  /* Grid Item */
                  <>
                    <div className="w-full h-full bg-muted rounded-lg overflow-hidden">
                      {asset.type === 'image' && asset.thumbnail ? (
                        <img
                          src={asset.thumbnail}
                          alt={asset.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <span className="font-mono text-xs text-muted-foreground">
                            {getFileTypeAbbr(asset.type)}
                          </span>
                        </div>
                      )}
                    </div>
                    
                    <div className="absolute inset-0 bg-background/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-end">
                      <div className="p-1 w-full">
                        <p className="font-mono text-xs truncate" title={asset.name}>
                          {asset.name}
                        </p>
                        <p className="font-mono text-xs text-muted-foreground">
                          {formatFileSize(asset.size)}
                        </p>
                      </div>
                    </div>

                    {selectedAssetIds.includes(asset.id) && (
                      <div className="absolute top-1 right-1 w-3 h-3 bg-foreground text-background rounded-full flex items-center justify-center">
                        <span className="font-mono text-xs">✓</span>
                      </div>
                    )}

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeAsset(asset.id);
                      }}
                      className="absolute top-1 left-1 h-3 w-3 p-0 opacity-0 group-hover:opacity-100 font-mono text-xs bg-background/80"
                    >
                      ×
                    </Button>
                  </>
                ) : (
                  /* List Item */
                  <div className="flex items-center gap-2 flex-1">
                    <div className="w-6 h-6 bg-muted rounded flex items-center justify-center">
                      <span className="font-mono text-xs text-muted-foreground">
                        {getFileTypeAbbr(asset.type)}
                      </span>
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <p className="font-mono text-xs truncate">{asset.name}</p>
                      <p className="font-mono text-xs text-muted-foreground">
                        {formatFileSize(asset.size)}
                      </p>
                    </div>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeAsset(asset.id);
                      }}
                      className="h-4 w-4 p-0 font-mono text-xs"
                    >
                      ×
                    </Button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Asset Details Modal */}
      {showDetails && selectedAssetForDetails && (
        <Dialog open={!!selectedAssetForDetails} onOpenChange={() => setSelectedAssetForDetails(null)}>
          <DialogContent className="font-mono">
            <DialogHeader>
              <DialogTitle className="font-mono">asset details</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-3">
              {selectedAssetForDetails.type === 'image' && (
                <div className="aspect-video bg-muted rounded-lg overflow-hidden">
                  <img
                    src={selectedAssetForDetails.url}
                    alt={selectedAssetForDetails.name}
                    className="w-full h-full object-contain"
                  />
                </div>
              )}
              
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <p className="text-muted-foreground">name</p>
                  <p className="truncate">{selectedAssetForDetails.name}</p>
                </div>
                
                <div>
                  <p className="text-muted-foreground">type</p>
                  <p>{selectedAssetForDetails.type}</p>
                </div>
                
                <div>
                  <p className="text-muted-foreground">size</p>
                  <p>{formatFileSize(selectedAssetForDetails.size)}</p>
                </div>
                
                <div>
                  <p className="text-muted-foreground">created</p>
                  <p>{selectedAssetForDetails.createdAt.toLocaleDateString()}</p>
                </div>
                
                {selectedAssetForDetails.dimensions && (
                  <div className="col-span-2">
                    <p className="text-muted-foreground">dimensions</p>
                    <p>{selectedAssetForDetails.dimensions.width} × {selectedAssetForDetails.dimensions.height}</p>
                  </div>
                )}
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={acceptedTypes.join(',')}
        onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
        className="hidden"
      />
    </div>
  );
}

// Export utility types for external use
export type { LarkAsset, LarkAssetManagerProps };