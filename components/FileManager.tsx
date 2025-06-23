import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Separator } from './ui/separator';
import { Progress } from './ui/progress';

interface FileItem {
  id: string;
  name: string;
  type: 'file' | 'folder';
  size?: number;
  modified: Date;
  path: string;
  children?: FileItem[];
  expanded?: boolean;
}

interface FileManagerProps {
  className?: string;
}

export function FileManager({ className = "" }: FileManagerProps) {
  const [files, setFiles] = useState<FileItem[]>([
    {
      id: '1',
      name: 'lark-workspace',
      type: 'folder',
      modified: new Date(),
      path: '/',
      expanded: true,
      children: [
        {
          id: '2',
          name: 'src',
          type: 'folder',
          modified: new Date(),
          path: '/src',
          expanded: true,
          children: [
            { id: '3', name: 'main.py', type: 'file', size: 2048, modified: new Date(), path: '/src/main.py' },
            { id: '4', name: 'config.json', type: 'file', size: 512, modified: new Date(), path: '/src/config.json' },
            { id: '5', name: 'utils.py', type: 'file', size: 1024, modified: new Date(), path: '/src/utils.py' }
          ]
        },
        {
          id: '6',
          name: 'docs',
          type: 'folder',
          modified: new Date(),
          path: '/docs',
          expanded: false,
          children: [
            { id: '7', name: 'readme.md', type: 'file', size: 3072, modified: new Date(), path: '/docs/readme.md' },
            { id: '8', name: 'api.md', type: 'file', size: 1536, modified: new Date(), path: '/docs/api.md' }
          ]
        }
      ]
    }
  ]);

  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'tree' | 'list' | 'grid'>('tree');

  const toggleFolder = (fileId: string) => {
    const toggleInTree = (items: FileItem[]): FileItem[] => {
      return items.map(item => {
        if (item.id === fileId) {
          return { ...item, expanded: !item.expanded };
        }
        if (item.children) {
          return { ...item, children: toggleInTree(item.children) };
        }
        return item;
      });
    };

    setFiles(toggleInTree(files));
  };

  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return '-';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const renderTreeView = (items: FileItem[], level: number = 0): React.ReactNode => {
    return items.map(item => (
      <div key={item.id} className="select-none">
        <div
          className={`flex items-center py-1 px-2 text-xs font-mono cursor-pointer hover:bg-accent/50 warp-section-highlight ${
            selectedFile === item.id ? 'bg-accent' : ''
          }`}
          style={{ paddingLeft: `${8 + level * 16}px` }}
          onClick={() => {
            if (item.type === 'folder') {
              toggleFolder(item.id);
            } else {
              setSelectedFile(item.id);
            }
          }}
        >
          {item.type === 'folder' && (
            <span className="mr-2 text-muted-foreground">
              {item.expanded ? '📂' : '📁'}
            </span>
          )}
          {item.type === 'file' && (
            <span className="mr-2 text-muted-foreground">📄</span>
          )}
          <span className="flex-1 animate-underline">{item.name}</span>
          <span className="text-xs text-muted-foreground ml-2">
            {formatFileSize(item.size)}
          </span>
        </div>
        {item.type === 'folder' && item.expanded && item.children && (
          <div className="animate-terminal-fade">
            {renderTreeView(item.children, level + 1)}
          </div>
        )}
      </div>
    ));
  };

  const renderListView = (items: FileItem[]): React.ReactNode => {
    const flattenFiles = (files: FileItem[]): FileItem[] => {
      const result: FileItem[] = [];
      
      const traverse = (items: FileItem[]) => {
        items.forEach(item => {
          result.push(item);
          if (item.children) {
            traverse(item.children);
          }
        });
      };
      
      traverse(files);
      return result;
    };

    const flatFiles = flattenFiles(items);

    return (
      <div className="space-y-1">
        {flatFiles.map(item => (
          <div
            key={item.id}
            className={`flex items-center justify-between p-2 text-xs font-mono cursor-pointer hover:bg-accent/50 warp-section-highlight rounded ${
              selectedFile === item.id ? 'bg-accent' : ''
            }`}
            onClick={() => setSelectedFile(item.id)}
          >
            <div className="flex items-center space-x-2">
              <span className="text-muted-foreground">
                {item.type === 'folder' ? '📁' : '📄'}
              </span>
              <span className="animate-underline">{item.name}</span>
              <Badge variant="outline" className="text-xs">
                {item.type}
              </Badge>
            </div>
            <div className="flex items-center space-x-2 text-muted-foreground">
              <span>{formatFileSize(item.size)}</span>
              <span>{item.modified.toLocaleDateString()}</span>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className={`space-y-6 font-mono ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary/10 rounded flex items-center justify-center">
            <span className="font-mono text-xs text-primary">fl</span>
          </div>
          <div>
            <h3 className="font-medium">File Manager</h3>
            <p className="text-sm text-muted-foreground">Project files and directories</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Tabs value={viewMode} onValueChange={(value: any) => setViewMode(value)} className="flex">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="tree" className="font-mono text-xs">
                <span className="mr-1">tr</span>
                Tree
              </TabsTrigger>
              <TabsTrigger value="list" className="font-mono text-xs">
                <span className="mr-1">ls</span>
                List
              </TabsTrigger>
              <TabsTrigger value="grid" className="font-mono text-xs">
                <span className="mr-1">gr</span>
                Grid
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="border-border/50">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <span className="font-mono text-xs text-blue-500">fl</span>
              <div>
                <div className="font-mono text-sm">Files</div>
                <div className="font-mono text-lg font-medium">12</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-border/50">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <span className="font-mono text-xs text-green-500">fd</span>
              <div>
                <div className="font-mono text-sm">Folders</div>
                <div className="font-mono text-lg font-medium">4</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-border/50">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <span className="font-mono text-xs text-purple-500">sz</span>
              <div>
                <div className="font-mono text-sm">Size</div>
                <div className="font-mono text-lg font-medium">2.4 MB</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* File Browser */}
      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="font-mono text-sm flex items-center space-x-2">
            <span className="text-primary">br</span>
            <span>File Browser</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search */}
          <div className="flex items-center space-x-2">
            <Input
              placeholder="Search files..."
              className="flex-1 font-mono text-xs"
            />
            <Button size="sm" className="font-mono text-xs">
              <span className="mr-1">sr</span>
              Search
            </Button>
          </div>

          <Separator />

          {/* File View */}
          <div className="max-h-96 overflow-y-auto subtle-scrollbar">
            {viewMode === 'tree' && renderTreeView(files)}
            {viewMode === 'list' && renderListView(files)}
            {viewMode === 'grid' && (
              <div className="grid grid-cols-4 gap-4">
                {files.map(item => (
                  <Card
                    key={item.id}
                    className={`cursor-pointer hover:bg-accent/50 warp-section-highlight ${
                      selectedFile === item.id ? 'bg-accent' : ''
                    }`}
                    onClick={() => setSelectedFile(item.id)}
                  >
                    <CardContent className="p-4 text-center">
                      <div className="text-2xl mb-2">
                        {item.type === 'folder' ? '📁' : '📄'}
                      </div>
                      <div className="font-mono text-xs truncate">{item.name}</div>
                      <div className="font-mono text-xs text-muted-foreground">
                        {formatFileSize(item.size)}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* File Actions */}
      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="font-mono text-sm flex items-center space-x-2">
            <span className="text-orange-500">ac</span>
            <span>Actions</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-2">
            <Button variant="outline" size="sm" className="font-mono text-xs warp-section-highlight">
              <span className="mr-1">nw</span>
              New
            </Button>
            <Button variant="outline" size="sm" className="font-mono text-xs warp-section-highlight">
              <span className="mr-1">up</span>
              Upload
            </Button>
            <Button variant="outline" size="sm" className="font-mono text-xs warp-section-highlight">
              <span className="mr-1">dl</span>
              Download
            </Button>
            <Button variant="outline" size="sm" className="font-mono text-xs warp-section-highlight">
              <span className="mr-1">rm</span>
              Delete
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}