import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Switch } from './ui/switch';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import { DeepSearch } from './DeepSearch';

interface TerminalLine {
  id: number;
  content: string;
  type: 'command' | 'output' | 'success' | 'error' | 'python' | 'ai' | 'codex';
}

interface TerminalProps {
  onModeChange: (mode: string) => void;
}

interface BookmarkEntry {
  id: string;
  url: string;
  title: string;
  tags: string[];
  project?: string;
  createdAt: Date;
}

export function Terminal({ onModeChange }: TerminalProps) {
  const [lines, setLines] = useState<TerminalLine[]>([
    { id: 0, content: '.\ lark terminal ready', type: 'output' },
    { id: 1, content: 'type "help" for available commands', type: 'output' },
    { id: 2, content: 'type "cx help" for codex bookmark commands', type: 'output' }
  ]);
  const [input, setInput] = useState('');
  const [isMultiLine, setIsMultiLine] = useState(false);
  const [multiLineBuffer, setMultiLineBuffer] = useState('');
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [activeToolset, setActiveToolset] = useState('quick');
  const [activeSearchTab, setActiveSearchTab] = useState('dictionaries');
  const [toolStates, setToolStates] = useState({
    autoSave: true,
    darkMode: false,
    notifications: true,
    debugMode: false
  });
  
  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const addLine = (content: string, type: TerminalLine['type'] = 'output') => {
    setLines(prev => [...prev, { 
      id: Date.now() + Math.random(), 
      content, 
      type 
    }]);
  };

  const getLineColor = (type: TerminalLine['type']) => {
    switch (type) {
      case 'command': return 'text-foreground';
      case 'success': return 'text-green-600';
      case 'error': return 'text-red-600';
      case 'python': return 'text-blue-600';
      case 'ai': return 'text-purple-600';
      case 'codex': return 'text-orange-600';
      default: return 'text-muted-foreground';
    }
  };

  // Tool execution functions
  const executeTool = (toolName: string) => {
    addLine(`❯ executing ${toolName}...`, 'command');
    
    switch (toolName) {
      case 'save':
        addLine('✓ project saved successfully', 'success');
        break;
      case 'build':
        addLine('.\ building project...', 'output');
        setTimeout(() => addLine('✓ build completed', 'success'), 1000);
        break;
      case 'test':
        addLine('.\ running tests...', 'output');
        setTimeout(() => addLine('✓ all tests passed (12/12)', 'success'), 1500);
        break;
      case 'deploy':
        addLine('.\ deploying to production...', 'output');
        setTimeout(() => addLine('✓ deployed successfully', 'success'), 2000);
        break;
      case 'format':
        addLine('✓ code formatted', 'success');
        break;
      case 'lint':
        addLine('.\ linting codebase...', 'output');
        setTimeout(() => addLine('✓ no lint errors found', 'success'), 800);
        break;
      default:
        addLine(`✓ ${toolName} executed`, 'success');
    }
  };

  const toggleToolState = (key: keyof typeof toolStates) => {
    setToolStates(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
    addLine(`✓ ${key} ${toolStates[key] ? 'disabled' : 'enabled'}`, 'success');
  };

  // Codex bookmark management functions
  const getBookmarks = (): BookmarkEntry[] => {
    const saved = localStorage.getItem('codex-bookmarks');
    return saved ? JSON.parse(saved) : [];
  };

  const saveBookmarks = (bookmarks: BookmarkEntry[]) => {
    localStorage.setItem('codex-bookmarks', JSON.stringify(bookmarks));
  };

  const processCodexCommand = (args: string[]) => {
    const subcommand = args[1]?.toLowerCase();
    
    switch (subcommand) {
      case 'help':
        addLine('.\ codex commands:', 'codex');
        addLine('  cx add <url> [--tags "tag1,tag2"] [--project "name"]', 'output');
        addLine('  cx list [--project "name"] [--tag "tag"]', 'output');
        addLine('  cx search <query>', 'output');
        addLine('  cx projects', 'output');
        addLine('  cx tags', 'output');
        addLine('  cx open <id>', 'output');
        addLine('  cx remove <id>', 'output');
        addLine('  cx gui - open bookmark manager', 'output');
        break;

      case 'add':
        if (args.length < 3) {
          addLine('.\ usage: cx add <url> [--tags "tag1,tag2"] [--project "name"]', 'error');
          return;
        }
        
        const url = args[2];
        let tags: string[] = [];
        let project: string | undefined;

        // Parse tags and project from arguments
        for (let i = 3; i < args.length; i++) {
          if (args[i] === '--tags' && args[i + 1]) {
            tags = args[i + 1].replace(/"/g, '').split(',').map(t => t.trim());
            i++; // skip next arg as it was consumed
          } else if (args[i] === '--project' && args[i + 1]) {
            project = args[i + 1].replace(/"/g, '');
            i++; // skip next arg as it was consumed
          }
        }

        try {
          const domain = new URL(url).hostname;
          const newBookmark: BookmarkEntry = {
            id: `bm${Date.now()}`,
            url,
            title: `Page from ${domain}`,
            tags,
            project,
            createdAt: new Date()
          };

          const bookmarks = getBookmarks();
          bookmarks.unshift(newBookmark);
          saveBookmarks(bookmarks);

          addLine(`.\ processing "${url}"...`, 'codex');
          addLine(`✓ archived as ${newBookmark.id}`, 'success');
          addLine(`  tags: [${tags.join(', ')}]`, 'output');
          if (project) {
            addLine(`  project: ${project}`, 'output');
          }
        } catch (e) {
          addLine(`.\ invalid url: ${url}`, 'error');
        }
        break;

      case 'list':
        const bookmarks = getBookmarks();
        let filteredBookmarks = bookmarks;

        // Filter by project or tag if specified
        const projectFilter = args.find((arg, i) => args[i-1] === '--project')?.replace(/"/g, '');
        const tagFilter = args.find((arg, i) => args[i-1] === '--tag')?.replace(/"/g, '');

        if (projectFilter) {
          filteredBookmarks = filteredBookmarks.filter(bm => bm.project === projectFilter);
        }
        if (tagFilter) {
          filteredBookmarks = filteredBookmarks.filter(bm => bm.tags.includes(tagFilter));
        }

        if (filteredBookmarks.length === 0) {
          addLine('.\ no bookmarks found', 'output');
        } else {
          addLine(`.\ bookmarks (${filteredBookmarks.length}):`, 'codex');
          filteredBookmarks.slice(0, 10).forEach(bm => {
            const date = bm.createdAt.toLocaleDateString();
            const tags = bm.tags.length > 0 ? ` [${bm.tags.join(', ')}]` : '';
            const project = bm.project ? ` (${bm.project})` : '';
            addLine(`  ${bm.id}: ${bm.title}${tags}${project} - ${date}`, 'output');
          });
          if (filteredBookmarks.length > 10) {
            addLine(`  ... and ${filteredBookmarks.length - 10} more`, 'output');
          }
        }
        break;

      case 'search':
        if (args.length < 3) {
          addLine('.\ usage: cx search <query>', 'error');
          return;
        }
        
        const query = args.slice(2).join(' ').toLowerCase();
        const allBookmarks = getBookmarks();
        const matches = allBookmarks.filter(bm => 
          bm.title.toLowerCase().includes(query) ||
          bm.url.toLowerCase().includes(query) ||
          bm.tags.some(tag => tag.toLowerCase().includes(query))
        );

        if (matches.length === 0) {
          addLine(`.\ no matches for "${query}"`, 'output');
        } else {
          addLine(`.\ search results for "${query}" (${matches.length}):`, 'codex');
          matches.slice(0, 5).forEach(bm => {
            addLine(`  ${bm.id}: ${bm.title}`, 'output');
          });
        }
        break;

      case 'projects':
        const projectBookmarks = getBookmarks();
        const projects = [...new Set(projectBookmarks.map(bm => bm.project).filter(Boolean))];
        
        if (projects.length === 0) {
          addLine('.\ no projects found', 'output');
        } else {
          addLine(`.\ projects (${projects.length}):`, 'codex');
          projects.forEach(project => {
            const count = projectBookmarks.filter(bm => bm.project === project).length;
            addLine(`  ${project} (${count} bookmarks)`, 'output');
          });
        }
        break;

      case 'tags':
        const tagBookmarks = getBookmarks();
        const allTags = [...new Set(tagBookmarks.flatMap(bm => bm.tags))].sort();
        
        if (allTags.length === 0) {
          addLine('.\ no tags found', 'output');
        } else {
          addLine(`.\ tags (${allTags.length}):`, 'codex');
          allTags.forEach(tag => {
            const count = tagBookmarks.filter(bm => bm.tags.includes(tag)).length;
            addLine(`  ${tag} (${count})`, 'output');
          });
        }
        break;

      case 'remove':
        if (args.length < 3) {
          addLine('.\ usage: cx remove <id>', 'error');
          return;
        }
        
        const removeId = args[2];
        const currentBookmarks = getBookmarks();
        const bookmarkToRemove = currentBookmarks.find(bm => bm.id === removeId);
        
        if (!bookmarkToRemove) {
          addLine(`.\ bookmark ${removeId} not found`, 'error');
        } else {
          const updatedBookmarks = currentBookmarks.filter(bm => bm.id !== removeId);
          saveBookmarks(updatedBookmarks);
          addLine(`.\ removed "${bookmarkToRemove.title}"`, 'success');
        }
        break;

      case 'open':
        if (args.length < 3) {
          addLine('.\ usage: cx open <id>', 'error');
          return;
        }
        
        const openId = args[2];
        const openBookmarks = getBookmarks();
        const bookmarkToOpen = openBookmarks.find(bm => bm.id === openId);
        
        if (!bookmarkToOpen) {
          addLine(`.\ bookmark ${openId} not found`, 'error');
        } else {
          addLine(`.\ opening "${bookmarkToOpen.title}"`, 'codex');
          addLine(`  url: ${bookmarkToOpen.url}`, 'output');
          // In a real implementation, this would open the URL
        }
        break;

      case 'gui':
        addLine('.\ opening codex gui...', 'codex');
        setTimeout(() => {
          onModeChange('codex');
        }, 500);
        break;

      default:
        addLine('.\ unknown codex command', 'error');
        addLine('  type "cx help" for available commands', 'output');
    }
  };

  const processCommand = (cmd: string) => {
    if (!cmd.trim()) return;

    addLine(`❯ ${cmd}`, 'command');
    setHistory(prev => [cmd, ...prev.slice(0, 49)]); // Keep last 50 commands
    setHistoryIndex(-1);

    const args = cmd.trim().split(' ');
    const command = args[0].toLowerCase();

    switch (command) {
      case 'help':
        addLine('.\ available commands:', 'output');
        addLine('  help - show this help', 'output');
        addLine('  clear - clear terminal', 'output');
        addLine('  python/py - enter python mode', 'output');
        addLine('  ai <message> - query ai assistant', 'output');
        addLine('  ls - list files', 'output');
        addLine('  pwd - show current directory', 'output');
        addLine('  echo <text> - print text', 'output');
        addLine('  date - show current date', 'output');
        addLine('  cx <command> - codex bookmark commands', 'codex');
        addLine('  search - access deep search toolset', 'output');
        addLine('  editor - switch to editor mode', 'output');
        addLine('  files - switch to files mode', 'output');
        break;

      case 'clear':
        setLines([]);
        break;

      case 'search':
        addLine('.\ opening deep search toolset...', 'output');
        setActiveToolset('search');
        break;

      case 'cx':
        processCodexCommand(args);
        break;

      case 'python':
      case 'py':
        setIsMultiLine(true);
        addLine('.\ python mode activated', 'python');
        addLine('  type your code and press ctrl+enter to execute', 'output');
        addLine('  press esc to exit python mode', 'output');
        break;

      case 'ai':
        if (args.length < 2) {
          addLine('.\ usage: ai <message>', 'error');
        } else {
          const message = args.slice(1).join(' ');
          addLine(`.\ processing: "${message}"`, 'ai');
          
          setTimeout(() => {
            let response = '';
            if (message.includes('bookmark') || message.includes('codex')) {
              response = 'the codex system helps organize knowledge\n\nuse "cx add <url>" to bookmark pages\nuse "cx list" to see your bookmarks\nuse "cx gui" to open the visual manager';
            } else if (message.includes('pytorch') || message.includes('torch')) {
              response = 'pytorch is a deep learning framework\n\nquick example:\nimport torch\nx = torch.randn(3, 3)\nprint(x)';
            } else {
              response = `regarding "${message}":\n\nthis is a simulated ai response\nin production, this would connect to a language model`;
            }
            
            addLine(`.\ ${response}`, 'ai');
          }, 1000);
        }
        break;

      case 'ls':
        addLine('.\ files:', 'output');
        addLine('  welcome.md', 'output');
        addLine('  notes/', 'output');
        addLine('  projects/', 'output');
        addLine('  codex-vault/', 'output');
        break;

      case 'pwd':
        addLine('/home/user/lark', 'output');
        break;

      case 'echo':
        if (args.length > 1) {
          addLine(args.slice(1).join(' '), 'output');
        }
        break;

      case 'date':
        addLine(new Date().toString(), 'output');
        break;

      case 'editor':
        addLine('.\ switching to editor mode...', 'output');
        setTimeout(() => onModeChange('editor'), 500);
        break;

      case 'files':
        addLine('.\ switching to files mode...', 'output');
        setTimeout(() => onModeChange('files'), 500);
        break;

      default:
        addLine(`.\ command not found: ${command}`, 'error');
        addLine('  type "help" for available commands', 'output');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (isMultiLine) {
      if (e.key === 'Escape') {
        setIsMultiLine(false);
        setMultiLineBuffer('');
        addLine('.\ exited python mode', 'output');
        return;
      }
      
      if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        if (multiLineBuffer.trim()) {
          addLine(`>>> ${multiLineBuffer}`, 'python');
          
          // Simulate python execution
          setTimeout(() => {
            try {
              if (multiLineBuffer.includes('print')) {
                const match = multiLineBuffer.match(/print\(['"](.+?)['"]\)/);
                if (match) {
                  addLine(match[1], 'output');
                } else {
                  addLine('Hello from Python!', 'output');
                }
              } else if (multiLineBuffer.includes('=')) {
                addLine('.\ variable assigned', 'output');
              } else {
                addLine('.\ code executed', 'output');
              }
            } catch {
              addLine('.\ python execution simulated', 'output');
            }
          }, 300);
        }
        setMultiLineBuffer('');
        return;
      }
      
      if (e.key === 'Enter') {
        e.preventDefault();
        setMultiLineBuffer(prev => prev + '\n');
        return;
      }
    } else {
      if (e.key === 'Enter') {
        e.preventDefault();
        processCommand(input);
        setInput('');
        return;
      }

      if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (historyIndex < history.length - 1) {
          const newIndex = historyIndex + 1;
          setHistoryIndex(newIndex);
          setInput(history[newIndex]);
        }
        return;
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (historyIndex > 0) {
          const newIndex = historyIndex - 1;
          setHistoryIndex(newIndex);
          setInput(history[newIndex]);
        } else if (historyIndex === 0) {
          setHistoryIndex(-1);
          setInput('');
        }
        return;
      }
    }
  };

  return (
    <div className="h-screen bg-background text-foreground flex flex-col font-mono">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center space-x-3">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="ml-4 text-sm animate-underline">.\ lark terminal</span>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="text-xs">
            link
          </Badge>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => onModeChange('editor')}
            className="text-xs px-2 animate-accent-line"
          >
            back
          </Button>
        </div>
      </div>

      {/* Toolset Section */}
      <div className="h-[50vh] border-b border-border bg-card/50 p-4">
        <div className="h-full border border-border/50 rounded bg-background/50 p-3 font-mono text-sm">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-border/30">
            <span className="text-xs text-muted-foreground animate-underline">.\ development toolset</span>
            <div className="flex items-center space-x-2 text-xs text-muted-foreground">
              <span>v1.0</span>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            </div>
          </div>
          
          <div className="h-full overflow-y-auto subtle-scrollbar">
            <Tabs value={activeToolset} onValueChange={setActiveToolset} className="h-full">
              <TabsList className="grid w-full grid-cols-5 mb-4">
                <TabsTrigger value="quick" className="font-mono text-xs">
                  <span className="mr-1">qt</span>
                  Quick
                </TabsTrigger>
                <TabsTrigger value="build" className="font-mono text-xs">
                  <span className="mr-1">bd</span>
                  Build
                </TabsTrigger>
                <TabsTrigger value="debug" className="font-mono text-xs">
                  <span className="mr-1">db</span>
                  Debug
                </TabsTrigger>
                <TabsTrigger value="config" className="font-mono text-xs">
                  <span className="mr-1">cf</span>
                  Config
                </TabsTrigger>
                <TabsTrigger value="search" className="font-mono text-xs">
                  <span className="mr-1">sr</span>
                  Search
                </TabsTrigger>
              </TabsList>

              <TabsContent value="quick" className="space-y-4 mt-0">
                <Card className="border-border/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="font-mono text-sm flex items-center gap-2">
                      <span className="text-primary">qt</span>
                      Quick Actions
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-4 gap-2">
                      {[
                        { abbr: 'sv', label: 'Save', action: 'save' },
                        { abbr: 'rf', label: 'Refresh', action: 'refresh' },
                        { abbr: 'fm', label: 'Format', action: 'format' },
                        { abbr: 'ln', label: 'Lint', action: 'lint' }
                      ].map((tool) => (
                        <button
                          key={tool.abbr}
                          onClick={() => executeTool(tool.action)}
                          className="flex flex-col items-center p-2 bg-muted/50 rounded border hover:bg-accent/50 warp-section-highlight transition-colors"
                        >
                          <div className="text-xs font-mono mb-1">{tool.abbr}</div>
                          <div className="text-xs text-muted-foreground">{tool.label}</div>
                        </button>
                      ))}
                    </div>
                    
                    <Separator />
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs">Project Status</span>
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <span className="font-mono text-xs text-muted-foreground">healthy</span>
                        </div>
                      </div>
                      <Progress value={85} className="h-1" />
                      <div className="text-xs text-muted-foreground font-mono">85% complete</div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="build" className="space-y-4 mt-0">
                <Card className="border-border/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="font-mono text-sm flex items-center gap-2">
                      <span className="text-blue-500">bd</span>
                      Build Tools
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="font-mono warp-section-highlight"
                        onClick={() => executeTool('build')}
                      >
                        <span className="mr-2">bd</span>
                        Build
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="font-mono warp-section-highlight"
                        onClick={() => executeTool('test')}
                      >
                        <span className="mr-2">ts</span>
                        Test
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="font-mono warp-section-highlight"
                        onClick={() => executeTool('deploy')}
                      >
                        <span className="mr-2">dp</span>
                        Deploy
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="font-mono warp-section-highlight"
                        onClick={() => executeTool('watch')}
                      >
                        <span className="mr-2">wt</span>
                        Watch
                      </Button>
                    </div>
                    
                    <Separator />
                    
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-mono text-xs">Build Progress</span>
                        <span className="font-mono text-xs text-muted-foreground">42%</span>
                      </div>
                      <Progress value={42} className="h-1" />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="debug" className="space-y-4 mt-0">
                <Card className="border-border/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="font-mono text-sm flex items-center gap-2">
                      <span className="text-yellow-500">db</span>
                      Debug Tools
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs">Debug Mode</span>
                        <Switch 
                          checked={toolStates.debugMode}
                          onCheckedChange={() => toggleToolState('debugMode')}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs">Auto Save</span>
                        <Switch 
                          checked={toolStates.autoSave}
                          onCheckedChange={() => toggleToolState('autoSave')}
                        />
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div className="grid grid-cols-2 gap-2">
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="font-mono text-xs warp-section-highlight"
                        onClick={() => executeTool('inspect')}
                      >
                        <span className="mr-1">in</span>
                        Inspect
                      </Button>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="font-mono text-xs warp-section-highlight"
                        onClick={() => executeTool('trace')}
                      >
                        <span className="mr-1">tr</span>
                        Trace
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="config" className="space-y-4 mt-0">
                <Card className="border-border/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="font-mono text-sm flex items-center gap-2">
                      <span className="text-purple-500">cf</span>
                      Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs">Notifications</span>
                        <Switch 
                          checked={toolStates.notifications}
                          onCheckedChange={() => toggleToolState('notifications')}
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs">Dark Theme</span>
                        <Switch 
                          checked={toolStates.darkMode}
                          onCheckedChange={() => toggleToolState('darkMode')}
                        />
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div className="space-y-2">
                      <div className="font-mono text-xs text-muted-foreground">System Info</div>
                      <div className="text-xs space-y-1">
                        <div className="flex justify-between">
                          <span className="font-mono">Version:</span>
                          <span className="font-mono text-muted-foreground">1.0.0</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="font-mono">Platform:</span>
                          <span className="font-mono text-muted-foreground">web</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="font-mono">Runtime:</span>
                          <span className="font-mono text-muted-foreground">browser</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="search" className="space-y-4 mt-0">
                <Card className="border-border/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="font-mono text-sm flex items-center gap-2">
                      <span className="text-green-500">sr</span>
                      Deep Search
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="space-y-3">
                      <div className="font-mono text-xs text-muted-foreground">Search Archives & Databases</div>
                      
                      <Tabs value={activeSearchTab} onValueChange={setActiveSearchTab} className="space-y-3">
                        <TabsList className="grid w-full grid-cols-3 text-xs">
                          <TabsTrigger value="dictionaries" className="font-mono">
                            <span className="mr-1">dc</span>
                            Dict
                          </TabsTrigger>
                          <TabsTrigger value="wikipedia" className="font-mono">
                            <span className="mr-1">wp</span>
                            Wiki
                          </TabsTrigger>
                          <TabsTrigger value="academic" className="font-mono">
                            <span className="mr-1">ac</span>
                            Academic
                          </TabsTrigger>
                        </TabsList>
                        
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <Button 
                            variant={activeSearchTab === 'library-congress' ? 'default' : 'ghost'} 
                            size="sm" 
                            className="font-mono warp-section-highlight"
                            onClick={() => setActiveSearchTab('library-congress')}
                          >
                            <span className="mr-1">lc</span>
                            LoC
                          </Button>
                          <Button 
                            variant={activeSearchTab === 'internet-archive' ? 'default' : 'ghost'} 
                            size="sm" 
                            className="font-mono warp-section-highlight"
                            onClick={() => setActiveSearchTab('internet-archive')}
                          >
                            <span className="mr-1">ia</span>
                            Archive
                          </Button>
                          <Button 
                            variant={activeSearchTab === 'news' ? 'default' : 'ghost'} 
                            size="sm" 
                            className="font-mono warp-section-highlight"
                            onClick={() => setActiveSearchTab('news')}
                          >
                            <span className="mr-1">nw</span>
                            News
                          </Button>
                          <Button 
                            variant={activeSearchTab === 'patents' ? 'default' : 'ghost'} 
                            size="sm" 
                            className="font-mono warp-section-highlight"
                            onClick={() => setActiveSearchTab('patents')}
                          >
                            <span className="mr-1">pt</span>
                            Patents
                          </Button>
                        </div>
                        
                        <Separator />
                        
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-xs">Active Database:</span>
                            <span className="font-mono text-xs text-muted-foreground">
                              {activeSearchTab.replace('-', ' ')}
                            </span>
                          </div>
                          <Button 
                            onClick={() => addLine(`.\ opening deep search interface for ${activeSearchTab}...`, 'output')}
                            className="w-full font-mono text-xs warp-section-highlight"
                            size="sm"
                          >
                            <span className="mr-2">sr</span>
                            Open Search Interface
                          </Button>
                        </div>
                      </Tabs>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Deep Search Interface */}
                <div className="max-h-64 overflow-y-auto subtle-scrollbar">
                  <DeepSearch activeTab={activeSearchTab} />
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>

      {/* Terminal Content */}
      <div 
        ref={terminalRef}
        className="flex-1 p-4 overflow-y-auto font-mono text-sm"
      >
        {lines.map((line) => (
          <div 
            key={line.id} 
            className={`whitespace-pre-wrap break-words leading-relaxed terminal-line-in ${getLineColor(line.type)}`}
          >
            {line.content}
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center space-x-2">
          <span className="text-muted-foreground shrink-0">
            {isMultiLine ? '...' : '❯'}
          </span>
          <input
            ref={inputRef}
            type="text"
            value={isMultiLine ? multiLineBuffer : input}
            onChange={(e) => isMultiLine ? setMultiLineBuffer(e.target.value) : setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent border-none outline-none font-mono"
            placeholder={isMultiLine ? "Enter Python code (Ctrl+Enter to execute, Esc to exit)" : "Enter command..."}
            autoFocus
          />
        </div>
        {isMultiLine && (
          <div className="mt-2 text-xs text-muted-foreground">
            Python mode: Ctrl+Enter to execute • Esc to exit
          </div>
        )}
      </div>
    </div>
  );
}