import React, { useState, useEffect } from 'react';
import { LarkEditor } from './components/LarkEditor';
import { Terminal } from './components/Terminal';
import { LarkWaveform } from './components/LarkWaveform';
import { BookmarkManager } from './components/BookmarkManager';
import { FileManager } from './components/FileManager';
import { TemplateManager } from './components/TemplateManager';
import { AIChat } from './components/AIChat';
import { SettingsPanel } from './components/SettingsPanel';
import { DesktopLauncher } from './components/DesktopLauncher';
import { Sidebar } from './components/ui/sidebar';
import { Button } from './components/ui/button';

type LarkMode = 'editor' | 'terminal' | 'ai' | 'files' | 'templates' | 'settings' | 'codex';
type AudioActivity = 'idle' | 'talking' | 'ai-speaking' | 'listening' | 'processing';
type ThemeMode = 'light' | 'dark' | 'auto';

interface AudioSettings {
  waveformEnabled: boolean;
  sensitivity: number; // 0.1 to 1.0
  animationSpeed: number; // 0.5 to 2.0
  phraseRotationEnabled: boolean;
  phraseRotationSpeed: number; // 1-10 seconds
  autoActivityDetection: boolean;
  visualFeedbackEnabled: boolean;
  waveformStyle: 'minimal' | 'standard' | 'detailed';
  waveformColor: 'default' | 'accent' | 'primary';
}

interface TreeNode {
  name: string;
  type: 'file' | 'folder';
  children?: TreeNode[];
  expanded?: boolean;
}

export default function App() {
  const [mode, setMode] = useState<LarkMode>('editor');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [bottomPanelOpen, setBottomPanelOpen] = useState(false);
  const [quickActionsOpen, setQuickActionsOpen] = useState(false);
  const [historyPanelOpen, setHistoryPanelOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [audioActivity, setAudioActivity] = useState<AudioActivity>('idle');
  const [theme, setTheme] = useState<ThemeMode>('light');
  const [treeExpanded, setTreeExpanded] = useState(true);
  const [isDesktopApp, setIsDesktopApp] = useState(false);
  const [isPWAInstalled, setIsPWAInstalled] = useState(false);
  const [treeNodes, setTreeNodes] = useState<TreeNode[]>([
    {
      name: 'lark-workspace',
      type: 'folder',
      expanded: true,
      children: [
        {
          name: 'src',
          type: 'folder',
          expanded: true,
          children: [
            { name: 'main.py', type: 'file' },
            { name: 'config.json', type: 'file' },
            { name: 'utils.py', type: 'file' },
            { name: 'models.py', type: 'file' },
            { name: 'data_loader.py', type: 'file' }
          ]
        },
        {
          name: 'docs',
          type: 'folder',
          expanded: false,
          children: [
            { name: 'readme.md', type: 'file' },
            { name: 'api.md', type: 'file' },
            { name: 'setup.md', type: 'file' }
          ]
        },
        {
          name: 'notebooks',
          type: 'folder',
          expanded: false,
          children: [
            { name: 'experiment.ipynb', type: 'file' },
            { name: 'analysis.ipynb', type: 'file' },
            { name: 'visualization.ipynb', type: 'file' }
          ]
        },
        {
          name: 'tests',
          type: 'folder',
          expanded: false,
          children: [
            { name: 'test_main.py', type: 'file' },
            { name: 'test_utils.py', type: 'file' }
          ]
        },
        { name: 'requirements.txt', type: 'file' },
        { name: 'README.md', type: 'file' },
        { name: '.gitignore', type: 'file' },
        { name: 'setup.py', type: 'file' }
      ]
    }
  ]);
  
  // Audio settings state
  const [audioSettings, setAudioSettings] = useState<AudioSettings>({
    waveformEnabled: true,
    sensitivity: 0.8,
    animationSpeed: 1.0,
    phraseRotationEnabled: true,
    phraseRotationSpeed: 3, // seconds
    autoActivityDetection: true,
    visualFeedbackEnabled: true,
    waveformStyle: 'minimal',
    waveformColor: 'default'
  });

  // Desktop app and PWA detection
  useEffect(() => {
    // Detect if running in desktop app
    const isElectron = typeof window !== 'undefined' && window.process && window.process.type;
    const isDesktop = window.matchMedia('(display-mode: standalone)').matches || 
                     window.navigator.standalone === true || 
                     isElectron;
    
    setIsDesktopApp(isDesktop);

    // Check if PWA is already installed
    if ('getInstalledRelatedApps' in navigator) {
      (navigator as any).getInstalledRelatedApps().then((relatedApps: any[]) => {
        setIsPWAInstalled(relatedApps.length > 0);
      });
    }

    // Handle PWA install prompt
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      (window as any).deferredPrompt = e;
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    // Handle successful PWA installation
    const handleAppInstalled = () => {
      setIsPWAInstalled(true);
      (window as any).deferredPrompt = null;
    };

    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  // Service Worker registration
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js')
        .then((registration) => {
          console.log('SW registered:', registration);
        })
        .catch((error) => {
          console.log('SW registration failed:', error);
        });
    }
  }, []);

  // Handle protocol and file handlers
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const handler = urlParams.get('handler');
    const handlerUrl = urlParams.get('url');

    if (handler === 'protocol' && handlerUrl) {
      // Handle lark:// protocol
      console.log('Protocol handler:', handlerUrl);
      // Parse and handle the protocol URL
    }

    if (handler === 'file') {
      // Handle file opening
      console.log('File handler activated');
    }

    if (handler === 'share') {
      // Handle shared content
      console.log('Share handler activated');
    }

    // Handle initial mode from URL
    const initialMode = urlParams.get('mode') as LarkMode;
    if (initialMode && ['editor', 'terminal', 'ai', 'files', 'templates', 'codex', 'settings'].includes(initialMode)) {
      setMode(initialMode);
    }
  }, []);

  // Theme management
  useEffect(() => {
    // Load saved theme preference
    const savedTheme = localStorage.getItem('lark-theme') as ThemeMode;
    if (savedTheme) {
      setTheme(savedTheme);
    } else {
      // Default to auto mode, which follows system preference
      setTheme('auto');
    }
  }, []);

  useEffect(() => {
    // Save theme preference
    localStorage.setItem('lark-theme', theme);
    
    // Apply theme to document
    const root = document.documentElement;
    
    if (theme === 'dark') {
      root.classList.add('dark');
    } else if (theme === 'light') {
      root.classList.remove('dark');
    } else {
      // Auto mode - follow system preference
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      if (mediaQuery.matches) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
      
      // Listen for system theme changes
      const handleChange = (e: MediaQueryListEvent) => {
        if (theme === 'auto') {
          if (e.matches) {
            root.classList.add('dark');
          } else {
            root.classList.remove('dark');
          }
        }
      };
      
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [theme]);

  const toggleTheme = () => {
    const modes: ThemeMode[] = ['light', 'dark', 'auto'];
    const currentIndex = modes.indexOf(theme);
    const nextIndex = (currentIndex + 1) % modes.length;
    setTheme(modes[nextIndex]);
  };

  const getThemeIcon = () => {
    switch (theme) {
      case 'light': return 'lt';
      case 'dark': return 'dk';
      case 'auto': return 'au';
      default: return 'au';
    }
  };

  const getThemeLabel = () => {
    switch (theme) {
      case 'light': return 'light';
      case 'dark': return 'dark';
      case 'auto': return 'auto';
      default: return 'auto';
    }
  };

  // Load audio settings from localStorage
  useEffect(() => {
    const savedAudioSettings = localStorage.getItem('lark-audio-settings');
    if (savedAudioSettings) {
      setAudioSettings(JSON.parse(savedAudioSettings));
    }
  }, []);

  // Save audio settings to localStorage
  useEffect(() => {
    localStorage.setItem('lark-audio-settings', JSON.stringify(audioSettings));
  }, [audioSettings]);

  // Initialize Lark on first load
  useEffect(() => {
    const welcomeMessage = `
# Welcome to Lark IANA

*A unified AI development environment for modern creators*

## What is Lark?

Lark is your comprehensive workspace for AI development, combining the power of:
- **Markdown editing** with live preview and syntax highlighting
- **Terminal interface** with development toolset integration
- **AI conversation** and intelligent text processing
- **File management** with tree view and project organization
- **Template system** for rapid prototyping and scaffolding
- **Codex bookmarks** for knowledge management and research
- **Deep search** across multiple databases and archives
- **Desktop app support** with native OS integration

## Development Toolset

### Terminal Tools
- **Quick Actions** - Save, refresh, format, lint with one click
- **Build Tools** - Build, test, deploy, and watch your projects
- **Debug Tools** - Inspect, trace, and debug with integrated controls
- **Search Tools** - Deep search across dictionaries, Wikipedia, academic papers, and archives

### File Management
- **Tree View** - Navigate your project structure with expandable folders
- **List/Grid Views** - Multiple ways to browse and organize files
- **Quick Actions** - Create, upload, download, and manage files
- **Search & Filter** - Find files quickly with integrated search

### AI Integration
- **Multiple Models** - GPT-4, GPT-3.5 Turbo, Claude 3 support
- **Smart Responses** - Context-aware assistance for development tasks
- **Code Help** - Debugging, optimization, and best practices
- **Documentation** - Automated documentation generation

### Template System
- **Language Templates** - Python, JavaScript, Markdown, LaTeX
- **Project Scaffolds** - ML projects, API documentation, research papers
- **Quick Start** - Instant project setup with best practices
- **Custom Templates** - Create and share your own templates

### Desktop Integration
- **Native Apps** - Electron, Swift, and PWA versions available
- **Protocol Handlers** - Deep linking from external applications
- **File Handlers** - Open files directly in Lark
- **Background Sync** - Offline capabilities and automatic syncing

### Audio & Visual Feedback
- **Waveform Visualization** - Real-time audio activity indicators
- **Terminal Animations** - Warp-style UI effects and transitions
- **Activity Detection** - Automatic mode switching based on usage
- **Theme System** - Light, dark, and auto modes with persistence

## Quick Start Guide

1. **Choose Your Mode** - Select from the sidebar (ed/tm/ai/fl/tp/cx/st)
2. **Configure Settings** - Personalize your workspace in Settings
3. **Start Creating** - Use templates or start from scratch
4. **Leverage AI** - Get help with coding, documentation, and research
5. **Organize Knowledge** - Use Codex to bookmark and manage resources
6. **Go Desktop** - Install as desktop app for enhanced features

## Terminal Commands

- \`help\` - Show available commands
- \`cx add <url>\` - Bookmark a webpage
- \`search\` - Access deep search toolset
- \`python\` - Enter Python mode
- \`ai <message>\` - Query AI assistant
- \`clear\` - Clear terminal output

## Keyboard Shortcuts

- \`Ctrl+Enter\` - Execute in Python mode
- \`Enter\` - Send message in AI chat
- \`Esc\` - Exit modes or close panels
- \`Shift+Enter\` - New line in text areas

---

*Ready to revolutionize your development workflow? Choose a tool from the sidebar and start building!*

**Version:** Lark IANA 1.0.0  
**Build:** Production  
**Platform:** ${isDesktopApp ? 'Desktop Application' : 'Universal Web Application'}
`;

    // Set initial content if none exists
    const savedContent = localStorage.getItem('lark-current-content');
    if (!savedContent) {
      localStorage.setItem('lark-current-content', welcomeMessage);
    }
  }, [isDesktopApp]);

  // Audio activity simulation based on mode and interactions
  useEffect(() => {
    if (!audioSettings.autoActivityDetection) return;

    let timeoutId: NodeJS.Timeout;

    if (mode === 'ai') {
      // Simulate AI conversation activity
      setAudioActivity('ai-speaking');
      timeoutId = setTimeout(() => {
        setAudioActivity('listening');
        setTimeout(() => setAudioActivity('idle'), 2000);
      }, 3000);
    } else if (mode === 'terminal') {
      // Simulate processing activity
      setAudioActivity('processing');
      timeoutId = setTimeout(() => setAudioActivity('idle'), 2000);
    } else if (mode === 'codex') {
      // Simulate knowledge processing activity
      setAudioActivity('processing');
      timeoutId = setTimeout(() => setAudioActivity('idle'), 1500);
    } else if (mode === 'files') {
      // Simulate file system activity
      setAudioActivity('processing');
      timeoutId = setTimeout(() => setAudioActivity('idle'), 1000);
    } else {
      // Default to idle for other modes
      setAudioActivity('idle');
    }

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [mode, audioSettings.autoActivityDetection]);

  // Demo function to simulate different audio activities
  const simulateAudioActivity = (activity: AudioActivity, duration: number = 3000) => {
    setAudioActivity(activity);
    setTimeout(() => setAudioActivity('idle'), duration);
  };

  // Update audio settings
  const updateAudioSettings = (newSettings: Partial<AudioSettings>) => {
    setAudioSettings(prev => ({ ...prev, ...newSettings }));
  };

  // Tree node toggle function
  const toggleTreeNode = (nodePath: number[]) => {
    const toggleNode = (nodes: TreeNode[], path: number[], depth: number = 0): TreeNode[] => {
      return nodes.map((node, index) => {
        if (depth === path.length - 1 && index === path[depth]) {
          return { ...node, expanded: !node.expanded };
        } else if (depth < path.length - 1 && index === path[depth] && node.children) {
          return { ...node, children: toggleNode(node.children, path, depth + 1) };
        }
        return node;
      });
    };

    setTreeNodes(prev => toggleNode(prev, nodePath));
  };

  // Render tree node
  const renderTreeNode = (node: TreeNode, level: number = 0, path: number[] = []): React.ReactNode => {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = node.expanded;

    return (
      <div key={`${path.join('-')}-${node.name}`} className="select-none">
        <div
          className={`flex items-center py-1 px-2 text-xs font-mono cursor-pointer hover:bg-sidebar-accent/50 warp-section-highlight ${
            sidebarOpen ? '' : 'justify-center'
          }`}
          style={{ paddingLeft: sidebarOpen ? `${8 + level * 12}px` : '8px' }}
          onClick={() => {
            if (hasChildren) {
              toggleTreeNode([...path]);
            } else {
              // Handle file click - could switch to editor mode with file content
              console.log('File clicked:', node.name);
              if (audioSettings.autoActivityDetection) {
                simulateAudioActivity('processing', 800);
              }
            }
          }}
        >
          {hasChildren && sidebarOpen && (
            <span className="mr-1 text-sidebar-foreground/60">
              {isExpanded ? '▼' : '▶'}
            </span>
          )}
          {!hasChildren && sidebarOpen && (
            <span className="mr-1 text-sidebar-foreground/60">📄</span>
          )}
          {hasChildren && !sidebarOpen && (
            <span className="text-sidebar-foreground/60">📁</span>
          )}
          {!hasChildren && !sidebarOpen && (
            <span className="text-sidebar-foreground/60">📄</span>
          )}
          {sidebarOpen && (
            <span
              className="text-sidebar-foreground/80 truncate animate-underline"
              title={node.name}
            >
              {node.name}
            </span>
          )}
        </div>
        {hasChildren && isExpanded && node.children && sidebarOpen && (
          <div className="animate-terminal-fade">
            {node.children.map((child, index) =>
              renderTreeNode(child, level + 1, [...path, index])
            )}
          </div>
        )}
      </div>
    );
  };

  const sidebarItems = [
    {
      id: 'editor' as LarkMode,
      label: 'Editor',
      icon: <span className="font-mono text-xs">ed</span>,
      description: 'Markdown editor with live preview'
    },
    {
      id: 'terminal' as LarkMode,
      label: 'Terminal',
      icon: <span className="font-mono text-xs">tm</span>,
      description: 'Development toolset and terminal'
    },
    {
      id: 'ai' as LarkMode,
      label: 'AI Chat',
      icon: <span className="font-mono text-xs">ai</span>,
      description: 'AI conversation and assistance'
    },
    {
      id: 'files' as LarkMode,
      label: 'Files',
      icon: <span className="font-mono text-xs">fl</span>,
      description: 'Project and file management'
    },
    {
      id: 'templates' as LarkMode,
      label: 'Templates',
      icon: <span className="font-mono text-xs">tp</span>,
      description: 'Quick start templates and scaffolds'
    },
    {
      id: 'codex' as LarkMode,
      label: 'Codex',
      icon: <span className="font-mono text-xs">cx</span>,
      description: 'Bookmark and knowledge manager'
    },
    {
      id: 'settings' as LarkMode,
      label: 'Settings',
      icon: <span className="font-mono text-xs">st</span>,
      description: 'Application preferences and config'
    }
  ];

  const renderMainContent = () => {
    switch (mode) {
      case 'terminal':
        return <Terminal onModeChange={setMode} />;
      case 'ai':
        return <AIChat />;
      case 'files':
        return <FileManager />;
      case 'templates':
        return <TemplateManager />;
      case 'codex':
        return <BookmarkManager />;
      case 'settings':
        return <SettingsPanel />;
      default:
        return (
          <LarkEditor 
            mode={mode} 
            audioSettings={audioSettings}
            onAudioSettingsChange={updateAudioSettings}
          />
        );
    }
  };

  // Calculate content height - optimized for different modes
  const getContentHeight = () => {
    // Base height optimized for desktop viewing
    let height = '700px';
    
    // Adjust for bottom panel
    if (bottomPanelOpen) {
      height = '550px'; // Accommodate bottom panel
    }
    
    // Different heights for different modes
    switch (mode) {
      case 'ai':
      case 'files':
      case 'templates':
      case 'settings':
        height = bottomPanelOpen ? '550px' : '700px';
        break;
      default:
        height = bottomPanelOpen ? '550px' : '700px';
    }
    
    return height;
  };

  // If terminal mode, render full-screen terminal only
  if (mode === 'terminal') {
    return (
      <div className="h-screen w-full bg-background">
        <Terminal onModeChange={setMode} />
      </div>
    );
  }

  return (
    <div className="min-h-screen w-full flex flex-col bg-background">
      <div className="flex-1 flex max-w-screen-2xl mx-auto w-full">
        {/* Sidebar */}
        <div className={`${sidebarOpen ? 'w-64' : 'w-16'} transition-all duration-300 border-r border-border bg-sidebar flex flex-col`}>
          <div className="p-4 border-b border-sidebar-border">
            <div className="flex items-center space-x-3">
              <div 
                className="w-8 h-8 bg-[rgba(255,255,255,1)] rounded-lg flex items-center justify-center text-[rgba(255,255,255,1)] font-[Aldrich] animate-border-grow cursor-pointer"
                onClick={() => simulateAudioActivity('talking', 2000)}
                title="Click to simulate talking"
              >
                <span className="text-[rgba(0,0,0,1)] text-sm text-[20px] font-[Zen_Kurenaido]">.\</span>
              </div>
              {sidebarOpen && (
                <div className="animate-terminal-line flex-1">
                  <h1 className="font-medium text-sidebar-foreground font-[Aldrich]">lark</h1>
                  <LarkWaveform 
                    isActive={audioSettings.waveformEnabled}
                    activity={audioActivity}
                    settings={audioSettings}
                  />
                </div>
              )}
            </div>
          </div>

          {/* File Tree Section */}
          <div className="border-b border-sidebar-border">
            <div className="p-2">
              <Button
                variant="ghost"
                size="sm"
                className={`w-full justify-start h-8 px-2 animate-accent-line ${sidebarOpen ? '' : 'px-2'}`}
                onClick={() => setTreeExpanded(!treeExpanded)}
              >
                <span className="animate-pulse-dot">
                  <span className="font-mono text-xs">tr</span>
                </span>
                {sidebarOpen && (
                  <div className="ml-3 text-left animate-bg-expand flex-1">
                    <div className="text-sm font-mono animate-underline flex items-center justify-between">
                      <span>Tree</span>
                      <span className="text-xs">
                        {treeExpanded ? '−' : '+'}
                      </span>
                    </div>
                  </div>
                )}
              </Button>
            </div>
            
            {treeExpanded && (
              <div className="pb-2 max-h-64 overflow-y-auto subtle-scrollbar">
                {treeNodes.map((node, index) => renderTreeNode(node, 0, [index]))}
              </div>
            )}
          </div>

          <nav className="flex-1 p-2 flex flex-col-reverse">
            {sidebarItems.map((item) => (
              <Button
                key={item.id}
                variant={mode === item.id ? 'default' : 'ghost'}
                className={`w-full justify-start mt-1 warp-section-highlight hover:bg-black/30 ${sidebarOpen ? 'px-3' : 'px-2'}`}
                onClick={() => {
                  setMode(item.id);
                  // Simulate different activities based on mode
                  if (audioSettings.autoActivityDetection) {
                    if (item.id === 'ai') {
                      simulateAudioActivity('ai-speaking', 4000);
                    } else if (item.id === 'terminal') {
                      simulateAudioActivity('processing', 2000);
                    } else if (item.id === 'codex') {
                      simulateAudioActivity('processing', 1500);
                    } else if (item.id === 'files') {
                      simulateAudioActivity('processing', 1200);
                    } else if (item.id === 'templates') {
                      simulateAudioActivity('processing', 1000);
                    } else if (item.id === 'settings') {
                      simulateAudioActivity('processing', 800);
                    }
                  }
                }}
              >
                <span className="animate-pulse-dot">{item.icon}</span>
                {sidebarOpen && (
                  <div className="ml-3 text-left animate-bg-expand">
                    <div className="text-sm font-mono animate-underline">{item.label}</div>
                    <div className="text-xs text-[rgba(104,104,108,1)] font-mono">{item.description}</div>
                  </div>
                )}
              </Button>
            ))}
          </nav>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="h-14 border-b border-border bg-card flex items-center justify-between px-6">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2 animate-slide-indicator">
                {mode === 'editor' && <span className="font-mono text-sm">ed</span>}
                {mode === 'terminal' && <span className="font-mono text-sm">tm</span>}
                {mode === 'ai' && <span className="font-mono text-sm">ai</span>}
                {mode === 'files' && <span className="font-mono text-sm">fl</span>}
                {mode === 'templates' && <span className="font-mono text-sm">tp</span>}
                {mode === 'codex' && <span className="font-mono text-sm">cx</span>}
                {mode === 'settings' && <span className="font-mono text-sm">st</span>}
                <span className="font-medium capitalize font-mono">{mode}</span>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-sm text-muted-foreground animate-terminal-line">
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xs">⚡</span>
                <span className="font-mono">Lark IANA v1.0</span>
                {isDesktopApp && (
                  <span className="font-mono text-xs text-green-600">Desktop</span>
                )}
                {isPWAInstalled && !isDesktopApp && (
                  <span className="font-mono text-xs text-blue-600">PWA</span>
                )}
                {audioSettings.waveformEnabled && audioActivity !== 'idle' && (
                  <div className="flex items-center space-x-1">
                    <span className="font-mono text-xs">•</span>
                    <span className="font-mono text-xs">(.audio)</span>
                  </div>
                )}
              </div>

              {/* Desktop Launcher */}
              {!isDesktopApp && (
                <DesktopLauncher compact />
              )}
              
              {/* Theme Toggle */}
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleTheme}
                className="h-8 px-2 animate-accent-line"
                title={`Theme: ${getThemeLabel()}`}
              >
                <div className="flex items-center space-x-1">
                  <span className="font-mono text-xs">{getThemeIcon()}</span>
                  <span className="font-mono text-xs">{getThemeLabel()}</span>
                </div>
              </Button>
            </div>
          </div>

          {/* Content Area - Fixed desktop-friendly height with scrolling */}
          <div 
            className="overflow-y-auto overflow-x-hidden bg-background border-b border-border"
            style={{ height: getContentHeight() }}
          >
            {renderMainContent()}
          </div>
        </div>
      </div>

      {/* Footer Control Bar - Always visible at bottom */}
      <div className="h-12 border-t border-border bg-card flex items-center justify-between px-4 max-w-screen-2xl mx-auto w-full">
        {/* Left Side Controls */}
        <div className="flex items-center space-x-2">
          {/* Sidebar Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="h-8 px-2 animate-accent-line"
          >
            <span className="font-mono text-xs">pl</span>
            {sidebarOpen ? <span className="font-mono text-xs ml-1">&lt;</span> : <span className="font-mono text-xs ml-1">&gt;</span>}
          </Button>

          {/* Bottom Panel Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setBottomPanelOpen(!bottomPanelOpen)}
            className="h-8 px-2 animate-accent-line"
          >
            <span className="font-mono text-xs">pb</span>
            {bottomPanelOpen ? <span className="text-foreground font-mono text-xs ml-1">.\</span> : <span className="text-foreground font-mono text-xs ml-1">.\</span>}
          </Button>

          {/* Search Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSearchOpen(!searchOpen)}
            className="h-8 px-2 animate-accent-line"
          >
            <span className="font-mono text-xs">sr</span>
          </Button>

          {/* Inline Search Input */}
          {searchOpen && (
            <div className="flex items-center space-x-2 terminal-section-fade">
              <input
                type="text"
                placeholder="Search files, content, commands..."
                className="w-64 px-3 py-1 bg-input border border-border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring font-mono animate-border-grow"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    simulateAudioActivity('processing', 1500);
                    console.log('Search:', e.currentTarget.value);
                  }
                  if (e.key === 'Escape') {
                    setSearchOpen(false);
                  }
                }}
                autoFocus
              />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSearchOpen(false)}
                className="h-8 px-2 animate-border-grow"
              >
                <span className="font-mono text-xs">×</span>
              </Button>
            </div>
          )}
        </div>

        {/* Center Status */}
        <div className="flex items-center space-x-4 text-xs text-muted-foreground">
          <span className="font-mono animate-pulse-dot">Ready</span>
          <span className="font-mono">•</span>
          <span className="font-mono animate-underline">{mode} mode</span>
          {isDesktopApp && (
            <>
              <span className="font-mono">•</span>
              <span className="font-mono text-green-600">Desktop App</span>
            </>
          )}
        </div>

        {/* Right Side Controls */}
        <div className="flex items-center space-x-2">
          {/* Audio Settings Quick Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => updateAudioSettings({ waveformEnabled: !audioSettings.waveformEnabled })}
            className={`h-8 px-2 animate-accent-line ${audioSettings.waveformEnabled ? 'bg-muted' : ''}`}
            title="Toggle waveform"
          >
            <span className="font-mono text-xs">wv</span>
          </Button>

          {/* History Panel */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setHistoryPanelOpen(!historyPanelOpen)}
            className="h-8 px-2 animate-accent-line"
          >
            <span className="font-mono text-xs">hs</span>
          </Button>

          {/* Quick Actions */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setQuickActionsOpen(!quickActionsOpen)}
            className="h-8 px-2 animate-accent-line"
          >
            <span className="font-mono text-xs">cm</span>
            {quickActionsOpen ? <span className="font-mono text-xs ml-1">^</span> : <span className="font-mono text-xs ml-1">v</span>}
          </Button>

          {/* Audio Activity Demo Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => simulateAudioActivity('listening', 3000)}
            className="h-8 px-2 animate-accent-line"
            title=".speak"
          >
            <span className="font-mono text-xs">sp</span>
          </Button>

          {/* Layers Panel */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => console.log('Layers panel - coming soon')}
            className="h-8 px-2 animate-accent-line"
          >
            <span className="font-mono text-xs">ly</span>
          </Button>
        </div>
      </div>

      {/* Bottom Panel (Console Output) - Below footer */}
      {bottomPanelOpen && (
        <div className="max-w-screen-2xl mx-auto w-full">
          <div className="h-48 border-t border-border bg-card p-4 terminal-section-fade">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium font-mono animate-underline">Console Output</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setBottomPanelOpen(false)}
                className="h-6 w-6 p-0 animate-border-grow"
              >
                <span className="font-mono text-xs">×</span>
              </Button>
            </div>
            <div className="h-32 bg-muted rounded p-2 font-mono text-sm overflow-y-auto animate-bg-expand">
              <div className="text-muted-foreground terminal-line-in">Console ready...</div>
              <div className="text-green-600 terminal-line-in">✓ Lark IANA initialized successfully</div>
              <div className="text-muted-foreground terminal-line-in">✓ Development toolset loaded</div>
              <div className="text-muted-foreground terminal-line-in">✓ Audio system configured</div>
              <div className="text-muted-foreground terminal-line-in">✓ File manager ready</div>
              <div className="text-muted-foreground terminal-line-in">✓ Template system loaded</div>
              <div className="text-muted-foreground terminal-line-in">✓ AI chat system ready</div>
              <div className="text-muted-foreground terminal-line-in">✓ Codex bookmark system ready</div>
              <div className="text-muted-foreground terminal-line-in">✓ Deep search tools active</div>
              <div className="text-muted-foreground terminal-line-in">✓ Theme system initialized</div>
              <div className="text-muted-foreground terminal-line-in">✓ File tree loaded</div>
              {isDesktopApp && (
                <div className="text-blue-600 terminal-line-in">✓ Desktop app features enabled</div>
              )}
              {isPWAInstalled && (
                <div className="text-purple-600 terminal-line-in">✓ PWA installation detected</div>
              )}
              <div className="text-muted-foreground terminal-line-in">All systems operational. Ready for development.</div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions Panel */}
      {quickActionsOpen && (
        <div className="absolute bottom-12 right-4 w-64 bg-card border border-border rounded-lg shadow-lg p-4 z-50 terminal-section-fade">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-sm font-mono animate-underline">Quick Actions</h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setQuickActionsOpen(false)}
              className="h-6 w-6 p-0 animate-border-grow"
            >
              <span className="font-mono text-xs">×</span>
            </Button>
          </div>
          <div className="space-y-2">
            <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight" onClick={() => setMode('editor')}>
              <span className="font-mono text-xs mr-2 animate-pulse-dot">ed</span>
              <span className="font-mono text-sm">New Document</span>
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight" onClick={() => setMode('terminal')}>
              <span className="font-mono text-xs mr-2 animate-pulse-dot">tm</span>
              <span className="font-mono text-sm">Open Terminal</span>
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight" onClick={() => {
              setMode('ai');
              simulateAudioActivity('ai-speaking', 3000);
            }}>
              <span className="font-mono text-xs mr-2 animate-pulse-dot">ai</span>
              <span className="font-mono text-sm">Ask AI</span>
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight" onClick={() => setMode('files')}>
              <span className="font-mono text-xs mr-2 animate-pulse-dot">fl</span>
              <span className="font-mono text-sm">File Manager</span>
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight" onClick={() => setMode('templates')}>
              <span className="font-mono text-xs mr-2 animate-pulse-dot">tp</span>
              <span className="font-mono text-sm">Templates</span>
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight" onClick={() => {
              setMode('codex');
              simulateAudioActivity('processing', 2000);
            }}>
              <span className="font-mono text-xs mr-2 animate-pulse-dot">cx</span>
              <span className="font-mono text-sm">Bookmark Manager</span>
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight" onClick={() => {
              setSearchOpen(!searchOpen);
              simulateAudioActivity('listening', 2000);
            }}>
              <span className="font-mono text-xs mr-2 animate-pulse-dot">sr</span>
              <span className="font-mono text-sm">Search Files</span>
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight" onClick={toggleTheme}>
              <span className="font-mono text-xs mr-2 animate-pulse-dot">{getThemeIcon()}</span>
              <span className="font-mono text-sm">Toggle Theme</span>
            </Button>
            {!isDesktopApp && (
              <Button variant="ghost" size="sm" className="w-full justify-start warp-section-highlight text-blue-600">
                <span className="font-mono text-xs mr-2 animate-pulse-dot">🚀</span>
                <span className="font-mono text-sm">Launch Desktop</span>
              </Button>
            )}
          </div>
        </div>
      )}

      {/* History Panel */}
      {historyPanelOpen && (
        <div className="absolute bottom-12 right-16 w-48 bg-card border border-border rounded-lg shadow-lg p-4 z-50 terminal-section-fade">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-sm font-mono animate-underline">Recent Files</h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setHistoryPanelOpen(false)}
              className="h-6 w-6 p-0 animate-border-grow"
            >
              <span className="font-mono text-xs">×</span>
            </Button>
          </div>
          <div className="space-y-2">
            <div className="text-sm text-muted-foreground font-mono warp-section-highlight cursor-pointer" onClick={() => setMode('editor')}>
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xs">📄</span>
                <span>main.py</span>
              </div>
              <div className="text-xs opacity-70">2 min ago</div>
            </div>
            <div className="text-sm text-muted-foreground font-mono warp-section-highlight cursor-pointer" onClick={() => setMode('editor')}>
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xs">📄</span>
                <span>readme.md</span>
              </div>
              <div className="text-xs opacity-70">5 min ago</div>
            </div>
            <div className="text-sm text-muted-foreground font-mono warp-section-highlight cursor-pointer" onClick={() => setMode('editor')}>
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xs">📄</span>
                <span>config.json</span>
              </div>
              <div className="text-xs opacity-70">1 hour ago</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}