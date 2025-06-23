import React, { useState, useRef } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ScrollArea } from './ui/scroll-area';

interface ConfigStep {
  id: string;
  title: string;
  description: string;
  code?: string;
  command?: string;
  status: 'pending' | 'complete' | 'error';
}

interface IntegrationTarget {
  id: string;
  name: string;
  description: string;
  icon: string;
  supported: boolean;
  complexity: 'simple' | 'moderate' | 'advanced';
}

export function IntegrationPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [configSteps, setConfigSteps] = useState<ConfigStep[]>([
    {
      id: 'env',
      title: 'environment setup',
      description: 'configure development environment variables',
      status: 'pending'
    },
    {
      id: 'deps',
      title: 'install dependencies',
      description: 'add required packages to your project',
      code: 'npm install @lark/core @lark/electron @lark/webstorm-plugin',
      status: 'pending'
    },
    {
      id: 'config',
      title: 'configure lark',
      description: 'initialize lark configuration file',
      status: 'pending'
    },
    {
      id: 'integrate',
      title: 'integrate components',
      description: 'embed lark components in your application',
      status: 'pending'
    }
  ]);

  const [selectedIntegration, setSelectedIntegration] = useState<string>('electron');

  const integrationTargets: IntegrationTarget[] = [
    {
      id: 'electron',
      name: 'Electron Apps',
      description: 'Desktop application integration with native OS features',
      icon: 'el',
      supported: true,
      complexity: 'moderate'
    },
    {
      id: 'webstorm',
      name: 'WebStorm IDE',
      description: 'IDE plugin for enhanced development workflow',
      icon: 'ws',
      supported: true,
      complexity: 'simple'
    },
    {
      id: 'vscode',
      name: 'VS Code Extension',
      description: 'Extension for Visual Studio Code integration',
      icon: 'vs',
      supported: true,
      complexity: 'simple'
    },
    {
      id: 'web',
      name: 'Web Application',
      description: 'Browser-based integration for web apps',
      icon: 'wb',
      supported: true,
      complexity: 'simple'
    },
    {
      id: 'cli',
      name: 'CLI Tool',
      description: 'Command-line interface for terminal workflows',
      icon: 'tm',
      supported: true,
      complexity: 'advanced'
    },
    {
      id: 'api',
      name: 'REST API',
      description: 'HTTP API for service integration',
      icon: 'ap',
      supported: false,
      complexity: 'advanced'
    }
  ];

  const toggleStepStatus = (stepId: string) => {
    setConfigSteps(prev => prev.map(step => 
      step.id === stepId 
        ? { ...step, status: step.status === 'complete' ? 'pending' : 'complete' }
        : step
    ));
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'simple': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'moderate': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'advanced': return 'bg-red-500/20 text-red-400 border-red-500/30';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getSelectedTarget = () => integrationTargets.find(t => t.id === selectedIntegration);

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="text-center py-8">
        <div className="w-16 h-16 bg-[rgba(255,255,255,1)] rounded-lg flex items-center justify-center mx-auto mb-4 animate-border-grow">
          <span className="text-[rgba(0,0,0,1)] font-mono text-2xl">.\</span>
        </div>
        <h1 className="text-2xl font-mono mb-2 animate-underline">lark integration</h1>
        <p className="font-mono text-sm text-muted-foreground">
          integrate the lark ai development environment into your workflow
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {integrationTargets.map((target) => (
          <Card 
            key={target.id}
            className={`cursor-pointer transition-all duration-200 warp-section-highlight ${
              selectedIntegration === target.id ? 'ring-2 ring-primary' : ''
            } ${!target.supported ? 'opacity-50' : ''}`}
            onClick={() => target.supported && setSelectedIntegration(target.id)}
          >
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-muted rounded flex items-center justify-center animate-border-grow">
                    <span className="font-mono text-xs">{target.icon}</span>
                  </div>
                  <div>
                    <CardTitle className="font-mono text-sm animate-underline">{target.name}</CardTitle>
                    <Badge 
                      variant="outline" 
                      className={`text-xs mt-1 ${getComplexityColor(target.complexity)}`}
                    >
                      {target.complexity}
                    </Badge>
                  </div>
                </div>
                {!target.supported && (
                  <Badge variant="outline" className="text-xs font-mono">
                    soon
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription className="font-mono text-xs">
                {target.description}
              </CardDescription>
            </CardContent>
          </Card>
        ))}
      </div>

      {getSelectedTarget() && (
        <Card className="animate-bg-expand">
          <CardHeader>
            <CardTitle className="font-mono text-sm animate-underline">
              {getSelectedTarget()?.name} Integration
            </CardTitle>
            <CardDescription className="font-mono text-xs">
              follow the steps below to integrate lark into your {getSelectedTarget()?.name.toLowerCase()}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {configSteps.map((step, index) => (
                <div 
                  key={step.id}
                  className={`flex items-center space-x-3 p-3 border rounded transition-colors cursor-pointer warp-section-highlight ${
                    step.status === 'complete' ? 'bg-green-500/10 border-green-500/30' : 'border-border'
                  }`}
                  onClick={() => toggleStepStatus(step.id)}
                >
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-mono ${
                    step.status === 'complete' 
                      ? 'bg-green-500 text-white' 
                      : 'bg-muted text-muted-foreground'
                  }`}>
                    {step.status === 'complete' ? '✓' : index + 1}
                  </div>
                  <div className="flex-1">
                    <div className="font-mono text-sm">{step.title}</div>
                    <div className="font-mono text-xs text-muted-foreground">{step.description}</div>
                  </div>
                  <span className="font-mono text-xs text-muted-foreground">
                    {step.status === 'complete' ? 'done' : 'todo'}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderElectronSetup = () => (
    <div className="space-y-6">
      <div>
        <h2 className="font-mono text-lg mb-2 animate-underline">electron integration</h2>
        <p className="font-mono text-sm text-muted-foreground">
          integrate lark into your electron desktop application
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm animate-underline">package.json</CardTitle>
          <CardDescription className="font-mono text-xs">
            add lark dependencies to your electron project
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded p-4 font-mono text-sm overflow-x-auto animate-bg-expand">
            <div className="flex items-center justify-between mb-2">
              <span className="text-muted-foreground text-xs">json</span>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => copyToClipboard(`{
  "dependencies": {
    "@lark/core": "^1.0.0",
    "@lark/electron": "^1.0.0",
    "@lark/components": "^1.0.0",
    "electron": "^28.0.0"
  },
  "main": "src/main.js",
  "scripts": {
    "start": "electron .",
    "lark:dev": "lark dev --electron",
    "lark:build": "lark build --target=electron"
  }
}`)}
                className="h-6 px-2 text-xs"
              >
                copy
              </Button>
            </div>
            <pre className="text-foreground">{`{
  "dependencies": {
    "@lark/core": "^1.0.0",
    "@lark/electron": "^1.0.0", 
    "@lark/components": "^1.0.0",
    "electron": "^28.0.0"
  },
  "main": "src/main.js",
  "scripts": {
    "start": "electron .",
    "lark:dev": "lark dev --electron",
    "lark:build": "lark build --target=electron"
  }
}`}</pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm animate-underline">main process</CardTitle>
          <CardDescription className="font-mono text-xs">
            configure electron main process with lark
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded p-4 font-mono text-sm overflow-x-auto animate-bg-expand">
            <div className="flex items-center justify-between mb-2">
              <span className="text-muted-foreground text-xs">src/main.js</span>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => copyToClipboard(`const { app, BrowserWindow } = require('electron');
const { LarkElectron } = require('@lark/electron');

// Initialize Lark
const lark = new LarkElectron({
  terminal: true,
  ai: true,
  fileManager: true
});

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    titleBarStyle: 'hiddenInset',
    frame: false
  });

  // Load Lark interface
  win.loadFile('src/renderer/index.html');
  
  // Setup Lark IPC handlers
  lark.setupIPC(win);
}

app.whenReady().then(createWindow);`)}
                className="h-6 px-2 text-xs"
              >
                copy
              </Button>
            </div>
            <pre className="text-foreground">{`const { app, BrowserWindow } = require('electron');
const { LarkElectron } = require('@lark/electron');

// Initialize Lark
const lark = new LarkElectron({
  terminal: true,
  ai: true,
  fileManager: true
});

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    titleBarStyle: 'hiddenInset',
    frame: false
  });

  // Load Lark interface
  win.loadFile('src/renderer/index.html');
  
  // Setup Lark IPC handlers
  lark.setupIPC(win);
}

app.whenReady().then(createWindow);`}</pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm animate-underline">renderer process</CardTitle>
          <CardDescription className="font-mono text-xs">
            integrate lark components in your electron renderer
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded p-4 font-mono text-sm overflow-x-auto animate-bg-expand">
            <div className="flex items-center justify-between mb-2">
              <span className="text-muted-foreground text-xs">src/renderer/app.tsx</span>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => copyToClipboard(`import React from 'react';
import { LarkApp, LarkProvider } from '@lark/components';

export function App() {
  return (
    <LarkProvider 
      config={{
        terminal: {
          enabled: true,
          shell: 'zsh',
          charmTools: true
        },
        ai: {
          enabled: true,
          provider: 'local', // or 'openai'
          model: 'gpt-4'
        },
        editor: {
          enabled: true,
          syntax: 'markdown',
          preview: true
        }
      }}
    >
      <LarkApp />
    </LarkProvider>
  );
}`)}
                className="h-6 px-2 text-xs"
              >
                copy
              </Button>
            </div>
            <pre className="text-foreground">{`import React from 'react';
import { LarkApp, LarkProvider } from '@lark/components';

export function App() {
  return (
    <LarkProvider 
      config={{
        terminal: {
          enabled: true,
          shell: 'zsh',
          charmTools: true
        },
        ai: {
          enabled: true,
          provider: 'local', // or 'openai'
          model: 'gpt-4'
        },
        editor: {
          enabled: true,
          syntax: 'markdown',
          preview: true
        }
      }}
    >
      <LarkApp />
    </LarkProvider>
  );
}`}</pre>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderWebStormSetup = () => (
    <div className="space-y-6">
      <div>
        <h2 className="font-mono text-lg mb-2 animate-underline">webstorm plugin</h2>
        <p className="font-mono text-sm text-muted-foreground">
          enhance your jetbrains webstorm ide with lark capabilities
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm animate-underline">plugin installation</CardTitle>
          <CardDescription className="font-mono text-xs">
            install the lark plugin from jetbrains marketplace
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center space-x-2 p-3 bg-muted/50 rounded animate-bg-expand">
            <span className="font-mono text-xs">1.</span>
            <span className="font-mono text-sm">open webstorm preferences</span>
            <Badge variant="outline" className="font-mono text-xs">⌘,</Badge>
          </div>
          <div className="flex items-center space-x-2 p-3 bg-muted/50 rounded animate-bg-expand">
            <span className="font-mono text-xs">2.</span>
            <span className="font-mono text-sm">navigate to plugins</span>
          </div>
          <div className="flex items-center space-x-2 p-3 bg-muted/50 rounded animate-bg-expand">
            <span className="font-mono text-xs">3.</span>
            <span className="font-mono text-sm">search for "lark ai dev"</span>
          </div>
          <div className="flex items-center space-x-2 p-3 bg-muted/50 rounded animate-bg-expand">
            <span className="font-mono text-xs">4.</span>
            <span className="font-mono text-sm">install and restart webstorm</span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm animate-underline">plugin configuration</CardTitle>
          <CardDescription className="font-mono text-xs">
            configure lark plugin settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded p-4 font-mono text-sm animate-bg-expand">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">ai provider</span>
                <select className="px-2 py-1 border border-border rounded bg-background text-sm">
                  <option>local</option>
                  <option>openai</option>
                  <option>anthropic</option>
                </select>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">terminal integration</span>
                <input type="checkbox" className="rounded" defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">charm tools</span>
                <input type="checkbox" className="rounded" defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">markdown preview</span>
                <input type="checkbox" className="rounded" defaultChecked />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm animate-underline">keyboard shortcuts</CardTitle>
          <CardDescription className="font-mono text-xs">
            lark plugin keyboard shortcuts for webstorm
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex items-center justify-between p-2 border rounded">
              <span className="font-mono text-sm">open lark terminal</span>
              <Badge variant="outline" className="font-mono text-xs">⌘⇧T</Badge>
            </div>
            <div className="flex items-center justify-between p-2 border rounded">
              <span className="font-mono text-sm">ask ai about selection</span>
              <Badge variant="outline" className="font-mono text-xs">⌘⇧A</Badge>
            </div>
            <div className="flex items-center justify-between p-2 border rounded">
              <span className="font-mono text-sm">format with gum</span>
              <Badge variant="outline" className="font-mono text-xs">⌘⇧G</Badge>
            </div>
            <div className="flex items-center justify-between p-2 border rounded">
              <span className="font-mono text-sm">render markdown</span>
              <Badge variant="outline" className="font-mono text-xs">⌘⇧M</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderEnvironmentSetup = () => (
    <div className="space-y-6">
      <div>
        <h2 className="font-mono text-lg mb-2 animate-underline">environment configuration</h2>
        <p className="font-mono text-sm text-muted-foreground">
          configure development environment for lark integration
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm animate-underline">environment variables</CardTitle>
          <CardDescription className="font-mono text-xs">
            required environment variables for lark integration
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded p-4 font-mono text-sm animate-bg-expand">
            <div className="flex items-center justify-between mb-2">
              <span className="text-muted-foreground text-xs">.env</span>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => copyToClipboard(`# Lark Configuration
LARK_MODE=development
LARK_AI_PROVIDER=local
LARK_TERMINAL_SHELL=zsh
LARK_CHARM_TOOLS=true

# AI Configuration (optional)
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here

# Terminal Configuration
LARK_TERMINAL_THEME=minimal
LARK_TERMINAL_FONT_SIZE=14

# Editor Configuration
LARK_EDITOR_THEME=light
LARK_EDITOR_SYNTAX_HIGHLIGHT=true

# File Manager Configuration
LARK_FILES_AUTO_SAVE=true
LARK_FILES_BACKUP=true`)}
                className="h-6 px-2 text-xs"
              >
                copy
              </Button>
            </div>
            <pre className="text-foreground">{`# Lark Configuration
LARK_MODE=development
LARK_AI_PROVIDER=local
LARK_TERMINAL_SHELL=zsh
LARK_CHARM_TOOLS=true

# AI Configuration (optional)
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here

# Terminal Configuration
LARK_TERMINAL_THEME=minimal
LARK_TERMINAL_FONT_SIZE=14

# Editor Configuration
LARK_EDITOR_THEME=light
LARK_EDITOR_SYNTAX_HIGHLIGHT=true

# File Manager Configuration
LARK_FILES_AUTO_SAVE=true
LARK_FILES_BACKUP=true`}</pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="font-mono text-sm animate-underline">lark.config.js</CardTitle>
          <CardDescription className="font-mono text-xs">
            main configuration file for lark integration
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded p-4 font-mono text-sm overflow-x-auto animate-bg-expand">
            <div className="flex items-center justify-between mb-2">
              <span className="text-muted-foreground text-xs">javascript</span>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => copyToClipboard(`module.exports = {
  // Core configuration
  mode: process.env.LARK_MODE || 'development',
  
  // AI configuration
  ai: {
    enabled: true,
    provider: process.env.LARK_AI_PROVIDER || 'local',
    models: {
      text: 'gpt-4',
      code: 'codex',
      chat: 'gpt-3.5-turbo'
    }
  },
  
  // Terminal configuration
  terminal: {
    enabled: true,
    shell: process.env.LARK_TERMINAL_SHELL || 'bash',
    charm: {
      gum: true,
      glow: true,
      mods: true,
      pop: true,
      skate: true
    }
  },
  
  // Editor configuration
  editor: {
    enabled: true,
    markdown: {
      preview: true,
      syntax: true,
      export: ['html', 'pdf']
    }
  },
  
  // Integration settings
  integrations: {
    electron: true,
    webstorm: true,
    vscode: false
  }
};`)}
                className="h-6 px-2 text-xs"
              >
                copy
              </Button>
            </div>
            <pre className="text-foreground">{`module.exports = {
  // Core configuration
  mode: process.env.LARK_MODE || 'development',
  
  // AI configuration
  ai: {
    enabled: true,
    provider: process.env.LARK_AI_PROVIDER || 'local',
    models: {
      text: 'gpt-4',
      code: 'codex', 
      chat: 'gpt-3.5-turbo'
    }
  },
  
  // Terminal configuration
  terminal: {
    enabled: true,
    shell: process.env.LARK_TERMINAL_SHELL || 'bash',
    charm: {
      gum: true,
      glow: true,
      mods: true,
      pop: true,
      skate: true
    }
  },
  
  // Editor configuration
  editor: {
    enabled: true,
    markdown: {
      preview: true,
      syntax: true,
      export: ['html', 'pdf']
    }
  },
  
  // Integration settings
  integrations: {
    electron: true,
    webstorm: true,
    vscode: false
  }
};`}</pre>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderAPIReference = () => (
    <div className="space-y-6">
      <div>
        <h2 className="font-mono text-lg mb-2 animate-underline">api reference</h2>
        <p className="font-mono text-sm text-muted-foreground">
          programmatic api for lark integration
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="font-mono text-sm animate-underline">LarkCore</CardTitle>
            <CardDescription className="font-mono text-xs">
              core functionality and state management
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 font-mono text-xs">
              <div className="p-2 bg-muted/50 rounded">
                <code>new LarkCore(config)</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>lark.initialize()</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>lark.getState()</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>lark.setState(state)</code>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="font-mono text-sm animate-underline">LarkTerminal</CardTitle>
            <CardDescription className="font-mono text-xs">
              terminal interface and command execution
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 font-mono text-xs">
              <div className="p-2 bg-muted/50 rounded">
                <code>terminal.execute(cmd)</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>terminal.clear()</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>terminal.history()</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>terminal.charm.gum()</code>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="font-mono text-sm animate-underline">LarkAI</CardTitle>
            <CardDescription className="font-mono text-xs">
              ai assistance and text processing
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 font-mono text-xs">
              <div className="p-2 bg-muted/50 rounded">
                <code>ai.ask(question)</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>ai.process(text)</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>ai.generate(prompt)</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>ai.models.list()</code>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="font-mono text-sm animate-underline">LarkEditor</CardTitle>
            <CardDescription className="font-mono text-xs">
              markdown editing and file management
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 font-mono text-xs">
              <div className="p-2 bg-muted/50 rounded">
                <code>editor.open(file)</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>editor.save(content)</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>editor.render(markdown)</code>
              </div>
              <div className="p-2 bg-muted/50 rounded">
                <code>editor.export(format)</code>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="max-w-6xl mx-auto p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-[rgba(255,255,255,1)] rounded-lg flex items-center justify-center animate-border-grow">
                <span className="text-[rgba(0,0,0,1)] font-mono text-lg">.\</span>
              </div>
              <div>
                <h1 className="text-xl font-mono animate-underline">lark integration</h1>
                <p className="font-mono text-sm text-muted-foreground">
                  developer integration guide v1.0
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="outline" className="font-mono text-xs">
                beta
              </Badge>
              <Button variant="outline" size="sm" className="font-mono text-sm">
                <span className="font-mono text-xs mr-1">gh</span>
                github
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-5 font-mono">
            <TabsTrigger value="overview" className="font-mono text-sm">overview</TabsTrigger>
            <TabsTrigger value="electron" className="font-mono text-sm">electron</TabsTrigger>
            <TabsTrigger value="webstorm" className="font-mono text-sm">webstorm</TabsTrigger>
            <TabsTrigger value="environment" className="font-mono text-sm">environment</TabsTrigger>
            <TabsTrigger value="api" className="font-mono text-sm">api</TabsTrigger>
          </TabsList>

          <div className="mt-6">
            <TabsContent value="overview" className="space-y-6">
              {renderOverview()}
            </TabsContent>

            <TabsContent value="electron" className="space-y-6">
              {renderElectronSetup()}
            </TabsContent>

            <TabsContent value="webstorm" className="space-y-6">
              {renderWebStormSetup()}
            </TabsContent>

            <TabsContent value="environment" className="space-y-6">
              {renderEnvironmentSetup()}
            </TabsContent>

            <TabsContent value="api" className="space-y-6">
              {renderAPIReference()}
            </TabsContent>
          </div>
        </Tabs>
      </div>

      {/* Footer */}
      <div className="border-t border-border bg-card mt-12">
        <div className="max-w-6xl mx-auto p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 font-mono text-sm text-muted-foreground">
              <span>lark dev environment</span>
              <span>•</span>
              <span>integration guide</span>
              <span>•</span>
              <span>v1.0.0</span>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="ghost" size="sm" className="font-mono text-sm">
                <span className="font-mono text-xs mr-1">dc</span>
                docs
              </Button>
              <Button variant="ghost" size="sm" className="font-mono text-sm">
                <span className="font-mono text-xs mr-1">sp</span>
                support
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}