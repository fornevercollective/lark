import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Separator } from './ui/separator';
import { Progress } from './ui/progress';

interface DesktopApp {
  id: string;
  name: string;
  description: string;
  platform: 'electron' | 'swift' | 'web' | 'fiddle' | 'webstorm' | 'xcode';
  icon: string;
  available: boolean;
  launchUrl?: string;
  requirements?: string[];
  features: string[];
}

interface DesktopLauncherProps {
  className?: string;
  compact?: boolean;
}

export function DesktopLauncher({ className = "", compact = false }: DesktopLauncherProps) {
  const [isLauncherOpen, setIsLauncherOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('electron');
  const [launchStatus, setLaunchStatus] = useState<Record<string, 'idle' | 'launching' | 'success' | 'error'>>({});
  const [platformSupport, setPlatformSupport] = useState<Record<string, boolean>>({});

  const desktopApps: DesktopApp[] = [
    {
      id: 'electron-app',
      name: 'Lark Desktop',
      description: 'Full-featured Electron application with native OS integration',
      platform: 'electron',
      icon: '⚡',
      available: true,
      launchUrl: 'lark://launch/electron',
      requirements: ['Electron Runtime', 'Node.js 18+'],
      features: ['Native file system', 'System notifications', 'Menu bar integration', 'Auto-updater']
    },
    {
      id: 'electron-fiddle',
      name: 'Electron Fiddle',
      description: 'Run Lark in Electron Fiddle for development and testing',
      platform: 'fiddle',
      icon: '🧪',
      available: true,
      launchUrl: 'electron-fiddle://import?gist=lark-figma-app',
      requirements: ['Electron Fiddle'],
      features: ['Live code editing', 'Hot reload', 'Debugging tools', 'Package testing']
    },
    {
      id: 'swift-app',
      name: 'Lark for macOS',
      description: 'Native Swift application optimized for macOS',
      platform: 'swift',
      icon: '🍎',
      available: true,
      launchUrl: 'lark-macos://launch',
      requirements: ['macOS 12+', 'Swift Runtime'],
      features: ['SwiftUI interface', 'Touch Bar support', 'Spotlight integration', 'iCloud sync']
    },
    {
      id: 'xcode-project',
      name: 'Xcode Project',
      description: 'Open as Xcode project for iOS/macOS development',
      platform: 'xcode',
      icon: '🔨',
      available: true,
      launchUrl: 'xcode://clone?repo=https://github.com/fornevercollective/lark/lark-ios',
      requirements: ['Xcode 15+', 'iOS 17+ / macOS 14+'],
      features: ['Source code access', 'iOS deployment', 'Interface Builder', 'Instruments profiling']
    },
    {
      id: 'webstorm-project',
      name: 'WebStorm Project',
      description: 'Open in JetBrains WebStorm for web development',
      platform: 'webstorm',
      icon: '💎',
      available: true,
      launchUrl: 'webstorm://open?url=https://github.com/fornevercollective//lark/lark-web',
      requirements: ['WebStorm 2023.3+', 'Node.js 18+'],
      features: ['Intelligent coding assistance', 'Debugger', 'Version control', 'Database tools']
    },
    {
      id: 'web-app',
      name: 'Progressive Web App',
      description: 'Install as PWA for offline access and native-like experience',
      platform: 'web',
      icon: '🌐',
      available: true,
      requirements: ['Modern browser', 'Service Worker support'],
      features: ['Offline functionality', 'Push notifications', 'App-like experience', 'Auto-updates']
    }
  ];

  // Platform detection
  useEffect(() => {
    const detectPlatform = () => {
      const ua = navigator.userAgent;
      const platform = navigator.platform;
      
      const support = {
        electron: true, // Always available via download
        swift: platform.includes('Mac'),
        xcode: platform.includes('Mac'),
        webstorm: true, // Cross-platform
        fiddle: true, // Cross-platform
        web: 'serviceWorker' in navigator
      };
      
      setPlatformSupport(support);
    };

    detectPlatform();
  }, []);

  const handleLaunch = async (app: DesktopApp) => {
    setLaunchStatus(prev => ({ ...prev, [app.id]: 'launching' }));

    try {
      if (app.platform === 'web') {
        // PWA installation
        if ('serviceWorker' in navigator) {
          await navigator.serviceWorker.register('/sw.js');
        }
        // Trigger PWA install prompt if available
        if ((window as any).deferredPrompt) {
          (window as any).deferredPrompt.prompt();
        }
      } else if (app.launchUrl) {
        // Try to open the desktop app URL
        const link = document.createElement('a');
        link.href = app.launchUrl;
        link.click();
      }

      // Simulate launch delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setLaunchStatus(prev => ({ ...prev, [app.id]: 'success' }));
      
      // Reset status after delay
      setTimeout(() => {
        setLaunchStatus(prev => ({ ...prev, [app.id]: 'idle' }));
      }, 3000);

    } catch (error) {
      console.error('Launch failed:', error);
      setLaunchStatus(prev => ({ ...prev, [app.id]: 'error' }));
      
      // Reset status after delay
      setTimeout(() => {
        setLaunchStatus(prev => ({ ...prev, [app.id]: 'idle' }));
      }, 3000);
    }
  };

  const getStatusIcon = (appId: string) => {
    const status = launchStatus[appId] || 'idle';
    switch (status) {
      case 'launching': return '⏳';
      case 'success': return '✅';
      case 'error': return '❌';
      default: return '';
    }
  };

  const getStatusText = (appId: string) => {
    const status = launchStatus[appId] || 'idle';
    switch (status) {
      case 'launching': return 'Launching...';
      case 'success': return 'Launched!';
      case 'error': return 'Failed to launch';
      default: return '';
    }
  };

  const renderAppCard = (app: DesktopApp) => (
    <Card key={app.id} className="border-border/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-lg">
              {app.icon}
            </div>
            <div>
              <CardTitle className="font-mono text-sm animate-underline">
                {app.name}
              </CardTitle>
              <p className="text-xs text-muted-foreground font-mono mt-1">
                {app.description}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {platformSupport[app.platform] ? (
              <Badge variant="outline" className="text-xs font-mono text-green-600">
                Compatible
              </Badge>
            ) : (
              <Badge variant="outline" className="text-xs font-mono text-yellow-600">
                Limited
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Requirements */}
        {app.requirements && (
          <div>
            <h4 className="font-mono text-xs text-muted-foreground mb-2">Requirements</h4>
            <div className="flex flex-wrap gap-1">
              {app.requirements.map(req => (
                <Badge key={req} variant="secondary" className="text-xs font-mono">
                  {req}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Features */}
        <div>
          <h4 className="font-mono text-xs text-muted-foreground mb-2">Features</h4>
          <div className="grid grid-cols-2 gap-1 text-xs font-mono">
            {app.features.slice(0, 4).map(feature => (
              <div key={feature} className="flex items-center space-x-1">
                <span className="text-green-500">•</span>
                <span>{feature}</span>
              </div>
            ))}
          </div>
        </div>

        <Separator />

        {/* Launch Button */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-xs font-mono">
            <span className="text-muted-foreground">Status:</span>
            <span className="flex items-center space-x-1">
              <span>{getStatusIcon(app.id)}</span>
              <span>{getStatusText(app.id) || 'Ready'}</span>
            </span>
          </div>
          
          <Button
            size="sm"
            onClick={() => handleLaunch(app)}
            disabled={launchStatus[app.id] === 'launching'}
            className="font-mono text-xs warp-section-highlight"
          >
            {launchStatus[app.id] === 'launching' ? (
              <>
                <span className="mr-1 animate-spin">⚡</span>
                Launch
              </>
            ) : (
              <>
                <span className="mr-1">{app.icon}</span>
                Launch
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  if (compact) {
    return (
      <Dialog open={isLauncherOpen} onOpenChange={setIsLauncherOpen}>
        <DialogTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2 animate-accent-line font-mono text-xs"
            title="Launch Desktop App"
          >
            <span className="mr-1">🚀</span>
            <span>Desktop</span>
          </Button>
        </DialogTrigger>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="font-mono text-sm animate-underline">
              🚀 Launch Desktop Version
            </DialogTitle>
            <p className="text-sm text-muted-foreground font-mono">
              Run Lark natively on your desktop with enhanced features and performance
            </p>
          </DialogHeader>
          
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="electron" className="font-mono text-xs">
                <span className="mr-1">⚡</span>
                Electron
              </TabsTrigger>
              <TabsTrigger value="native" className="font-mono text-xs">
                <span className="mr-1">🍎</span>
                Native
              </TabsTrigger>
              <TabsTrigger value="dev" className="font-mono text-xs">
                <span className="mr-1">🛠️</span>
                Dev Tools
              </TabsTrigger>
            </TabsList>

            <TabsContent value="electron" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {desktopApps
                  .filter(app => ['electron', 'fiddle', 'web'].includes(app.platform))
                  .map(renderAppCard)}
              </div>
            </TabsContent>

            <TabsContent value="native" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {desktopApps
                  .filter(app => ['swift', 'xcode'].includes(app.platform))
                  .map(renderAppCard)}
              </div>
            </TabsContent>

            <TabsContent value="dev" className="space-y-4">
              <div className="grid grid-cols-1 gap-4">
                {desktopApps
                  .filter(app => ['webstorm', 'fiddle'].includes(app.platform))
                  .map(renderAppCard)}
              </div>
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <div className={`space-y-6 font-mono ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary/10 rounded flex items-center justify-center">
            <span className="font-mono text-xs text-primary">🚀</span>
          </div>
          <div>
            <h3 className="font-medium">Desktop Launcher</h3>
            <p className="text-sm text-muted-foreground">Launch Lark in desktop environments</p>
          </div>
        </div>
      </div>

      {/* Platform Status */}
      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="font-mono text-sm flex items-center space-x-2">
            <span className="text-blue-500">🖥️</span>
            <span>Platform Compatibility</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(platformSupport).map(([platform, supported]) => (
              <div key={platform} className="flex items-center justify-between p-2 rounded border">
                <span className="font-mono text-sm capitalize">{platform}</span>
                <Badge variant={supported ? 'default' : 'secondary'} className="text-xs font-mono">
                  {supported ? '✅ Ready' : '⚠️ Limited'}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Desktop Apps */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="electron" className="font-mono text-xs">
            <span className="mr-1">⚡</span>
            Electron
          </TabsTrigger>
          <TabsTrigger value="native" className="font-mono text-xs">
            <span className="mr-1">🍎</span>
            Native
          </TabsTrigger>
          <TabsTrigger value="dev" className="font-mono text-xs">
            <span className="mr-1">🛠️</span>
            Dev Tools
          </TabsTrigger>
        </TabsList>

        <TabsContent value="electron" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {desktopApps
              .filter(app => ['electron', 'fiddle', 'web'].includes(app.platform))
              .map(renderAppCard)}
          </div>
        </TabsContent>

        <TabsContent value="native" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {desktopApps
              .filter(app => ['swift', 'xcode'].includes(app.platform))
              .map(renderAppCard)}
          </div>
        </TabsContent>

        <TabsContent value="dev" className="space-y-4">
          <div className="grid grid-cols-1 gap-4">
            {desktopApps
              .filter(app => ['webstorm', 'fiddle'].includes(app.platform))
              .map(renderAppCard)}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}