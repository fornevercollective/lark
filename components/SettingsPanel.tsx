import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Switch } from './ui/switch';
import { Separator } from './ui/separator';
import { Slider } from './ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface Settings {
  theme: 'light' | 'dark' | 'auto';
  fontSize: number;
  fontFamily: string;
  autoSave: boolean;
  notifications: boolean;
  shortcuts: boolean;
  analytics: boolean;
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

interface SettingsPanelProps {
  className?: string;
}

export function SettingsPanel({ className = "" }: SettingsPanelProps) {
  const [settings, setSettings] = useState<Settings>({
    theme: 'auto',
    fontSize: 14,
    fontFamily: 'monospace',
    autoSave: true,
    notifications: true,
    shortcuts: true,
    analytics: false,
    waveformEnabled: true,
    sensitivity: 0.8,
    animationSpeed: 1.0,
    phraseRotationEnabled: true,
    phraseRotationSpeed: 3,
    autoActivityDetection: true,
    visualFeedbackEnabled: true,
    waveformStyle: 'minimal',
    waveformColor: 'default'
  });

  const [activeTab, setActiveTab] = useState('general');

  useEffect(() => {
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('lark-settings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  useEffect(() => {
    // Save settings to localStorage
    localStorage.setItem('lark-settings', JSON.stringify(settings));
  }, [settings]);

  const updateSetting = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const resetSettings = () => {
    const defaultSettings: Settings = {
      theme: 'auto',
      fontSize: 14,
      fontFamily: 'monospace',
      autoSave: true,
      notifications: true,
      shortcuts: true,
      analytics: false,
      waveformEnabled: true,
      sensitivity: 0.8,
      animationSpeed: 1.0,
      phraseRotationEnabled: true,
      phraseRotationSpeed: 3,
      autoActivityDetection: true,
      visualFeedbackEnabled: true,
      waveformStyle: 'minimal',
      waveformColor: 'default'
    };
    setSettings(defaultSettings);
  };

  return (
    <div className={`space-y-6 font-mono ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary/10 rounded flex items-center justify-center">
            <span className="font-mono text-xs text-primary">st</span>
          </div>
          <div>
            <h3 className="font-medium">Settings</h3>
            <p className="text-sm text-muted-foreground">Configure Lark preferences</p>
          </div>
        </div>
        
        <Button
          variant="outline"
          size="sm"
          onClick={resetSettings}
          className="font-mono text-xs warp-section-highlight"
        >
          <span className="mr-1">rs</span>
          Reset
        </Button>
      </div>

      {/* Settings Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="general" className="font-mono text-xs">
            <span className="mr-1">gn</span>
            General
          </TabsTrigger>
          <TabsTrigger value="appearance" className="font-mono text-xs">
            <span className="mr-1">ap</span>
            Appearance
          </TabsTrigger>
          <TabsTrigger value="audio" className="font-mono text-xs">
            <span className="mr-1">au</span>
            Audio
          </TabsTrigger>
          <TabsTrigger value="advanced" className="font-mono text-xs">
            <span className="mr-1">ad</span>
            Advanced
          </TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-4">
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="font-mono text-sm flex items-center space-x-2">
                <span className="text-blue-500">gn</span>
                <span>General Settings</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-mono text-sm">Auto Save</div>
                    <div className="text-xs text-muted-foreground">Automatically save changes</div>
                  </div>
                  <Switch
                    checked={settings.autoSave}
                    onCheckedChange={(checked) => updateSetting('autoSave', checked)}
                  />
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-mono text-sm">Notifications</div>
                    <div className="text-xs text-muted-foreground">Show system notifications</div>
                  </div>
                  <Switch
                    checked={settings.notifications}
                    onCheckedChange={(checked) => updateSetting('notifications', checked)}
                  />
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-mono text-sm">Keyboard Shortcuts</div>
                    <div className="text-xs text-muted-foreground">Enable keyboard shortcuts</div>
                  </div>
                  <Switch
                    checked={settings.shortcuts}
                    onCheckedChange={(checked) => updateSetting('shortcuts', checked)}
                  />
                </div>
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-mono text-sm">Analytics</div>
                    <div className="text-xs text-muted-foreground">Share usage analytics</div>
                  </div>
                  <Switch
                    checked={settings.analytics}
                    onCheckedChange={(checked) => updateSetting('analytics', checked)}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="appearance" className="space-y-4">
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="font-mono text-sm flex items-center space-x-2">
                <span className="text-purple-500">ap</span>
                <span>Appearance</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="font-mono text-sm">Theme</div>
                <div className="grid grid-cols-3 gap-2">
                  {['light', 'dark', 'auto'].map(theme => (
                    <Button
                      key={theme}
                      variant={settings.theme === theme ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => updateSetting('theme', theme as 'light' | 'dark' | 'auto')}
                      className="font-mono text-xs warp-section-highlight"
                    >
                      {theme === 'light' && 'lt'}
                      {theme === 'dark' && 'dk'}
                      {theme === 'auto' && 'au'}
                      <span className="ml-1 capitalize">{theme}</span>
                    </Button>
                  ))}
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="font-mono text-sm">Font Size</div>
                <div className="flex items-center space-x-4">
                  <Slider
                    value={[settings.fontSize]}
                    onValueChange={([value]) => updateSetting('fontSize', value)}
                    min={10}
                    max={20}
                    step={1}
                    className="flex-1"
                  />
                  <Badge variant="outline" className="font-mono text-xs">
                    {settings.fontSize}px
                  </Badge>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="font-mono text-sm">Font Family</div>
                <select
                  value={settings.fontFamily}
                  onChange={(e) => updateSetting('fontFamily', e.target.value)}
                  className="w-full font-mono text-xs bg-background border border-border rounded px-3 py-2"
                >
                  <option value="monospace">Monospace</option>
                  <option value="'Fira Code', monospace">Fira Code</option>
                  <option value="'JetBrains Mono', monospace">JetBrains Mono</option>
                  <option value="'Source Code Pro', monospace">Source Code Pro</option>
                </select>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audio" className="space-y-4">
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="font-mono text-sm flex items-center space-x-2">
                <span className="text-green-500">au</span>
                <span>Audio Settings</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-sm">Waveform</div>
                  <div className="text-xs text-muted-foreground">Enable audio waveform</div>
                </div>
                <Switch
                  checked={settings.waveformEnabled}
                  onCheckedChange={(checked) => updateSetting('waveformEnabled', checked)}
                />
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="font-mono text-sm">Sensitivity</div>
                <div className="flex items-center space-x-4">
                  <Slider
                    value={[settings.sensitivity]}
                    onValueChange={([value]) => updateSetting('sensitivity', value)}
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    className="flex-1"
                  />
                  <Badge variant="outline" className="font-mono text-xs">
                    {(settings.sensitivity * 100).toFixed(0)}%
                  </Badge>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="font-mono text-sm">Animation Speed</div>
                <div className="flex items-center space-x-4">
                  <Slider
                    value={[settings.animationSpeed]}
                    onValueChange={([value]) => updateSetting('animationSpeed', value)}
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    className="flex-1"
                  />
                  <Badge variant="outline" className="font-mono text-xs">
                    {settings.animationSpeed.toFixed(1)}x
                  </Badge>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="font-mono text-sm">Waveform Style</div>
                <div className="grid grid-cols-3 gap-2">
                  {['minimal', 'standard', 'detailed'].map(style => (
                    <Button
                      key={style}
                      variant={settings.waveformStyle === style ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => updateSetting('waveformStyle', style as 'minimal' | 'standard' | 'detailed')}
                      className="font-mono text-xs warp-section-highlight"
                    >
                      {style.slice(0, 3)}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="advanced" className="space-y-4">
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="font-mono text-sm flex items-center space-x-2">
                <span className="text-red-500">ad</span>
                <span>Advanced Settings</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-sm">Auto Activity Detection</div>
                  <div className="text-xs text-muted-foreground">Automatically detect user activity</div>
                </div>
                <Switch
                  checked={settings.autoActivityDetection}
                  onCheckedChange={(checked) => updateSetting('autoActivityDetection', checked)}
                />
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-sm">Visual Feedback</div>
                  <div className="text-xs text-muted-foreground">Show visual activity indicators</div>
                </div>
                <Switch
                  checked={settings.visualFeedbackEnabled}
                  onCheckedChange={(checked) => updateSetting('visualFeedbackEnabled', checked)}
                />
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="font-mono text-sm">Phrase Rotation Speed</div>
                <div className="flex items-center space-x-4">
                  <Slider
                    value={[settings.phraseRotationSpeed]}
                    onValueChange={([value]) => updateSetting('phraseRotationSpeed', value)}
                    min={1}
                    max={10}
                    step={1}
                    className="flex-1"
                  />
                  <Badge variant="outline" className="font-mono text-xs">
                    {settings.phraseRotationSpeed}s
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* System Information */}
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="font-mono text-sm flex items-center space-x-2">
                <span className="text-yellow-500">sy</span>
                <span>System Information</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="font-mono text-muted-foreground">Version</div>
                  <div className="font-mono">1.0.0</div>
                </div>
                <div>
                  <div className="font-mono text-muted-foreground">Platform</div>
                  <div className="font-mono">Web</div>
                </div>
                <div>
                  <div className="font-mono text-muted-foreground">Runtime</div>
                  <div className="font-mono">Browser</div>
                </div>
                <div>
                  <div className="font-mono text-muted-foreground">Build</div>
                  <div className="font-mono">Production</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}