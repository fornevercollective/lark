import React, { useState, useEffect } from 'react';

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

type AudioActivity = 'idle' | 'talking' | 'ai-speaking' | 'listening' | 'processing';

interface LarkWaveformProps {
  isActive: boolean;
  activity: AudioActivity;
  settings: AudioSettings;
}

export function LarkWaveform({ isActive, activity, settings }: LarkWaveformProps) {
  const [currentPhraseIndex, setCurrentPhraseIndex] = useState(0);
  const [waveformBars, setWaveformBars] = useState<number[]>([]);

  // Terminal-style idle phrases
  const idlePhrases = [
    'ready',
    'listening ...',
    'standby',
    'awaiting input',
    'terminal ready',
    'system active',
    'development mode',
    'workspace ready'
  ];

  // Phrase rotation for idle state
  useEffect(() => {
    if (!settings.phraseRotationEnabled || activity !== 'idle') return;

    const interval = setInterval(() => {
      setCurrentPhraseIndex(prev => (prev + 1) % idlePhrases.length);
    }, settings.phraseRotationSpeed * 1000);

    return () => clearInterval(interval);
  }, [settings.phraseRotationEnabled, settings.phraseRotationSpeed, activity, idlePhrases.length]);

  // Waveform animation for active states
  useEffect(() => {
    if (!isActive || activity === 'idle') {
      setWaveformBars([]);
      return;
    }

    const generateBars = () => {
      const barCount = settings.waveformStyle === 'minimal' ? 3 : 
                      settings.waveformStyle === 'standard' ? 5 : 8;
      
      const bars = Array.from({ length: barCount }, () => {
        const baseHeight = activity === 'talking' ? 0.8 :
                          activity === 'ai-speaking' ? 0.6 :
                          activity === 'listening' ? 0.4 :
                          activity === 'processing' ? 0.7 : 0.3;
        
        const variation = (Math.random() - 0.5) * 0.4;
        return Math.max(0.1, Math.min(1, baseHeight + variation));
      });
      
      setWaveformBars(bars);
    };

    generateBars();
    const interval = setInterval(generateBars, 150 / settings.animationSpeed);

    return () => clearInterval(interval);
  }, [isActive, activity, settings.waveformStyle, settings.animationSpeed]);

  // Get activity display text
  const getActivityDisplay = () => {
    if (activity === 'idle') {
      return settings.phraseRotationEnabled ? idlePhrases[currentPhraseIndex] : 'idle state';
    }
    
    // Show (.audio) for any active audio state
    return '(.audio)';
  };

  // Get activity-specific styling
  const getActivityStyling = () => {
    if (activity === 'idle') {
      return 'text-sidebar-foreground/70';
    }
    
    // Active audio states get emphasized styling
    switch (activity) {
      case 'talking':
        return 'text-green-400';
      case 'ai-speaking':
        return 'text-blue-400';
      case 'listening':
        return 'text-yellow-400';
      case 'processing':
        return 'text-purple-400';
      default:
        return 'text-sidebar-foreground';
    }
  };

  // Render waveform bars
  const renderWaveform = () => {
    if (!isActive || activity === 'idle' || waveformBars.length === 0) {
      return null;
    }

    const colorClass = settings.waveformColor === 'accent' ? 'bg-accent' :
                      settings.waveformColor === 'primary' ? 'bg-primary' :
                      'bg-current';

    return (
      <div className="flex items-end space-x-1 h-4 ml-2">
        {waveformBars.map((height, index) => (
          <div
            key={index}
            className={`w-1 rounded-sm ${colorClass} waveform-bar opacity-70`}
            style={{
              height: `${height * 100}%`,
              animationDelay: `${index * 0.1}s`,
              animationDuration: `${1.2 / settings.animationSpeed}s`
            }}
          />
        ))}
      </div>
    );
  };

  if (!isActive) return null;

  return (
    <div className="flex items-center">
      <span 
        key={activity === 'idle' ? currentPhraseIndex : activity} 
        className={`font-mono text-xs animate-terminal-fade ${getActivityStyling()}`}
      >
        {getActivityDisplay()}
      </span>
      {renderWaveform()}
    </div>
  );
}