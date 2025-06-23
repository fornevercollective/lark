# Lark Design System

> **Philosophy**: Terminal-centric, minimal, and functional. Every interaction should feel precise and intentional, like commanding a sophisticated development environment.

## Core Principles

### 1. Terminal-First Design
- **Monospace Typography**: All text uses monospace fonts for consistency with terminal aesthetics
- **2-Letter Codes**: Icons are represented as 2-letter monospace codes (`ed`, `tm`, `ai`, etc.)
- **Command-Style Interactions**: Actions feel like terminal commands with immediate feedback
- **Minimal Visual Noise**: Clean lines, subtle borders, and purposeful spacing

### 2. Responsive Animations
- **Micro-interactions**: Every hover, click, and state change has subtle animation feedback
- **Terminal-Style Transitions**: Animations mimic terminal behaviors (line drawing, text appearing, etc.)
- **Performance-First**: All animations use CSS transforms and opacity for optimal performance

### 3. Adaptive Theming
- **Three-Mode System**: Light, Dark, and Auto (follows system preference)
- **Semantic Color Variables**: Colors adapt contextually across themes
- **Consistent Contrast**: Maintains accessibility standards in all theme modes

---

## Typography

### Font Stack
```css
font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
```

### Scale & Hierarchy
- **Base Size**: 14px (customizable via `--font-size`)
- **H1**: `text-2xl` + `font-medium` (Large headings)
- **H2**: `text-xl` + `font-medium` (Section headings)  
- **H3**: `text-lg` + `font-medium` (Subsection headings)
- **H4**: `text-base` + `font-medium` (Component headings)
- **Body**: `text-base` + `font-normal` (Default text)
- **Small**: `text-sm` + `font-mono` (UI labels, metadata)
- **Tiny**: `text-xs` + `font-mono` (Status, icons, codes)

### Typography Classes
```css
.font-mono         /* Monospace font family */
.text-xs           /* 12px - Icon codes, status text */
.text-sm           /* 14px - UI labels, descriptions */
.text-base         /* 16px - Body text, buttons */
.text-lg           /* 18px - Subsection headings */
.text-xl           /* 20px - Section headings */
.text-2xl          /* 24px - Page headings */
```

---

## Color System

### Semantic Color Variables

#### Light Theme
```css
--background: #ffffff           /* Primary background */
--foreground: oklch(0.145 0 0)  /* Primary text */
--card: #ffffff                 /* Card backgrounds */
--muted: #ececf0               /* Subtle backgrounds */
--muted-foreground: #717182     /* Secondary text */
--border: rgba(0, 0, 0, 0.1)   /* Subtle borders */
--accent: #e9ebef              /* Hover states */
```

#### Dark Theme
```css
--background: oklch(0.145 0 0)  /* Dark background */
--foreground: oklch(0.985 0 0)  /* Light text */
--card: oklch(0.145 0 0)       /* Dark card backgrounds */
--muted: oklch(0.269 0 0)      /* Dark subtle backgrounds */
--border: oklch(0.269 0 0)     /* Dark borders */
--accent: oklch(0.269 0 0)     /* Dark hover states */
```

### Usage Guidelines
- **Foreground/Background**: Primary text and background pairing
- **Muted**: Secondary text, placeholders, inactive states
- **Border**: Subtle divisions, input borders, card outlines
- **Accent**: Hover states, selected items, active elements
- **Card**: Elevated surfaces, modals, panels

---

## Component System

### Buttons

#### Variants
```typescript
type ButtonVariant = 'default' | 'ghost' | 'outline' | 'secondary'
```

#### Sizes
```typescript
type ButtonSize = 'sm' | 'default' | 'lg'
```

#### Terminal-Style Button Pattern
```tsx
<Button variant="ghost" size="sm" className="h-8 px-2 animate-accent-line">
  <span className="font-mono text-xs">ed</span>
</Button>
```

### Input Components

#### Standard Input
```tsx
<Input 
  className="font-mono text-sm" 
  placeholder="terminal-style placeholder..."
/>
```

#### Search Input Pattern
```tsx
<input
  className="w-64 px-3 py-1 bg-input border border-border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-ring font-mono animate-border-grow"
  placeholder="Search files, content, commands..."
/>
```

### Cards

#### Standard Card Pattern
```tsx
<Card className="cursor-pointer hover:bg-muted/50 transition-colors animate-terminal-fade">
  <CardHeader className="pb-3">
    <CardTitle className="text-sm font-mono truncate animate-underline">
      Title
    </CardTitle>
    <CardDescription className="text-xs font-mono mt-1">
      Metadata
    </CardDescription>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
</Card>
```

---

## Animation System

### Core Animation Classes

#### Hover Effects
```css
.animate-underline      /* Sliding underline on hover */
.animate-accent-line    /* Vertical accent line on hover */
.animate-border-grow    /* Growing border effect */
.animate-bg-expand      /* Expanding background */
```

#### Terminal-Style Animations
```css
.animate-terminal-line  /* Terminal prompt line extension */
.animate-pulse-dot      /* Pulsing status indicator */
.animate-slide-indicator /* Sliding active indicator */
.warp-section-highlight /* Section highlight with transform */
```

#### Transition Animations
```css
.terminal-section-fade  /* Fade-in for panels */
.terminal-line-in      /* Slide-in for terminal text */
.animate-terminal-fade  /* Standard fade for components */
```

### Animation Usage Guidelines

1. **Micro-interactions**: Every interactive element should have hover feedback
2. **State Changes**: Use fade transitions for showing/hiding content
3. **Performance**: All animations use `transform` and `opacity` properties
4. **Duration**: Keep animations between 150-400ms for responsiveness
5. **Easing**: Use `cubic-bezier(0.4, 0, 0.2, 1)` for smooth, natural motion

---

## Layout Patterns

### Application Shell
```tsx
<div className="min-h-screen w-full flex flex-col bg-background">
  <div className="flex-1 flex max-w-screen-2xl mx-auto w-full">
    {/* Sidebar */}
    <div className="w-64 border-r border-border bg-sidebar">
      {/* Navigation */}
    </div>
    
    {/* Main Content */}
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="h-14 border-b border-border bg-card">
        {/* Header content */}
      </div>
      
      {/* Content Area */}
      <div className="flex-1 overflow-y-auto bg-background">
        {/* Main content */}
      </div>
    </div>
  </div>
  
  {/* Footer */}
  <div className="h-12 border-t border-border bg-card">
    {/* Footer controls */}
  </div>
</div>
```

### Panel Patterns

#### Popup Panel
```tsx
<div className="absolute bottom-12 right-4 w-64 bg-card border border-border rounded-lg shadow-lg p-4 z-50 terminal-section-fade">
  <div className="flex items-center justify-between mb-3">
    <h3 className="font-medium text-sm font-mono animate-underline">Panel Title</h3>
    <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
      <span className="font-mono text-xs">×</span>
    </Button>
  </div>
  {/* Panel content */}
</div>
```

#### Overlay Panel
```tsx
<div className="absolute inset-x-4 bottom-4 top-20 bg-card/95 backdrop-blur-sm border border-border rounded-lg shadow-xl z-50 terminal-section-fade">
  {/* Overlay content */}
</div>
```

---

## Iconography

### 2-Letter Code System

#### Core Application Icons
```typescript
const ICON_CODES = {
  // Primary modes
  ed: 'Editor',      // Markdown editing
  tm: 'Terminal',    // Command interface  
  ai: 'AI Chat',     // AI conversation
  fl: 'Files',       // File management
  tp: 'Templates',   // Template system
  cx: 'Codex',       // Bookmark manager
  st: 'Settings',    // Preferences
  
  // UI Controls
  pl: 'Panel Left',  // Sidebar toggle
  pb: 'Panel Bottom', // Bottom panel
  sr: 'Search',      // Search function
  cm: 'Commands',    // Quick actions
  hs: 'History',     // Recent files
  ly: 'Layers',      // Layer panel
  
  // Theme & Media
  lt: 'Light',       // Light theme
  dk: 'Dark',        // Dark theme  
  au: 'Auto',        // Auto theme
  wv: 'Waveform',    // Audio waveform
  sp: 'Speak',       // Audio input
} as const;
```

### Icon Usage Guidelines
1. **Consistency**: Always use 2-letter codes, never single letters or longer abbreviations
2. **Context**: Codes should be intuitive within the terminal/development context
3. **Spacing**: Icons have consistent spacing and sizing (`text-xs` class)
4. **Contrast**: Icons inherit text color and work across all themes

---

## Scrollbar System

### Subtle Scrollbar (10% Visibility)
```css
.subtle-scrollbar {
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 0, 0, 0.1) transparent;
}

.subtle-scrollbar::-webkit-scrollbar {
  width: 8px;
}

.subtle-scrollbar::-webkit-scrollbar-thumb {
  background-color: rgba(0, 0, 0, 0.1);
  border-radius: 4px;
  transition: background-color 0.2s ease;
}
```

### Usage
Apply `.subtle-scrollbar` to any scrollable container for consistent, minimal scrollbar styling that works across themes.

---

## Terminal-Specific Patterns

### Command Input Pattern
```tsx
<div className="flex items-center space-x-2">
  <span className="text-muted-foreground shrink-0">❯</span>
  <input
    className="flex-1 bg-transparent border-none outline-none font-mono"
    placeholder="Enter command..."
  />
</div>
```

### Status Line Pattern
```tsx
<div className="flex items-center space-x-4 text-xs text-muted-foreground">
  <span className="font-mono animate-pulse-dot">Ready</span>
  <span className="font-mono">•</span>
  <span className="font-mono animate-underline">{mode} mode</span>
</div>
```

### Terminal Output Pattern
```tsx
<div className="space-y-1">
  {lines.map((line) => (
    <div key={line.id} className={`whitespace-pre-wrap font-mono text-sm leading-relaxed terminal-line-in ${getLineColor(line.type)}`}>
      {line.content}
    </div>
  ))}
</div>
```

---

## Theme System

### Theme Toggle Implementation
```tsx
const [theme, setTheme] = useState<'light' | 'dark' | 'auto'>('auto');

const toggleTheme = () => {
  const modes = ['light', 'dark', 'auto'] as const;
  const currentIndex = modes.indexOf(theme);
  const nextIndex = (currentIndex + 1) % modes.length;
  setTheme(modes[nextIndex]);
};

// Theme application
useEffect(() => {
  const root = document.documentElement;
  
  if (theme === 'dark') {
    root.classList.add('dark');
  } else if (theme === 'light') {
    root.classList.remove('dark');
  } else {
    // Auto mode follows system preference
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    root.classList.toggle('dark', mediaQuery.matches);
  }
}, [theme]);
```

### Theme-Aware Components
Components automatically adapt to theme changes through CSS custom properties. No additional theme-specific code is needed in components.

---

## Accessibility Guidelines

### Keyboard Navigation
- All interactive elements must be keyboard accessible
- Focus indicators should be visible and consistent
- Tab order should be logical and predictable

### Screen Readers
- Use semantic HTML elements
- Provide appropriate ARIA labels for custom components
- Ensure sufficient color contrast in all themes

### Motion Preferences
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation: none !important;
    transition: none !important;
  }
}
```

---

## Development Guidelines

### Component Creation
1. **Start with shadcn/ui**: Use existing components as base when possible
2. **Apply Terminal Patterns**: Add appropriate animation classes and monospace styling
3. **Theme Compatibility**: Test in all three theme modes
4. **Responsive Design**: Ensure components work on mobile and desktop

### Animation Implementation
1. **Use CSS Classes**: Prefer predefined animation classes over inline styles
2. **Performance**: Only animate `transform` and `opacity` properties
3. **Timing**: Keep animations under 400ms for responsiveness
4. **Fallbacks**: Provide reduced-motion alternatives

### Code Style
```tsx
// ✅ Good: Terminal-style component
<Button variant="ghost" size="sm" className="h-8 px-2 animate-accent-line">
  <span className="font-mono text-xs">ed</span>
</Button>

// ❌ Avoid: Non-terminal styling
<Button className="bg-blue-500 text-white rounded-full">
  Editor 📝
</Button>
```

---

## File Organization

### Component Structure
```
/components
  /ui              # shadcn/ui components (customized)
  /figma           # Figma-specific components
  LarkEditor.tsx   # Main editor component
  Terminal.tsx     # Terminal interface
  BookmarkManager.tsx  # Codex system
  ...
```

### Styling Structure
```
/styles
  globals.css      # Global styles, theme variables, animations
```

### Asset Organization
- **Images**: Use ImageWithFallback component for consistency
- **Icons**: 2-letter text codes only, no icon files
- **Fonts**: System monospace fonts, no custom font files

---

## Testing Guidelines

### Visual Testing
- Test all components in Light, Dark, and Auto themes
- Verify animations work smoothly across browsers
- Check responsive behavior on mobile and desktop

### Interaction Testing  
- Verify all hover states work correctly
- Test keyboard navigation flows
- Ensure touch interactions work on mobile

### Performance Testing
- Monitor animation performance with dev tools
- Check for layout shifts during theme transitions
- Verify smooth scrolling with subtle scrollbars

---

## Future Considerations

### Planned Enhancements
- **Command Palette**: Global keyboard shortcut system
- **Plugin System**: Extensible component architecture  
- **Advanced Theming**: Custom color scheme support
- **Mobile Optimization**: Touch-first interaction patterns

### Maintenance
- **Regular Audits**: Review components for consistency
- **Performance Monitoring**: Track animation performance
- **Accessibility Testing**: Regular WCAG compliance checks
- **Documentation Updates**: Keep design system current with changes

---

*This design system is a living document. Update it as the Lark interface evolves to maintain consistency and quality across the entire application.*