import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Separator } from './ui/separator';

interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  language: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  tags: string[];
  content: string;
  created: Date;
  usage: number;
}

interface TemplateManagerProps {
  className?: string;
}

export function TemplateManager({ className = "" }: TemplateManagerProps) {
  const [templates] = useState<Template[]>([
    {
      id: '1',
      name: 'Python ML Starter',
      description: 'Basic machine learning project template with PyTorch',
      category: 'Machine Learning',
      language: 'Python',
      difficulty: 'beginner',
      tags: ['pytorch', 'ml', 'starter'],
      content: '# ML Project Template\n\nimport torch\nimport numpy as np\n\n# Your code here',
      created: new Date(),
      usage: 142
    },
    {
      id: '2',
      name: 'API Documentation',
      description: 'REST API documentation template with examples',
      category: 'Documentation',
      language: 'Markdown',
      difficulty: 'beginner',
      tags: ['api', 'docs', 'rest'],
      content: '# API Documentation\n\n## Endpoints\n\n### GET /api/data\n\nReturns data',
      created: new Date(),
      usage: 87
    },
    {
      id: '3',
      name: 'Data Analysis Notebook',
      description: 'Jupyter notebook template for data analysis workflows',
      category: 'Data Science',
      language: 'Python',
      difficulty: 'intermediate',
      tags: ['jupyter', 'pandas', 'analysis'],
      content: '# Data Analysis\n\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n# Load data',
      created: new Date(),
      usage: 203
    },
    {
      id: '4',
      name: 'Research Paper',
      description: 'Academic research paper template with LaTeX formatting',
      category: 'Academic',
      language: 'LaTeX',
      difficulty: 'advanced',
      tags: ['research', 'academic', 'latex'],
      content: '\\documentclass{article}\n\\title{Research Title}\n\\author{Author Name}',
      created: new Date(),
      usage: 45
    }
  ]);

  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);

  const categories = ['all', ...Array.from(new Set(templates.map(t => t.category)))];

  const filteredTemplates = templates.filter(template => {
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    const matchesSearch = template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    return matchesCategory && matchesSearch;
  });

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-500';
      case 'intermediate': return 'text-yellow-500';
      case 'advanced': return 'text-red-500';
      default: return 'text-muted-foreground';
    }
  };

  const useTemplate = (template: Template) => {
    // In a real implementation, this would create a new file/project
    console.log('Using template:', template.name);
    // Copy content to clipboard or create new document
    navigator.clipboard?.writeText(template.content);
  };

  return (
    <div className={`space-y-6 font-mono ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary/10 rounded flex items-center justify-center">
            <span className="font-mono text-xs text-primary">tp</span>
          </div>
          <div>
            <h3 className="font-medium">Template Manager</h3>
            <p className="text-sm text-muted-foreground">Quick start templates and scaffolds</p>
          </div>
        </div>
        
        <Button className="font-mono text-xs warp-section-highlight">
          <span className="mr-1">nw</span>
          New Template
        </Button>
      </div>

      {/* Search and Filters */}
      <Card className="border-border/50">
        <CardContent className="p-4">
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <Input
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="font-mono text-xs"
              />
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="font-mono text-xs text-muted-foreground">Category:</span>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="font-mono text-xs bg-background border border-border rounded px-2 py-1"
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category === 'all' ? 'All' : category}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Templates Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredTemplates.map(template => (
          <Card
            key={template.id}
            className="cursor-pointer hover:bg-accent/50 warp-section-highlight border-border/50"
            onClick={() => setSelectedTemplate(template)}
          >
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <CardTitle className="font-mono text-sm animate-underline">
                    {template.name}
                  </CardTitle>
                  <p className="text-xs text-muted-foreground mt-1">
                    {template.description}
                  </p>
                </div>
                <div className={`font-mono text-xs ${getDifficultyColor(template.difficulty)}`}>
                  {template.difficulty.slice(0, 3)}
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono text-muted-foreground">{template.language}</span>
                <span className="font-mono text-muted-foreground">{template.usage} uses</span>
              </div>
              
              <div className="flex flex-wrap gap-1">
                {template.tags.map(tag => (
                  <Badge key={tag} variant="outline" className="text-xs font-mono">
                    {tag}
                  </Badge>
                ))}
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-muted-foreground">
                  {template.category}
                </span>
                <Button
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    useTemplate(template);
                  }}
                  className="font-mono text-xs warp-section-highlight"
                >
                  <span className="mr-1">us</span>
                  Use
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Template Preview Modal */}
      {selectedTemplate && (
        <Card className="border-border/50">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="font-mono text-sm flex items-center space-x-2">
                <span className="text-primary">pv</span>
                <span>Template Preview</span>
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedTemplate(null)}
                className="font-mono text-xs"
              >
                ×
              </Button>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <h4 className="font-mono font-medium">{selectedTemplate.name}</h4>
              <p className="text-sm text-muted-foreground">{selectedTemplate.description}</p>
              
              <div className="flex items-center space-x-4 text-xs">
                <span className="font-mono">Language: {selectedTemplate.language}</span>
                <span className="font-mono">Category: {selectedTemplate.category}</span>
                <span className={`font-mono ${getDifficultyColor(selectedTemplate.difficulty)}`}>
                  {selectedTemplate.difficulty}
                </span>
              </div>
            </div>
            
            <Separator />
            
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm">Template Content</span>
                <Button
                  size="sm"
                  onClick={() => useTemplate(selectedTemplate)}
                  className="font-mono text-xs warp-section-highlight"
                >
                  <span className="mr-1">cp</span>
                  Copy & Use
                </Button>
              </div>
              
              <pre className="bg-muted p-3 rounded text-xs font-mono overflow-x-auto max-h-64 overflow-y-auto subtle-scrollbar">
                {selectedTemplate.content}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="font-mono text-sm flex items-center space-x-2">
            <span className="text-orange-500">qa</span>
            <span>Quick Actions</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <Button variant="outline" size="sm" className="font-mono text-xs warp-section-highlight">
              <span className="mr-1">py</span>
              Python
            </Button>
            <Button variant="outline" size="sm" className="font-mono text-xs warp-section-highlight">
              <span className="mr-1">js</span>
              JavaScript
            </Button>
            <Button variant="outline" size="sm" className="font-mono text-xs warp-section-highlight">
              <span className="mr-1">md</span>
              Markdown
            </Button>
            <Button variant="outline" size="sm" className="font-mono text-xs warp-section-highlight">
              <span className="mr-1">nb</span>
              Notebook
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}