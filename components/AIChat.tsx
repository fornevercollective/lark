import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { Textarea } from './ui/textarea';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  tokens?: number;
}

interface AIModel {
  id: string;
  name: string;
  description: string;
  maxTokens: number;
  costPer1k: number;
}

interface AIChatProps {
  className?: string;
}

export function AIChat({ className = "" }: AIChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      role: 'system',
      content: 'AI assistant ready. Type your message below.',
      timestamp: new Date(),
      tokens: 12
    }
  ]);

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const models: AIModel[] = [
    {
      id: 'gpt-4',
      name: 'GPT-4',
      description: 'Most capable model, best for complex tasks',
      maxTokens: 8192,
      costPer1k: 0.03
    },
    {
      id: 'gpt-3.5-turbo',
      name: 'GPT-3.5 Turbo',
      description: 'Fast and efficient for most tasks',
      maxTokens: 4096,
      costPer1k: 0.002
    },
    {
      id: 'claude-3',
      name: 'Claude 3',
      description: 'Excellent for analysis and reasoning',
      maxTokens: 100000,
      costPer1k: 0.015
    }
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      tokens: Math.ceil(input.length / 4) // Rough token estimate
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: generateMockResponse(input),
        timestamp: new Date(),
        tokens: Math.ceil(Math.random() * 200 + 50)
      };

      setMessages(prev => [...prev, aiResponse]);
      setIsLoading(false);
    }, 1000 + Math.random() * 2000);
  };

  const generateMockResponse = (userInput: string): string => {
    const input = userInput.toLowerCase();
    
    if (input.includes('code') || input.includes('python') || input.includes('javascript')) {
      return `Here's a code example for your request:

\`\`\`python
def example_function():
    """
    This is a sample function based on your request.
    """
    result = "Hello, Lark!"
    return result

# Usage
output = example_function()
print(output)
\`\`\`

This code demonstrates the basic structure. Would you like me to explain any part of it or modify it for your specific use case?`;
    }
    
    if (input.includes('markdown') || input.includes('documentation')) {
      return `# Documentation Template

Here's a markdown template for your documentation:

## Overview
Brief description of your project or feature.

## Getting Started
1. First step
2. Second step
3. Third step

## API Reference
- \`function_name()\` - Description
- \`another_function()\` - Description

## Examples
\`\`\`bash
# Command example
npm install package-name
\`\`\`

Would you like me to customize this template for your specific needs?`;
    }
    
    if (input.includes('help') || input.includes('how')) {
      return `I'm here to help! I can assist you with:

**Development Tasks:**
- Code review and optimization
- Bug fixing and debugging
- Architecture planning
- Documentation writing

**AI & ML Projects:**
- Model training strategies
- Data preprocessing
- Performance optimization
- Framework recommendations

**General Programming:**
- Language-specific questions
- Best practices
- Tool recommendations
- Project planning

What specific area would you like to explore further?`;
    }
    
    return `I understand you're asking about "${userInput}". 

This is a simulated AI response for demonstration purposes. In a production environment, this would connect to a real language model API.

Key points to consider:
- Your request has been processed
- I can help with various development tasks
- Feel free to ask follow-up questions
- I can provide code examples, explanations, and guidance

Is there a specific aspect you'd like me to elaborate on?`;
  };

  const clearChat = () => {
    setMessages([
      {
        id: '1',
        role: 'system',
        content: 'Chat cleared. AI assistant ready.',
        timestamp: new Date(),
        tokens: 8
      }
    ]);
  };

  const totalTokens = messages.reduce((sum, msg) => sum + (msg.tokens || 0), 0);
  const currentModel = models.find(m => m.id === selectedModel) || models[0];

  return (
    <div className={`space-y-6 font-mono ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary/10 rounded flex items-center justify-center">
            <span className="font-mono text-xs text-primary">ai</span>
          </div>
          <div>
            <h3 className="font-medium">AI Assistant</h3>
            <p className="text-sm text-muted-foreground">Chat with AI for development help</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="font-mono text-xs">
            {totalTokens} tokens
          </Badge>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearChat}
            className="font-mono text-xs animate-accent-line"
          >
            <span className="mr-1">cl</span>
            Clear
          </Button>
        </div>
      </div>

      {/* Model Selection */}
      <Card className="border-border/50">
        <CardHeader className="pb-3">
          <CardTitle className="font-mono text-sm flex items-center space-x-2">
            <span className="text-blue-500">md</span>
            <span>AI Model</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {models.map(model => (
              <div
                key={model.id}
                className={`p-3 border rounded cursor-pointer warp-section-highlight ${
                  selectedModel === model.id
                    ? 'border-primary bg-accent/50'
                    : 'border-border hover:bg-accent/20'
                }`}
                onClick={() => setSelectedModel(model.id)}
              >
                <div className="font-mono text-sm font-medium">{model.name}</div>
                <div className="text-xs text-muted-foreground mb-2">{model.description}</div>
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono">{model.maxTokens.toLocaleString()} tokens</span>
                  <span className="font-mono">${model.costPer1k}/1k</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Chat Messages */}
      <Card className="border-border/50">
        <CardHeader className="pb-3">
          <CardTitle className="font-mono text-sm flex items-center space-x-2">
            <span className="text-green-500">ch</span>
            <span>Conversation</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 overflow-y-auto subtle-scrollbar space-y-4 mb-4">
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[80%] p-3 rounded animate-terminal-fade ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : message.role === 'system'
                      ? 'bg-muted'
                      : 'bg-accent'
                  }`}
                >
                  <div className="whitespace-pre-wrap text-sm font-mono">
                    {message.content}
                  </div>
                  <div className="flex items-center justify-between mt-2 text-xs opacity-70">
                    <span className="font-mono">
                      {message.role === 'user' ? 'you' : message.role}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className="font-mono">{message.tokens} tokens</span>
                      <span className="font-mono">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-accent p-3 rounded animate-terminal-fade">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-current rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-current rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-current rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                    <span className="font-mono text-xs ml-2">AI thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <Separator className="my-4" />

          {/* Input */}
          <div className="space-y-3">
            <Textarea
              placeholder="Ask me anything about development, coding, or AI..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              className="font-mono text-sm resize-none"
              rows={3}
            />
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                <span className="font-mono">Model: {currentModel.name}</span>
                <span className="font-mono">•</span>
                <span className="font-mono">Enter to send, Shift+Enter for new line</span>
              </div>
              
              <Button
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                className="font-mono text-xs warp-section-highlight"
              >
                <span className="mr-1">→</span>
                Send
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Prompts */}
      <Card className="border-border/50">
        <CardHeader className="pb-3">
          <CardTitle className="font-mono text-sm flex items-center space-x-2">
            <span className="text-purple-500">qp</span>
            <span>Quick Prompts</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {[
              'Help me debug this Python code',
              'Explain this algorithm step by step',
              'Write documentation for my function',
              'Optimize this SQL query',
              'Create a unit test for this code',
              'Suggest best practices for this pattern'
            ].map((prompt, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => setInput(prompt)}
                className="font-mono text-xs text-left justify-start warp-section-highlight"
              >
                {prompt}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}