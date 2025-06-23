import React, { useState } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { 
  Search, 
  BookOpen, 
  Globe, 
  Archive, 
  FileText, 
  Brain, 
  Key, 
  ShieldAlert, 
  Clock, 
  Image, 
  FileDigit,
  Loader2,
  ExternalLink,
  ChevronRight,
  ChevronDown,
  Database,
  Newspaper,
  GraduationCap,
  Library,
  History,
  Languages,
  BookMarked,
  ScrollText,
  MapPin,
  Calendar,
  User,
  Tag,
  Filter,
  Download,
  Copy,
  Share2,
  MoreHorizontal
} from "lucide-react";

interface SearchResult {
  id: string;
  title: string;
  description: string;
  source: string;
  url?: string;
  displayUrl?: string;
  type: string;
  relevance: number;
  metadata?: {
    author?: string;
    date?: string;
    location?: string;
    language?: string;
    tags?: string[];
    category?: string;
    fileType?: string;
    pages?: number;
  };
}

interface DeepSearchProps {
  activeTab: string;
}

export function DeepSearch({ activeTab }: DeepSearchProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [expandedResult, setExpandedResult] = useState<string | null>(null);

  // Helper function to get tab icon
  const getTabIcon = (tab: string) => {
    const icons = {
      'dictionaries': <BookOpen className="size-6 text-primary" />,
      'thesaurus': <FileText className="size-6 text-primary" />,
      'encyclopedia': <Brain className="size-6 text-primary" />,
      'wikipedia': <Globe className="size-6 text-primary" />,
      'library-congress': <Archive className="size-6 text-primary" />,
      'internet-archive': <Clock className="size-6 text-primary" />,
      'national-archives': <FileDigit className="size-6 text-primary" />,
      'academic': <GraduationCap className="size-6 text-primary" />,
      'patents': <Key className="size-6 text-primary" />,
      'legal': <ShieldAlert className="size-6 text-primary" />,
      'news': <FileText className="size-6 text-primary" />,
      'newspapers': <Newspaper className="size-6 text-primary" />,
      'media': <Image className="size-6 text-primary" />
    };
    
    return icons[tab as keyof typeof icons] || <Search className="size-6 text-primary" />;
  };

  // Helper function to get tab title
  const getTabTitle = (tab: string): string => {
    const titles = {
      'dictionaries': 'Dictionary Search',
      'thesaurus': 'Thesaurus',
      'encyclopedia': 'Encyclopedia',
      'wikipedia': 'Wikipedia',
      'library-congress': 'Library of Congress',
      'internet-archive': 'Internet Archive',
      'national-archives': 'National Archives',
      'academic': 'Academic Papers',
      'patents': 'Patent Database',
      'legal': 'Legal Database',
      'news': 'News Archives',
      'newspapers': 'Historical Newspapers',
      'media': 'Media Collections'
    };
    
    return titles[tab as keyof typeof titles] || 'Deep Search';
  };

  // Helper function to get tab description
  const getTabDescription = (tab: string): string => {
    const descriptions = {
      'dictionaries': 'Comprehensive definitions, etymology, and usage examples',
      'thesaurus': 'Synonyms, antonyms, and related terms',
      'encyclopedia': 'In-depth articles and reference materials',
      'wikipedia': 'Collaborative encyclopedia with global coverage',
      'library-congress': 'Historical documents and government archives',
      'internet-archive': 'Digital preservation of web pages and media',
      'national-archives': 'Official government records and documents',
      'academic': 'Peer-reviewed research papers and scholarly articles',
      'patents': 'Patent filings and intellectual property records',
      'legal': 'Legal cases, statutes, and regulatory documents',
      'news': 'Contemporary and historical news coverage',
      'newspapers': 'Historical newspaper archives and collections',
      'media': 'Digital media collections and multimedia archives'
    };
    
    return descriptions[tab as keyof typeof descriptions] || 'Search across comprehensive archives and databases';
  };

  const generateMockResults = (query: string, source: string): SearchResult[] => {
    const baseResults = {
      dictionaries: [
        {
          id: '1',
          title: `${query} | Definition, Etymology & Usage - Oxford English Dictionary`,
          description: `Comprehensive definition of "${query}" with detailed etymology, pronunciation guide, and usage examples from historical and contemporary sources. Includes related terms and word origins.`,
          source: 'Oxford English Dictionary',
          url: 'https://oed.com',
          displayUrl: 'oed.com › definition › ' + query.toLowerCase(),
          type: 'definition',
          relevance: 95,
          metadata: {
            author: 'Oxford University Press',
            date: '2024',
            language: 'English',
            tags: ['definition', 'etymology', 'pronunciation'],
            category: 'Dictionary',
            fileType: 'HTML'
          }
        },
        {
          id: '2',
          title: `${query} Synonyms, Antonyms & Related Words - Merriam-Webster`,
          description: `Find synonyms, antonyms, and related terms for "${query}". Includes contextual usage examples, word relationships, and comprehensive thesaurus entries with pronunciation guides.`,
          source: 'Merriam-Webster',
          url: 'https://merriam-webster.com',
          displayUrl: 'merriam-webster.com › thesaurus › ' + query.toLowerCase(),
          type: 'thesaurus',
          relevance: 90,
          metadata: {
            author: 'Merriam-Webster',
            date: '2024',
            language: 'English',
            tags: ['synonyms', 'antonyms', 'thesaurus'],
            category: 'Reference',
            fileType: 'HTML'
          }
        },
        {
          id: '3',
          title: `Cambridge Dictionary: ${query} meaning & definition`,
          description: `Clear definition of "${query}" with examples from written and spoken English. British and American pronunciations with audio, plus grammar information and usage notes.`,
          source: 'Cambridge Dictionary',
          url: 'https://dictionary.cambridge.org',
          displayUrl: 'dictionary.cambridge.org › dictionary › english › ' + query.toLowerCase(),
          type: 'definition',
          relevance: 88,
          metadata: {
            author: 'Cambridge University Press',
            date: '2024',
            language: 'English',
            tags: ['definition', 'pronunciation', 'examples'],
            category: 'Dictionary',
            fileType: 'HTML'
          }
        }
      ],
      wikipedia: [
        {
          id: '3',
          title: `${query} - Wikipedia`,
          description: `Comprehensive encyclopedia article about "${query}" with references, citations, and multilingual versions. Collaborative content from thousands of contributors worldwide.`,
          source: 'Wikipedia',
          url: 'https://wikipedia.org',
          displayUrl: 'en.wikipedia.org › wiki › ' + query.replace(' ', '_'),
          type: 'encyclopedia',
          relevance: 88,
          metadata: {
            author: 'Wikipedia Contributors',
            date: '2024-01-15',
            language: 'Multiple',
            tags: ['encyclopedia', 'open-source', 'collaborative'],
            category: 'Encyclopedia',
            fileType: 'HTML'
          }
        },
        {
          id: '4',
          title: `${query} (disambiguation) - Wikipedia`,
          description: `Disambiguation page for "${query}" listing multiple meanings, uses, and related topics. Links to specific articles for each usage of the term.`,
          source: 'Wikipedia',
          url: 'https://wikipedia.org',
          displayUrl: 'en.wikipedia.org › wiki › ' + query.replace(' ', '_') + '_(disambiguation)',
          type: 'disambiguation',
          relevance: 75,
          metadata: {
            author: 'Wikipedia Contributors',
            date: '2024-01-10',
            language: 'English',
            tags: ['disambiguation', 'multiple-meanings'],
            category: 'Encyclopedia',
            fileType: 'HTML'
          }
        }
      ],
      'library-congress': [
        {
          id: '4',
          title: `Historical Records and Documents: ${query} - Library of Congress`,
          description: `Historical documents, manuscripts, and archival materials related to "${query}" from the Library of Congress collection. Digitized primary sources and government records.`,
          source: 'Library of Congress',
          url: 'https://loc.gov',
          displayUrl: 'loc.gov › collections › search › ' + query.toLowerCase().replace(' ', '-'),
          type: 'historical-document',
          relevance: 82,
          metadata: {
            author: 'Library of Congress',
            date: '1950-2020',
            location: 'Washington, D.C.',
            tags: ['historical', 'archival', 'manuscripts'],
            category: 'Government Archive',
            fileType: 'PDF, TIFF'
          }
        },
        {
          id: '5',
          title: `${query} Research Guide - Library of Congress Research Guides`,
          description: `Comprehensive research guide for "${query}" including recommended databases, key resources, and research strategies. Created by Library of Congress librarians.`,
          source: 'Library of Congress',
          url: 'https://guides.loc.gov',
          displayUrl: 'guides.loc.gov › ' + query.toLowerCase().replace(' ', '-') + '-research-guide',
          type: 'research-guide',
          relevance: 78,
          metadata: {
            author: 'LC Reference Librarians',
            date: '2023',
            tags: ['research-guide', 'resources', 'bibliography'],
            category: 'Research Guide',
            fileType: 'HTML'
          }
        }
      ],
      'internet-archive': [
        {
          id: '5',
          title: `Archived Web Pages: ${query} - Internet Archive Wayback Machine`,
          description: `Historical web pages, digital books, software, and multimedia content related to "${query}" from the Wayback Machine. Digital preservation spanning decades.`,
          source: 'Internet Archive',
          url: 'https://archive.org',
          displayUrl: 'web.archive.org › search › ' + query.toLowerCase().replace(' ', '+'),
          type: 'web-archive',
          relevance: 85,
          metadata: {
            author: 'Internet Archive',
            date: '1996-2024',
            tags: ['wayback-machine', 'digital-preservation', 'web-history'],
            category: 'Digital Archive',
            fileType: 'HTML, PDF, various'
          }
        }
      ],
      academic: [
        {
          id: '6',
          title: `Academic Research Papers on ${query} - Google Scholar`,
          description: `Peer-reviewed research papers, academic journals, and scholarly articles about "${query}" from multiple databases. Citations, abstracts, and full-text access where available.`,
          source: 'Google Scholar',
          url: 'https://scholar.google.com',
          displayUrl: 'scholar.google.com › scholar › search › ' + query.toLowerCase().replace(' ', '+'),
          type: 'academic-paper',
          relevance: 91,
          metadata: {
            author: 'Various Researchers',
            date: '2020-2024',
            tags: ['peer-reviewed', 'research', 'scholarly'],
            category: 'Academic Research',
            fileType: 'PDF'
          }
        },
        {
          id: '7',
          title: `${query}: Recent Publications - PubMed`,
          description: `Recent scientific publications and medical literature on "${query}". Abstracts from biomedical journals, clinical studies, and research findings.`,
          source: 'PubMed',
          url: 'https://pubmed.ncbi.nlm.nih.gov',
          displayUrl: 'pubmed.ncbi.nlm.nih.gov › search › ' + query.toLowerCase().replace(' ', '+'),
          type: 'medical-research',
          relevance: 89,
          metadata: {
            author: 'Medical Researchers',
            date: '2020-2024',
            tags: ['medical', 'scientific', 'peer-reviewed'],
            category: 'Medical Literature',
            fileType: 'Abstract, PDF'
          }
        }
      ],
      news: [
        {
          id: '7',
          title: `${query} News Coverage - Historical and Current Articles`,
          description: `Historical and contemporary news articles about "${query}" from major publications and news archives. Coverage from leading newspapers and media outlets.`,
          source: 'News Archives',
          url: 'https://newspapers.com',
          displayUrl: 'newspapers.com › search › ' + query.toLowerCase().replace(' ', '+'),
          type: 'news-article',
          relevance: 78,
          metadata: {
            author: 'Various News Outlets',
            date: '1990-2024',
            tags: ['journalism', 'current-events', 'historical-news'],
            category: 'News Media',
            fileType: 'HTML, PDF'
          }
        }
      ]
    };

    return baseResults[source as keyof typeof baseResults] || [];
  };

  // Mock search function
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Generate mock results based on active tab and search query
    const mockResults = generateMockResults(searchQuery, activeTab);
    setSearchResults(mockResults);
    setIsSearching(false);
  };

  const handleLinkClick = (result: SearchResult) => {
    // In a real implementation, this would open the actual link
    console.log('Opening:', result.url);
    window.open(result.url, '_blank');
  };

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
            {getTabIcon(activeTab)}
          </div>
          <div>
            <h3 className="text-lg font-semibold">{getTabTitle(activeTab)}</h3>
            <p className="text-sm text-muted-foreground">{getTabDescription(activeTab)}</p>
          </div>
        </div>
        
        {/* Search Input */}
        <div className="flex gap-2 max-w-2xl mx-auto">
          <div className="relative flex-grow">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground size-4" />
            <Input
              placeholder={`Search ${getTabTitle(activeTab).toLowerCase()}...`}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
          </div>
          <Button onClick={handleSearch} disabled={isSearching || !searchQuery.trim()}>
            {isSearching ? (
              <>
                <Loader2 className="size-4 mr-2 animate-spin" />
                Searching...
              </>
            ) : (
              <>
                <Search className="size-4 mr-2" />
                Search
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Search Results - Google Style */}
      {searchResults.length > 0 && (
        <div className="max-w-4xl mx-auto">
          {/* Results Info */}
          <div className="mb-4 text-sm text-muted-foreground">
            About {searchResults.length} results (0.{Math.floor(Math.random() * 9) + 1}{Math.floor(Math.random() * 9) + 1} seconds)
          </div>

          <div className="space-y-6">
            {searchResults.map((result) => (
              <div key={result.id} className="group">
                {/* URL Breadcrumb */}
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-4 h-4 rounded-full bg-muted flex items-center justify-center">
                    <Globe className="size-3 text-muted-foreground" />
                  </div>
                  <span className="text-sm text-green-700 dark:text-green-400">
                    {result.displayUrl || result.url}
                  </span>
                  <MoreHorizontal className="size-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>

                {/* Title Link */}
                <h3 className="mb-1">
                  <button
                    onClick={() => handleLinkClick(result)}
                    className="text-blue-700 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 text-lg font-normal hover:underline text-left"
                  >
                    {result.title}
                  </button>
                </h3>

                {/* Description */}
                <p className="text-sm text-muted-foreground leading-5 mb-2 max-w-3xl">
                  {result.description}
                </p>

                {/* Metadata and Additional Links */}
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  {result.metadata?.date && (
                    <span>{result.metadata.date}</span>
                  )}
                  {result.metadata?.fileType && (
                    <span className="flex items-center gap-1">
                      <FileDigit className="size-3" />
                      {result.metadata.fileType}
                    </span>
                  )}
                  {result.metadata?.pages && (
                    <span>{result.metadata.pages} pages</span>
                  )}
                  
                  {/* Additional Action Links */}
                  <div className="flex items-center gap-3 ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="text-blue-600 dark:text-blue-400 hover:underline">
                      Cached
                    </button>
                    <button className="text-blue-600 dark:text-blue-400 hover:underline">
                      Similar
                    </button>
                    <button className="text-blue-600 dark:text-blue-400 hover:underline">
                      Cite
                    </button>
                  </div>
                </div>

                {/* Related/Similar Results (occasionally) */}
                {result.id === '1' && searchResults.length > 1 && (
                  <div className="mt-3 ml-6 border-l-2 border-muted pl-4">
                    <div className="text-sm">
                      <span className="text-muted-foreground">More results from </span>
                      <span className="text-blue-600 dark:text-blue-400">{result.source.toLowerCase().replace(' ', '')}.com</span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Related Searches */}
          <div className="mt-8 pt-6 border-t">
            <h4 className="text-sm font-medium text-muted-foreground mb-3">Related searches</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {[
                `${searchQuery} definition`,
                `${searchQuery} examples`,
                `${searchQuery} etymology`,
                `${searchQuery} pronunciation`,
                `history of ${searchQuery}`,
                `${searchQuery} research`
              ].map((query, index) => (
                <button
                  key={index}
                  className="text-left p-2 rounded hover:bg-muted/50 text-blue-600 dark:text-blue-400 text-sm"
                  onClick={() => {
                    setSearchQuery(query);
                    handleSearch();
                  }}
                >
                  <Search className="size-3 mr-2 inline" />
                  {query}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search Tips */}
      {searchResults.length === 0 && !isSearching && (
        <Card className="p-6">
          <h4 className="font-medium mb-3">Search Tips</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-muted-foreground">
            <div>
              <h5 className="font-medium text-foreground mb-2">Query Techniques:</h5>
              <ul className="space-y-1">
                <li>• Use quotes for exact phrases</li>
                <li>• Add + to require terms</li>
                <li>• Use - to exclude terms</li>
                <li>• Try wildcards with *</li>
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-foreground mb-2">Available Sources:</h5>
              <ul className="space-y-1">
                <li>• {getTabTitle(activeTab)} database</li>
                <li>• Cross-referenced materials</li>
                <li>• Historical archives</li>
                <li>• Multi-language support</li>
              </ul>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}