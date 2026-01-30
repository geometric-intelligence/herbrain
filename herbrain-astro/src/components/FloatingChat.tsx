import { useState, useRef, useEffect } from 'react';
import { getApiKey, setApiKey, clearApiKey } from '../lib/storage';
import OpenAI from 'openai';
import ReactMarkdown from 'react-markdown';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Floating AI Neurobot chat component - Amazon Rufus style
 * Displays a subtle floating button in the bottom-right corner
 * that expands into a chat panel when clicked.
 */
export default function FloatingChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKeyState] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [keyInput, setKeyInput] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const key = getApiKey();
    setApiKeyState(key);
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const saveApiKey = () => {
    if (keyInput.trim()) {
      setApiKey(keyInput.trim());
      setApiKeyState(keyInput.trim());
      setKeyInput('');
      setShowSettings(false);
    }
  };

  const removeApiKey = () => {
    clearApiKey();
    setApiKeyState(null);
    setShowSettings(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !apiKey || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const client = new OpenAI({
        apiKey,
        dangerouslyAllowBrowser: true,
      });

      const systemMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
        {
          role: 'system',
          content:
            'You are a neuroscientist specializing in the pregnancy and postpartum brain. You answer questions using short, precise sentences. Only respond to questions related to neuroscience of pregnancy, hormones, and motherhood, and women\'s brains. If a question is outside this scope, politely decline to answer.',
        },
        {
          role: 'system',
          content:
            'You are a helpful assistant explaining brain changes during pregnancy. Focus on the relationship between hormones and brain structure.',
        },
        {
          role: 'system',
          content:
            'The 3D mesh visualization shows subcortical structures including the accumbens nucleus, Amygdala, Caudate nucleus, Hippocampus, Globus pallidus (Pallidum), Putamen, and Thalamus. Red indicates areas that are growing as a result of pregnancy, and blue shows areas that are shrinking. Beige areas have not changed from the pre-pregnancy state.',
        },
      ];

      const apiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
        ...systemMessages,
        ...messages.map((m) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
        { role: 'user', content: userMessage },
      ];

      const response = await client.chat.completions.create({
        model: 'gpt-4o',
        messages: apiMessages,
        max_tokens: 500,
      });

      const assistantMessage = response.choices[0]?.message?.content || 'No response';
      setMessages((prev) => [...prev, { role: 'assistant', content: assistantMessage }]);
    } catch (err: any) {
      console.error('OpenAI API error:', err);
      const errorMessage =
        err?.message || 'Failed to get response from AI. Please check your API key.';
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${errorMessage}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating Trigger Button - Amazon Rufus Style */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 
                     bg-herbrain-dark text-white px-4 py-2.5 rounded-full 
                     shadow-lg hover:shadow-xl hover:scale-105
                     transition-all duration-200 group"
          aria-label="Open AI Neurobot chat"
        >
          {/* Colorful AI Sparkle Icon */}
          <div className="relative w-5 h-5 flex items-center justify-center">
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none">
              {/* Orange sparkle */}
              <path 
                d="M12 2L13.09 8.26L18 6L14.74 10.91L21 12L14.74 13.09L18 18L13.09 15.74L12 22L10.91 15.74L6 18L9.26 13.09L3 12L9.26 10.91L6 6L10.91 8.26L12 2Z" 
                fill="url(#sparkleGradient)"
              />
              <defs>
                <linearGradient id="sparkleGradient" x1="3" y1="2" x2="21" y2="22">
                  <stop offset="0%" stopColor="#F97316" />
                  <stop offset="50%" stopColor="#3B82F6" />
                  <stop offset="100%" stopColor="#22C55E" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <span className="font-medium text-sm">Neurobot</span>
          {apiKey && (
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 ml-0.5"></span>
          )}
        </button>
      )}

      {/* Chat Panel */}
      {isOpen && (
        <div 
          className="fixed bottom-6 right-6 z-50 w-[360px] sm:w-[400px] 
                     bg-white rounded-2xl shadow-2xl border border-herbrain-border/60
                     flex flex-col overflow-hidden
                     animate-in slide-in-from-bottom-4 duration-300"
          style={{ maxHeight: 'calc(100vh - 100px)', height: '520px' }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-herbrain-border/40 bg-herbrain-surface/30">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-herbrain-dark flex items-center justify-center">
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <path 
                    d="M12 2L13.09 8.26L18 6L14.74 10.91L21 12L14.74 13.09L18 18L13.09 15.74L12 22L10.91 15.74L6 18L9.26 13.09L3 12L9.26 10.91L6 6L10.91 8.26L12 2Z" 
                    fill="url(#sparkleGradientHeader)"
                  />
                  <defs>
                    <linearGradient id="sparkleGradientHeader" x1="3" y1="2" x2="21" y2="22">
                      <stop offset="0%" stopColor="#F97316" />
                      <stop offset="50%" stopColor="#3B82F6" />
                      <stop offset="100%" stopColor="#22C55E" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-herbrain-dark">Neurobot</h3>
                <p className="text-xs text-herbrain-muted">AI assistant for brain science</p>
              </div>
            </div>
            
            <div className="flex items-center gap-1">
              {/* Settings Button */}
              <button
                onClick={() => setShowSettings(!showSettings)}
                className={`p-2 rounded-lg transition-colors ${
                  showSettings ? 'bg-herbrain-surface' : 'hover:bg-herbrain-surface'
                }`}
                title="Settings"
              >
                <svg className="w-4 h-4 text-herbrain-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </button>
              
              {/* Close Button */}
              <button
                onClick={() => setIsOpen(false)}
                className="p-2 rounded-lg hover:bg-herbrain-surface transition-colors"
                title="Close"
              >
                <svg className="w-4 h-4 text-herbrain-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="px-4 py-3 bg-herbrain-surface/50 border-b border-herbrain-border/30">
              <p className="text-xs text-herbrain-muted mb-2">
                Enter your OpenAI API key. Stored locally.
              </p>
              {apiKey ? (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-500"></span>
                    <span className="text-xs text-herbrain-dark">API key configured</span>
                  </div>
                  <button
                    onClick={removeApiKey}
                    className="text-xs text-red-500 hover:text-red-600 font-medium"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <div className="flex gap-2">
                  <input
                    type="password"
                    value={keyInput}
                    onChange={(e) => setKeyInput(e.target.value)}
                    placeholder="sk-..."
                    className="premium-input flex-1 text-sm py-1.5"
                  />
                  <button onClick={saveApiKey} className="premium-btn text-xs py-1.5 px-3">
                    Save
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Chat Messages */}
          <div
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto p-4 custom-scrollbar"
          >
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center px-4">
                <div className="w-12 h-12 rounded-full bg-herbrain-surface flex items-center justify-center mb-3">
                  <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none">
                    <path 
                      d="M12 2L13.09 8.26L18 6L14.74 10.91L21 12L14.74 13.09L18 18L13.09 15.74L12 22L10.91 15.74L6 18L9.26 13.09L3 12L9.26 10.91L6 6L10.91 8.26L12 2Z" 
                      fill="url(#sparkleGradientEmpty)"
                    />
                    <defs>
                      <linearGradient id="sparkleGradientEmpty" x1="3" y1="2" x2="21" y2="22">
                        <stop offset="0%" stopColor="#F97316" />
                        <stop offset="50%" stopColor="#3B82F6" />
                        <stop offset="100%" stopColor="#22C55E" />
                      </linearGradient>
                    </defs>
                  </svg>
                </div>
                <p className="text-sm text-herbrain-dark font-medium mb-1">
                  Hi! I'm Neurobot
                </p>
                <p className="text-xs text-herbrain-muted">
                  {apiKey
                    ? 'Ask me about brain changes during pregnancy'
                    : 'Configure your API key in settings to start'}
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={
                      msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'
                    }
                  >
                    {msg.role === 'user' ? (
                      msg.content
                    ) : (
                      <ReactMarkdown
                        components={{
                          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                          em: ({ children }) => <em className="italic">{children}</em>,
                          ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                          ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                          li: ({ children }) => <li>{children}</li>,
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    )}
                  </div>
                ))}
                {loading && (
                  <div className="chat-bubble-assistant">
                    <span className="animate-pulse">Thinking...</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Input Form */}
          <div className="p-3 border-t border-herbrain-border/30 bg-white">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={apiKey ? 'Ask about pregnancy brain...' : 'Configure API key first'}
                disabled={!apiKey || loading}
                className="premium-input flex-1 text-sm py-2"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />
              <button
                type="submit"
                disabled={!apiKey || loading || !input.trim()}
                className="premium-btn px-3 py-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </button>
            </form>
            <p className="text-[10px] text-herbrain-muted/60 mt-2 text-center">
              Educational only. Not medical advice.
            </p>
          </div>
        </div>
      )}
    </>
  );
}
