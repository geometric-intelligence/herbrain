import { useState, useRef, useEffect } from 'react';
import { getApiKey, setApiKey, clearApiKey } from '../lib/storage';
import OpenAI from 'openai';
import ReactMarkdown from 'react-markdown';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface GptChatProps {
  week: number;
  getMeshScreenshot?: () => Promise<string | null>;
}

export default function GptChat({ week, getMeshScreenshot }: GptChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKeyState] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [keyInput, setKeyInput] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
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

  // Auto-expand when there are messages
  useEffect(() => {
    if (messages.length > 0) {
      setIsExpanded(true);
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
    setIsExpanded(true);

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
        {
          role: 'system',
          content:
            `CURRENT APP STATE - You have access to the following information about the user's current view:
- Gestational Week Slider: Currently set to week ${week} (out of 40 weeks total)

When the user asks about the current slider value, gestational week, or what week they're viewing, tell them it is week ${week}. Use this context to provide relevant information about brain changes at this specific stage of pregnancy.`,
        },
      ];

      const apiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
        ...systemMessages,
        ...messages.map((m) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
      ];

      let imageContent: OpenAI.Chat.ChatCompletionContentPart[] = [
        { type: 'text', text: userMessage },
      ];

      if (getMeshScreenshot) {
        try {
          const screenshot = await getMeshScreenshot();
          if (screenshot) {
            imageContent.push({
              type: 'image_url',
              image_url: { url: screenshot },
            });
          }
        } catch (err) {
          console.warn('Could not capture mesh screenshot:', err);
        }
      }

      apiMessages.push({
        role: 'user',
        content: imageContent,
      });

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
    <div className="premium-card-static overflow-hidden">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-5 hover:bg-herbrain-surface/50 transition-colors"
      >
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-herbrain-green to-herbrain-green-dark flex items-center justify-center flex-shrink-0 shadow-sm">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
            </svg>
          </div>
          <div className="text-left">
            <h3 className="text-base font-semibold text-herbrain-dark">
              AI Neurobot
            </h3>
            <p className="text-sm text-herbrain-muted">
              Ask questions about brain changes during pregnancy
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {apiKey && (
            <span className="w-2 h-2 rounded-full bg-green-500"></span>
          )}
          <svg 
            className={`w-5 h-5 text-herbrain-muted transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expandable Content */}
      <div className={`transition-all duration-300 ease-in-out overflow-hidden ${isExpanded ? 'max-h-[600px] opacity-100' : 'max-h-0 opacity-0'}`}>
        <div className="px-5 pb-5 border-t border-herbrain-border/30">
          {/* Settings Toggle */}
          <div className="flex justify-end pt-3 pb-2">
            <button
              onClick={(e) => { e.stopPropagation(); setShowSettings(!showSettings); }}
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
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="mb-4 p-4 bg-herbrain-surface/70 rounded-xl">
              <p className="text-sm text-herbrain-muted mb-3">
                Enter your OpenAI API key. Stored locally in your browser.
              </p>
              {apiKey ? (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-500"></span>
                    <span className="text-sm text-herbrain-dark">API key configured</span>
                  </div>
                  <button
                    onClick={removeApiKey}
                    className="text-sm text-red-500 hover:text-red-600 font-medium"
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
                    className="premium-input flex-1"
                  />
                  <button onClick={saveApiKey} className="premium-btn">
                    Save
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Chat Messages */}
          <div
            ref={chatContainerRef}
            className="h-44 overflow-y-auto rounded-xl bg-herbrain-surface/40 p-4 mb-4 custom-scrollbar"
          >
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <p className="text-sm text-herbrain-muted/70 text-center">
                  {apiKey
                    ? 'Ask a question about the brain visualization...'
                    : 'Configure your API key to start chatting'}
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
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={apiKey ? 'Ask about pregnancy brain changes...' : 'Configure API key first'}
              disabled={!apiKey || loading}
              className="premium-input flex-1"
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
              className="premium-btn px-5"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </button>
          </form>

          {/* Disclaimer */}
          <p className="text-xs text-herbrain-muted/60 mt-4 text-center">
            Educational tool only. May generate inaccurate responses. Not for medical advice.
          </p>
        </div>
      </div>
    </div>
  );
}
