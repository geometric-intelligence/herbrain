import { useState, useRef, useEffect } from 'react';
import { getApiKey, setApiKey, clearApiKey } from '../lib/storage';
import OpenAI from 'openai';

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
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Load API key on mount
  useEffect(() => {
    const key = getApiKey();
    setApiKeyState(key);
  }, []);

  // Auto-scroll to bottom
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

      // Build context
      const context = `Current gestational week: ${week}

Please analyze the brain visualization and provide insights about brain changes during pregnancy.`;

      // System messages for the neurobot
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
          content: context,
        },
        {
          role: 'system',
          content:
            'The 3D mesh visualization shows subcortical structures including the accumbens nucleus, Amygdala, Caudate nucleus, Hippocampus, Globus pallidus (Pallidum), Putamen, and Thalamus. Red indicates areas that are growing as a result of pregnancy, and blue shows areas that are shrinking. Beige areas have not changed from the pre-pregnancy state.',
        },
      ];

      // Build messages array
      const apiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
        ...systemMessages,
        ...messages.map((m) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
      ];

      // Try to get mesh screenshot if available
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
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-base font-semibold text-herbrain-dark">
            Ask the AI Neurobot
          </h3>
          <p className="text-xs text-herbrain-muted">
            Questions about brain changes during pregnancy
          </p>
        </div>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-1.5 text-herbrain-muted hover:text-herbrain-dark transition-colors"
          title="API Key Settings"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="mb-3 p-3 bg-gray-50 rounded-lg">
          <p className="text-xs text-herbrain-muted mb-2">
            Enter your OpenAI API key. Stored locally in your browser.
          </p>
          {apiKey ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-green-600">API key configured</span>
              <button
                onClick={removeApiKey}
                className="text-xs text-red-500 hover:text-red-700"
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
                className="flex-1 px-2 py-1.5 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-herbrain-green"
              />
              <button
                onClick={saveApiKey}
                className="px-3 py-1.5 text-xs bg-herbrain-green text-white rounded-lg hover:bg-herbrain-green/90"
              >
                Save
              </button>
            </div>
          )}
        </div>
      )}

      {/* Chat Messages */}
      <div
        ref={chatContainerRef}
        className="h-32 overflow-y-auto border border-gray-200 rounded-lg p-2.5 mb-3 bg-gray-50"
      >
        {messages.length === 0 ? (
          <p className="text-xs text-herbrain-muted text-center py-4">
            {apiKey
              ? 'Ask a question about the brain visualization...'
              : 'Configure your API key to start chatting'}
          </p>
        ) : (
          <div className="space-y-3">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={
                  msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'
                }
              >
                {msg.content}
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
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={apiKey ? 'Type your question...' : 'Configure API key first'}
          disabled={!apiKey || loading}
          className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-herbrain-green disabled:bg-gray-100"
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
          className="px-3 py-2 text-sm bg-herbrain-green text-white rounded-lg hover:bg-herbrain-green/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Ask
        </button>
      </form>

      {/* Disclaimer */}
      <p className="text-[10px] text-herbrain-muted mt-2">
        Educational tool only. May generate inaccurate responses. Not for medical use.
      </p>
    </div>
  );
}
