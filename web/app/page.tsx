"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";

export interface Source {
  shabad_id: number;
  ang: number;
  raag: string;
  writer: string;
  score: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  failedQuotes?: string[];
  loading?: boolean;
  questionType?: string;
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "/api";

const MAX_HISTORY_TURNS = 10;

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const buildHistory = useCallback(
    (msgs: Message[]) => {
      // Build rolling history from the last MAX_HISTORY_TURNS completed turns
      const completed = msgs.filter((m) => !m.loading);
      const recent = completed.slice(-MAX_HISTORY_TURNS * 2);
      return recent.map((m) => ({
        role: m.role,
        content: m.content,
      }));
    },
    []
  );

  const handleSubmit = useCallback(
    async (question: string) => {
      if (!question.trim() || isLoading) return;

      const userMsg: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: question.trim(),
      };

      const assistantPlaceholder: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: "",
        loading: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);
      setIsLoading(true);

      try {
        const history = buildHistory([...messages, userMsg]);

        const res = await fetch(`${API_BASE}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: question.trim(),
            history: history.slice(0, -1), // exclude the just-sent message
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(errData.detail ?? `HTTP ${res.status}`);
        }

        const data = await res.json();

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantPlaceholder.id
              ? {
                  ...m,
                  content: data.answer,
                  sources: data.sources ?? [],
                  failedQuotes: data.failed_quotes ?? [],
                  questionType: data.question_type ?? "conceptual",
                  loading: false,
                }
              : m
          )
        );
      } catch (err: unknown) {
        const message =
          err instanceof Error
            ? err.message
            : "An unexpected error occurred. Please try again.";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantPlaceholder.id
              ? {
                  ...m,
                  content: `Sorry, I encountered an error: ${message}`,
                  loading: false,
                }
              : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, messages, buildHistory]
  );

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto px-4">
      {/* Header */}
      <header className="flex-none py-4 border-b border-stone-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-amber-500 flex items-center justify-center text-white font-bold text-lg select-none">
            ੴ
          </div>
          <div>
            <h1 className="text-xl font-semibold text-stone-800">
              Gurbani Guidance
            </h1>
            <p className="text-xs text-stone-500">
              Questions answered from Sri Guru Granth Sahib Ji
            </p>
          </div>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center px-4 py-16 text-stone-400">
            <div className="gurmukhi text-5xl mb-4 text-amber-400">ੴ</div>
            <p className="text-lg font-medium text-stone-500 mb-2">
              Ask a question about Gurbani
            </p>
            <p className="text-sm max-w-md">
              All answers are grounded exclusively in Sri Guru Granth Sahib Ji.
              Quotes are verified against the corpus before being shown to you.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <footer className="flex-none pb-4 pt-2">
        <ChatInput onSubmit={handleSubmit} disabled={isLoading} />
        <p className="text-center text-xs text-stone-400 mt-2">
          Answers are grounded in SGGS · Not a substitute for a qualified Granthi
        </p>
      </footer>
    </div>
  );
}
