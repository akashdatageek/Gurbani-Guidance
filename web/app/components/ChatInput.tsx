"use client";

import { useState, useRef, useCallback, KeyboardEvent } from "react";

interface ChatInputProps {
  onSubmit: (question: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSubmit, disabled = false }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, disabled, onSubmit]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const handleInput = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, []);

  const isEmpty = value.trim().length === 0;

  return (
    <div className="flex items-end gap-2 rounded-2xl border border-stone-300 bg-white px-3 py-2 shadow-sm focus-within:border-amber-400 focus-within:ring-1 focus-within:ring-amber-400 transition-all">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onInput={handleInput}
        disabled={disabled}
        placeholder="Ask a question about Gurbani…"
        rows={1}
        className="flex-1 resize-none bg-transparent text-sm text-stone-800 placeholder-stone-400 outline-none leading-relaxed max-h-40 disabled:opacity-50"
        aria-label="Your question"
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || isEmpty}
        className={`flex-none w-8 h-8 rounded-full flex items-center justify-center transition-all
          ${
            disabled || isEmpty
              ? "bg-stone-200 text-stone-400 cursor-not-allowed"
              : "bg-amber-500 text-white hover:bg-amber-600 active:scale-95 shadow-sm"
          }`}
        aria-label="Send"
      >
        {disabled ? (
          <span className="w-4 h-4 border-2 border-stone-400 border-t-transparent rounded-full animate-spin" />
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-4 h-4"
          >
            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
          </svg>
        )}
      </button>
    </div>
  );
}
