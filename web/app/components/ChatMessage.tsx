"use client";

import ReactMarkdown from "react-markdown";
import type { Message, Source } from "../page";

interface ChatMessageProps {
  message: Message;
}

function SourceChip({ source }: { source: Source }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-amber-100 text-amber-800 border border-amber-200">
      <span className="font-semibold">Ang {source.ang}</span>
      <span className="text-amber-600">·</span>
      <span>{source.raag}</span>
      <span className="text-amber-600">·</span>
      <span className="truncate max-w-[120px]">{source.writer}</span>
    </span>
  );
}

const AGENT_LABELS: Record<string, { label: string; color: string }> = {
  conceptual:  { label: "Conceptual",  color: "bg-sky-100 text-sky-700 border-sky-200" },
  situational: { label: "Guidance",    color: "bg-emerald-100 text-emerald-700 border-emerald-200" },
  comparative: { label: "Comparative", color: "bg-violet-100 text-violet-700 border-violet-200" },
  rehat:       { label: "Conduct",     color: "bg-orange-100 text-orange-700 border-orange-200" },
  adversarial: { label: "Out of scope",color: "bg-stone-100 text-stone-500 border-stone-200" },
};

function AgentBadge({ type }: { type: string }) {
  const info = AGENT_LABELS[type] ?? AGENT_LABELS.conceptual;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs border ${info.color}`}>
      {info.label}
    </span>
  );
}

function FailedQuoteNotice({ count }: { count: number }) {
  if (count === 0) return null;
  return (
    <div className="mt-2 px-3 py-2 rounded-md bg-red-50 border border-red-200 text-red-700 text-xs flex items-start gap-2">
      <span className="flex-none mt-0.5">⚠</span>
      <span>
        {count} quote{count > 1 ? "s" : ""} could not be verified against Sri
        Guru Granth Sahib Ji and{" "}
        {count > 1 ? "have" : "has"} been removed.
      </span>
    </div>
  );
}

/** Detect Gurmukhi Unicode block (U+0A00–U+0A7F) and apply the correct font. */
function RenderLine({ text }: { text: string }) {
  const GURMUKHI_RE = /[਀-੿]+(?:\s[਀-੿]+)*/g;
  const segments: { gurmukhi: boolean; text: string }[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = GURMUKHI_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ gurmukhi: false, text: text.slice(lastIndex, match.index) });
    }
    segments.push({ gurmukhi: true, text: match[0] });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    segments.push({ gurmukhi: false, text: text.slice(lastIndex) });
  }

  return (
    <>
      {segments.map((seg, i) =>
        seg.gurmukhi ? (
          <span key={i} className="gurmukhi text-stone-800 font-medium">
            {seg.text}
          </span>
        ) : (
          <span key={i}>{seg.text}</span>
        )
      )}
    </>
  );
}

/** Process ReactNode children: apply RenderLine to string leaves. */
function withGurmukhi(children: React.ReactNode): React.ReactNode {
  if (typeof children === "string") return <RenderLine text={children} />;
  if (Array.isArray(children)) {
    return children.map((child, i) =>
      typeof child === "string" ? <RenderLine key={i} text={child} /> : child
    );
  }
  return children;
}

/** Render Markdown answer with Gurmukhi font detection. */
function AnswerText({ content }: { content: string }) {
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => (
          <p className="mb-3 last:mb-0 leading-relaxed">{withGurmukhi(children)}</p>
        ),
        h1: ({ children }) => (
          <h1 className="font-semibold text-stone-800 text-base mt-4 mb-1">{withGurmukhi(children)}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="font-semibold text-stone-800 text-base mt-4 mb-1">{withGurmukhi(children)}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="font-semibold text-stone-800 mt-3 mb-1">{withGurmukhi(children)}</h3>
        ),
        h4: ({ children }) => (
          <h4 className="font-semibold text-stone-700 mt-3 mb-1">{withGurmukhi(children)}</h4>
        ),
        strong: ({ children }) => (
          <strong className="font-semibold text-stone-800">{withGurmukhi(children)}</strong>
        ),
        em: ({ children }) => (
          <em className="italic text-stone-600">{withGurmukhi(children)}</em>
        ),
        ul: ({ children }) => (
          <ul className="list-disc pl-5 space-y-1 mb-3">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal pl-5 space-y-1 mb-3">{children}</ol>
        ),
        li: ({ children }) => (
          <li className="leading-relaxed">{withGurmukhi(children)}</li>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-amber-400 pl-3 my-2 text-stone-600 italic">
            {children}
          </blockquote>
        ),
        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-amber-700 underline underline-offset-2 hover:text-amber-900"
          >
            {children}
          </a>
        ),
        hr: () => <hr className="border-stone-200 my-3" />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const isLoading = message.loading;

  if (isUser) {
    return (
      <div className="flex justify-end message-enter">
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-amber-500 text-white px-4 py-2.5 text-sm shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  // Assistant bubble
  return (
    <div className="flex justify-start message-enter">
      <div className="max-w-[90%] space-y-2">
        {/* Avatar + bubble */}
        <div className="flex items-start gap-2">
          <div className="flex-none w-7 h-7 rounded-full bg-stone-700 flex items-center justify-center text-amber-300 text-xs font-semibold select-none gurmukhi mt-0.5">
            ੴ
          </div>

          <div className="rounded-2xl rounded-tl-sm bg-white border border-stone-200 px-4 py-3 text-sm text-stone-700 shadow-sm leading-relaxed min-w-[60px]">
            {isLoading ? (
              <span className="flex items-center gap-1.5 text-stone-400">
                <span className="animate-bounce [animation-delay:0ms]">·</span>
                <span className="animate-bounce [animation-delay:150ms]">·</span>
                <span className="animate-bounce [animation-delay:300ms]">·</span>
              </span>
            ) : (
              <AnswerText content={message.content} />
            )}
          </div>
        </div>

        {/* Agent type + sources */}
        {!isLoading && (
          <div className="ml-9 flex flex-wrap gap-1.5 items-center">
            {message.questionType && (
              <AgentBadge type={message.questionType} />
            )}
            {message.sources && message.sources.map((src) => (
              <SourceChip key={`${src.shabad_id}-${src.ang}`} source={src} />
            ))}
          </div>
        )}

        {/* Failed quote notice */}
        {!isLoading &&
          message.failedQuotes &&
          message.failedQuotes.length > 0 && (
            <div className="ml-9">
              <FailedQuoteNotice count={message.failedQuotes.length} />
            </div>
          )}
      </div>
    </div>
  );
}
