"use client";

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

/** Render answer text: plain text is shown as-is. */
function AnswerText({ content }: { content: string }) {
  // Split on double newlines for paragraphs
  const paragraphs = content.split(/\n{2,}/).filter(Boolean);
  if (paragraphs.length <= 1) {
    return <RenderLine text={content} />;
  }
  return (
    <>
      {paragraphs.map((para, i) => (
        <p key={i} className="mb-3 last:mb-0">
          <RenderLine text={para} />
        </p>
      ))}
    </>
  );
}

/** Render a single line, detecting Gurmukhi Unicode block (U+0A00–U+0A7F). */
function RenderLine({ text }: { text: string }) {
  // Segment text into Gurmukhi and Latin spans
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

        {/* Sources */}
        {!isLoading && message.sources && message.sources.length > 0 && (
          <div className="ml-9 flex flex-wrap gap-1.5">
            {message.sources.map((src) => (
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
