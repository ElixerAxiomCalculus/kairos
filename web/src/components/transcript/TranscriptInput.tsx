import { Mic, Send } from "lucide-react";
import { useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { nextUtteranceId, sendSpeechChunk, streamText } from "@/features/sessions/sessionController";
import styles from "./transcript.module.css";

interface FormValues {
  query: string;
}

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((e: { resultIndex: number; results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }> }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

function speechCtor(): (new () => SpeechRecognitionLike) | null {
  const w = window as unknown as Record<string, unknown>;
  return (w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null) as (new () => SpeechRecognitionLike) | null;
}

/** Live input: typed text is streamed in 3-word chunks; the browser mic streams interim ASR words. */
export function TranscriptInput() {
  const { register, handleSubmit, reset, formState } = useForm<FormValues>({ defaultValues: { query: "" } });
  const [listening, setListening] = useState(false);
  const recRef = useRef<SpeechRecognitionLike | null>(null);
  const sentRef = useRef({ uid: "", words: 0 });
  const Ctor = speechCtor();

  const onSubmit = handleSubmit(async ({ query }) => {
    if (!query.trim()) return;
    reset();
    await streamText(query);
  });

  const toggleMic = () => {
    if (!Ctor) return;
    if (listening) {
      recRef.current?.stop();
      return;
    }
    const rec = new Ctor();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-IN";
    sentRef.current = { uid: nextUtteranceId("mic"), words: 0 };
    rec.onresult = (e) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        const words = res[0].transcript.trim().split(/\s+/).filter(Boolean);
        const fresh = words.slice(sentRef.current.words);
        // send only stable growth: interim results are sent in groups of 3 words, finals flush the rest
        if (res.isFinal || fresh.length >= 3) {
          sendSpeechChunk(sentRef.current.uid, fresh.join(" "), res.isFinal);
          sentRef.current.words = res.isFinal ? 0 : words.length;
          if (res.isFinal) sentRef.current.uid = nextUtteranceId("mic");
        }
      }
    };
    rec.onend = () => setListening(false);
    recRef.current = rec;
    rec.start();
    setListening(true);
  };

  return (
    <form className={styles.input} onSubmit={onSubmit} aria-label="Stream a query">
      <input
        {...register("query", { maxLength: 400 })}
        placeholder="Type a question — it streams in chunks"
        aria-label="Query text"
        autoComplete="off"
      />
      {Ctor ? (
        <button type="button" className={`btn btn-icon ${listening ? styles.micOn : ""}`} onClick={toggleMic}
          aria-pressed={listening} aria-label={listening ? "Stop microphone" : "Speak (browser speech recognition)"}>
          <Mic size={15} />
        </button>
      ) : null}
      <button type="submit" className="btn btn-icon" aria-label="Stream query" disabled={formState.isSubmitting}>
        <Send size={15} />
      </button>
    </form>
  );
}
