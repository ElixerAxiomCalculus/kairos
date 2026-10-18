import { Mic, Send, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { abandonVoiceTurn, beginVoiceTurn, sendVoiceSnapshot, streamText } from "@/features/sessions/sessionController";
import { useStream } from "@/stores/streamStore";
import styles from "./transcript.module.css";

interface FormValues { query: string }
type MicPhase = "IDLE" | "REQUESTING_PERMISSION" | "LISTENING" | "TRANSCRIBING" | "STOPPING" | "FINALIZING" | "ERROR";
type SpeechResult = { isFinal: boolean; 0: { transcript: string } };
type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onresult: ((event: { results: ArrayLike<SpeechResult> }) => void) | null;
  onerror: ((event: { error: string; message?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort?: () => void;
};

function speechCtor(): (new () => SpeechRecognitionLike) | null {
  const browser = window as unknown as Record<string, unknown>;
  return (browser.SpeechRecognition ?? browser.webkitSpeechRecognition ?? null) as (new () => SpeechRecognitionLike) | null;
}

function joinText(...parts: string[]): string {
  return parts.filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
}

/** Live input with explicit Start/Stop turn boundaries and cumulative ASR snapshots. */
export function TranscriptInput() {
  const { register, handleSubmit, reset, formState } = useForm<FormValues>({ defaultValues: { query: "" } });
  const connection = useStream((state) => state.connection);
  const pushError = useStream((state) => state.pushError);
  const [phase, setPhase] = useState<MicPhase>("IDLE");
  const [preview, setPreview] = useState("");
  const recRef = useRef<SpeechRecognitionLike | null>(null);
  const uidRef = useRef<string | null>(null);
  const baseTextRef = useRef("");
  const currentTextRef = useRef("");
  const desiredListeningRef = useRef(false);
  const stoppingRef = useRef(false);
  const finalizedRef = useRef(false);
  const restartTimerRef = useRef<number | null>(null);
  const Ctor = speechCtor();

  const onSubmit = handleSubmit(async ({ query }) => {
    if (!query.trim()) return;
    reset();
    await streamText(query);
  });

  const finishTurn = useCallback(() => {
    if (finalizedRef.current) return;
    finalizedRef.current = true;
    desiredListeningRef.current = false;
    stoppingRef.current = false;
    setPhase("FINALIZING");
    const uid = uidRef.current;
    if (uid && !sendVoiceSnapshot(uid, currentTextRef.current, true, true)) abandonVoiceTurn(uid);
    recRef.current = null;
    uidRef.current = null;
    baseTextRef.current = "";
    window.setTimeout(() => setPreview(""), 700);
    setPhase("IDLE");
  }, []);

  const startRecognizer = useCallback(() => {
    if (!Ctor || !uidRef.current || !desiredListeningRef.current) return;
    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-IN";
    recognition.onstart = () => setPhase("LISTENING");
    recognition.onresult = (event) => {
      const finalParts: string[] = [];
      const interimParts: string[] = [];
      for (let index = 0; index < event.results.length; index += 1) {
        const result = event.results[index];
        const transcript = result[0]?.transcript?.trim() ?? "";
        if (transcript) (result.isFinal ? finalParts : interimParts).push(transcript);
      }
      const fullText = joinText(baseTextRef.current, joinText(...finalParts, ...interimParts));
      currentTextRef.current = fullText;
      setPreview(fullText);
      setPhase(interimParts.length ? "TRANSCRIBING" : "LISTENING");
      const uid = uidRef.current;
      if (uid) sendVoiceSnapshot(uid, fullText, interimParts.length === 0 && finalParts.length > 0);
    };
    recognition.onerror = (event) => {
      if (stoppingRef.current && event.error === "aborted") return;
      if (event.error === "no-speech") return;
      desiredListeningRef.current = false;
      stoppingRef.current = true;
      const message = event.error === "not-allowed" || event.error === "service-not-allowed"
        ? "Microphone permission was denied. You can still type a query."
        : `Speech recognition stopped (${event.error}${event.message ? `: ${event.message}` : ""}).`;
      pushError(message);
      setPhase("ERROR");
      window.setTimeout(finishTurn, 300);
    };
    recognition.onend = () => {
      if (stoppingRef.current || !desiredListeningRef.current) return finishTurn();
      baseTextRef.current = currentTextRef.current;
      restartTimerRef.current = window.setTimeout(startRecognizer, 120);
    };
    recRef.current = recognition;
    try {
      recognition.start();
    } catch (error) {
      desiredListeningRef.current = false;
      stoppingRef.current = true;
      pushError(`Could not start speech recognition: ${(error as Error).message}`);
      finishTurn();
    }
  }, [Ctor, finishTurn, pushError]);

  const startMic = () => {
    if (!Ctor || phase !== "IDLE") return;
    const utteranceId = beginVoiceTurn();
    if (!utteranceId) return;
    uidRef.current = utteranceId;
    baseTextRef.current = "";
    currentTextRef.current = "";
    finalizedRef.current = false;
    stoppingRef.current = false;
    desiredListeningRef.current = true;
    setPreview("");
    setPhase("REQUESTING_PERMISSION");
    startRecognizer();
  };

  const stopMic = useCallback(() => {
    if (!uidRef.current || stoppingRef.current || finalizedRef.current) return;
    desiredListeningRef.current = false;
    stoppingRef.current = true;
    setPhase("STOPPING");
    try {
      recRef.current?.stop();
      window.setTimeout(finishTurn, 800);
    } catch {
      finishTurn();
    }
  }, [finishTurn]);

  useEffect(() => {
    if (connection !== "LIVE" && uidRef.current && !finalizedRef.current) {
      pushError("Live connection was interrupted; microphone input was stopped and queued for safe resume.");
      stopMic();
    }
  }, [connection, pushError, stopMic]);

  useEffect(() => () => {
    if (restartTimerRef.current !== null) window.clearTimeout(restartTimerRef.current);
    desiredListeningRef.current = false;
    if (uidRef.current && !finalizedRef.current) finishTurn();
    recRef.current?.abort?.();
  }, [finishTurn]);

  const active = !["IDLE", "ERROR"].includes(phase);
  return (
    <div className={styles.inputArea} data-mic-phase={phase}>
      {preview ? <div className={styles.preview} data-testid="provisional-transcript"><span className="px">PROVISIONAL</span> {preview}</div> : null}
      {!Ctor ? <div className={styles.preview}>Speech recognition is unavailable in this browser; use typed input.</div> : null}
      <form className={styles.input} onSubmit={onSubmit} aria-label="Stream a query">
        <input {...register("query", { maxLength: 400 })} placeholder="Type a question — it streams in chunks"
          aria-label="Query text" autoComplete="off" disabled={active} />
        {Ctor ? (
          <button type="button" className={`btn btn-icon ${active ? styles.micOn : ""}`}
            onClick={active ? stopMic : startMic} disabled={phase === "FINALIZING" || phase === "STOPPING"}
            aria-pressed={active} aria-label={active ? "Stop microphone" : "Start microphone"}
            data-testid="microphone-button" title={phase.replaceAll("_", " ")}>
            {active ? <Square size={13} /> : <Mic size={15} />}
          </button>
        ) : null}
        <button type="submit" className="btn btn-icon" aria-label="Stream query" disabled={formState.isSubmitting || active}>
          <Send size={15} />
        </button>
      </form>
    </div>
  );
}
