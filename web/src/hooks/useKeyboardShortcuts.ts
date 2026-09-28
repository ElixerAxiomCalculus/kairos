import { useEffect } from "react";
import { navigate } from "@/app/router";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

/** Global shortcuts. Ignored while typing in an input. */
export function useKeyboardShortcuts(enabled = true): void {
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      const ui = useUi.getState();
      const stream = useStream.getState();
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        ui.setPaletteOpen(!ui.paletteOpen);
        return;
      }
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey || ui.paletteOpen) return;
      switch (e.key) {
        case "1":
          ui.setFocus("answer");
          break;
        case "2":
          ui.setFocus("evidence");
          break;
        case "3":
          ui.setFocus("systems");
          break;
        case "0":
          ui.setFocus("split");
          break;
        case "d":
        case "D":
          ui.toggleDemo();
          break;
        case "p":
        case "P":
          ui.togglePresentation();
          break;
        case " ":
          e.preventDefault();
          stream.togglePlay();
          break;
        case "ArrowRight":
          stream.stepEvent(1);
          break;
        case "ArrowLeft":
          stream.stepEvent(-1);
          break;
        case "g":
        case "G":
          ui.setFocus("evidence");
          break;
        case "i":
        case "I":
          ui.setFocus("systems");
          break;
        case "t":
        case "T":
          ui.toggle("showTelemetry");
          break;
        case "b":
        case "B":
          navigate("/benchmark");
          break;
        case "Escape":
          ui.closeInspector();
          ui.closeEventInspector();
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}
