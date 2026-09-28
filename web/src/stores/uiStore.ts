import { create } from "zustand";

export type FocusMode = "split" | "answer" | "evidence" | "systems";
export type GraphFilter = "ALL" | "ACTIVE" | "CONFLICTS" | "UPDATED" | "UNSUPPORTED";
export type Theme = "dark" | "light";

export type Selection =
  | { kind: "claim"; id: string }
  | { kind: "chunk"; id: string; claimId?: string }
  | { kind: "intent"; id: string }
  | { kind: "event"; id: string }
  | null;

const THEME_KEY = "kairos.theme";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage unavailable: fall back to dark */
  }
  return "dark";
}

export interface UiState {
  theme: Theme;
  focus: FocusMode;
  demoMode: boolean;
  presentation: boolean;
  selection: Selection;
  hover: Selection;
  inspectorOpen: boolean;
  eventInspectorOpen: boolean;
  paletteOpen: boolean;
  graphFilter: GraphFilter;
  viewVersion: number | null; // null = follow latest
  compareVersion: number | null;
  showIntentGraph: boolean;
  showEvidenceGraph: boolean;
  showTelemetry: boolean;
  endSessionOpen: boolean;

  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
  setFocus: (f: FocusMode) => void;
  toggleDemo: () => void;
  togglePresentation: () => void;
  select: (s: Selection) => void;
  setHover: (s: Selection) => void;
  closeInspector: () => void;
  closeEventInspector: () => void;
  setPaletteOpen: (o: boolean) => void;
  setGraphFilter: (f: GraphFilter) => void;
  setViewVersion: (v: number | null) => void;
  setCompareVersion: (v: number | null) => void;
  toggle: (k: "showIntentGraph" | "showEvidenceGraph" | "showTelemetry") => void;
  setEndSessionOpen: (o: boolean) => void;
}

export const useUi = create<UiState>((set, get) => ({
  theme: initialTheme(),
  focus: "split",
  demoMode: false,
  presentation: false,
  selection: null,
  hover: null,
  inspectorOpen: false,
  eventInspectorOpen: false,
  paletteOpen: false,
  graphFilter: "ALL",
  viewVersion: null,
  compareVersion: null,
  showIntentGraph: true,
  showEvidenceGraph: true,
  showTelemetry: true,
  endSessionOpen: false,

  setTheme: (theme) => {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* ignore */
    }
    document.documentElement.dataset.theme = theme;
    set({ theme });
  },
  toggleTheme: () => get().setTheme(get().theme === "dark" ? "light" : "dark"),
  setFocus: (focus) => set({ focus }),
  toggleDemo: () => set((s) => ({ demoMode: !s.demoMode })),
  togglePresentation: () => set((s) => ({ presentation: !s.presentation })),
  select: (selection) =>
    set({
      selection,
      inspectorOpen: selection !== null && selection.kind !== "event",
      eventInspectorOpen: selection?.kind === "event",
    }),
  setHover: (hover) => set({ hover }),
  closeInspector: () => set({ inspectorOpen: false, selection: null }),
  closeEventInspector: () => set({ eventInspectorOpen: false, selection: null }),
  setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
  setGraphFilter: (graphFilter) => set({ graphFilter }),
  setViewVersion: (viewVersion) => set({ viewVersion }),
  setCompareVersion: (compareVersion) => set({ compareVersion }),
  toggle: (k) => set((s) => ({ [k]: !s[k] }) as Partial<UiState>),
  setEndSessionOpen: (endSessionOpen) => set({ endSessionOpen }),
}));
