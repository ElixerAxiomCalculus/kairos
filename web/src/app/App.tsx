import * as Tooltip from "@radix-ui/react-tooltip";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { lazy, Suspense, useEffect } from "react";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import LandingRoute from "@/routes/LandingRoute";
import WorkspaceRoute from "@/routes/WorkspaceRoute";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import { CommandPalette } from "./CommandPalette";
import { useRoute } from "./router";
import styles from "./shell.module.css";
import { TopBar } from "./TopBar";

const BenchmarkRoute = lazy(() => import("@/routes/BenchmarkRoute"));
const ExperimentsRoute = lazy(() => import("@/routes/ExperimentsRoute"));
const ArchitectureRoute = lazy(() => import("@/routes/ArchitectureRoute"));

function Errors() {
  const errors = useStream((s) => s.errors);
  const recent = errors.slice(-3);
  return (
    <div className={styles.errors} aria-live="assertive">
      <AnimatePresence>
        {recent.map((e) => (
          <motion.div key={e.at + e.message} className={styles.toast} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }} role="alert" data-testid="error-toast">
            <span style={{ flex: 1 }}>{e.message}</span>
            <button className="btn btn-icon" style={{ height: 22, width: 22 }} aria-label="Dismiss"
              onClick={() => useStream.setState((s) => ({ errors: s.errors.filter((x) => x !== e) }))}><X size={12} /></button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

export function App() {
  const route = useRoute();
  const theme = useUi((s) => s.theme);
  const presentation = useUi((s) => s.presentation);
  useKeyboardShortcuts(true);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    // auto-expire errors so toasts don't accumulate during a demo
    const t = window.setInterval(() => {
      const cutoff = Date.now() - 8000;
      const errs = useStream.getState().errors;
      if (errs.some((e) => e.at < cutoff)) useStream.setState({ errors: errs.filter((e) => e.at >= cutoff) });
    }, 2000);
    return () => window.clearInterval(t);
  }, []);

  return (
    <Tooltip.Provider>
      <div className={`app ${presentation ? "presentation" : ""}`}>
        <TopBar />
        <Suspense fallback={<div className="empty"><span className="px">LOADING VIEW</span></div>}>
          {route === "/" ? <LandingRoute /> : null}
          {route === "/live" ? <WorkspaceRoute /> : null}
          {route === "/benchmark" ? <BenchmarkRoute /> : null}
          {route === "/experiments" ? <ExperimentsRoute /> : null}
          {route === "/architecture" ? <ArchitectureRoute /> : null}
        </Suspense>
        <CommandPalette />
        <Errors />
      </div>
    </Tooltip.Provider>
  );
}
