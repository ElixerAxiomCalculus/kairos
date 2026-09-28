import * as Dialog from "@radix-ui/react-dialog";
import * as Tabs from "@radix-ui/react-tabs";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { useStream } from "@/stores/streamStore";
import { useUi } from "@/stores/uiStore";
import { transition } from "@/tokens/motion";
import { DASH, clock, ms } from "@/utils/format";
import styles from "./timeline.module.css";

/** Level 3 (formatted schema) and level 4 (raw JSON) telemetry for one event. */
export function EventInspector() {
  const open = useUi((s) => s.eventInspectorOpen);
  const selection = useUi((s) => s.selection);
  const close = useUi((s) => s.closeEventInspector);
  const events = useStream((s) => s.events);
  const ev = selection?.kind === "event" ? events.find((e) => e.event_id === selection.id) : undefined;

  return (
    <Dialog.Root open={open && Boolean(ev)} onOpenChange={(o) => !o && close()} modal={false}>
      <AnimatePresence>
        {open && ev ? (
          <Dialog.Portal forceMount>
            <Dialog.Content asChild forceMount onInteractOutside={(e) => e.preventDefault()} aria-describedby={undefined}>
              <motion.div className={styles.sheet} initial={{ y: 16, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
                exit={{ y: 16, opacity: 0 }} transition={transition.normal} data-testid="event-inspector">
                <Tabs.Root defaultValue="schema" style={{ display: "contents" }}>
                  <header className="panel-head">
                    <Dialog.Title className="px" style={{ margin: 0, fontWeight: 400 }}>
                      {ev.event_id.toUpperCase()} · {ev.event_type}
                    </Dialog.Title>
                    <div style={{ display: "flex", gap: 8 }}>
                      <Tabs.List className={styles.tabs} aria-label="Event view">
                        <Tabs.Trigger value="schema" className="btn px">SCHEMA</Tabs.Trigger>
                        <Tabs.Trigger value="json" className="btn px">JSON</Tabs.Trigger>
                      </Tabs.List>
                      <Dialog.Close className="btn btn-icon" aria-label="Close event inspector"><X size={15} /></Dialog.Close>
                    </div>
                  </header>
                  <div className={styles.sheetBody}>
                    <Tabs.Content value="schema">
                      <dl className={styles.schema}>
                        <dt>timestamp</dt><dd>{clock(ev.timestamp)} ({ev.timestamp.toFixed(4)} s)</dd>
                        <dt>session_id</dt><dd>{ev.session_id}</dd>
                        <dt>event_type</dt><dd>{ev.event_type}</dd>
                        <dt>parent_event_ids</dt><dd>{ev.parent_event_ids.join(", ") || DASH}</dd>
                        <dt>utterance_id</dt><dd>{ev.utterance_id ?? DASH}</dd>
                        <dt>query_version</dt><dd>{ev.query_version ?? DASH}</dd>
                        <dt>answer_version</dt><dd>{ev.answer_version ?? DASH}</dd>
                        <dt>retrieval_trigger</dt><dd>{ev.retrieval_trigger ?? DASH}</dd>
                        <dt>sub_query_id</dt><dd>{ev.sub_query_id ?? DASH}</dd>
                        <dt>document_ids</dt><dd>{ev.document_ids.join(", ") || DASH}</dd>
                        <dt>chunk_ids</dt><dd>{ev.chunk_ids.join(", ") || DASH}</dd>
                        <dt>latency</dt><dd>{ms(ev.latency_ms ?? undefined)}</dd>
                        <dt>token_cost</dt><dd>{Object.keys(ev.token_cost).length ? JSON.stringify(ev.token_cost) : DASH}</dd>
                        <dt>confidence</dt><dd>{ev.confidence ?? DASH}</dd>
                        <dt>citations</dt><dd>{ev.citations.join(", ") || DASH}</dd>
                      </dl>
                    </Tabs.Content>
                    <Tabs.Content value="json">
                      <pre className={styles.json}>{JSON.stringify(ev, null, 2)}</pre>
                    </Tabs.Content>
                  </div>
                </Tabs.Root>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        ) : null}
      </AnimatePresence>
    </Dialog.Root>
  );
}
