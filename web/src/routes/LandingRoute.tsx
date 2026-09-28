import { motion } from "framer-motion";
import { BRAND } from "@/app/brand";
import { navigate } from "@/app/router";
import { BranchLine } from "@/components/common/BranchLine";
import { startLive, startMock } from "@/features/sessions/sessionController";
import { EASE } from "@/tokens/motion";
import styles from "./routes.module.css";

/** Typography is the hero. No illustration, no gradients. */
export default function LandingRoute() {
  const enter = async () => {
    navigate("/live");
    const ok = await startLive();
    if (!ok) await startMock();
  };
  const replay = async () => {
    navigate("/live");
    await startMock();
  };
  return (
    <main className={styles.landing} data-testid="landing">
      <motion.div className={`${styles.eyebrow} px`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6 }}>
        <span>{BRAND.eyebrow}</span>
        <BranchLine width={64} height={12} />
        <span>{BRAND.name}</span>
      </motion.div>
      <motion.h1 className={styles.hero} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: EASE.standard }}>
        {BRAND.hero[0]}<br /><em>{BRAND.hero[1]}</em> {BRAND.hero[2]}
      </motion.h1>
      <motion.p className={styles.lede} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25, duration: 0.6 }}>
        {BRAND.lede}
      </motion.p>
      <div className={styles.cta}>
        <button className="btn btn-primary px" onClick={enter} data-testid="enter-live">ENTER LIVE SYSTEM</button>
        <button className="btn px" onClick={replay} data-testid="replay-demo">REPLAY DEMO</button>
        <button className="btn px" onClick={() => navigate("/benchmark")}>BENCHMARK</button>
      </div>
      <section className={styles.principles} aria-label="Principles">
        {BRAND.principles.map((p, i) => (
          <motion.div key={p.key} className={styles.principle} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 + i * 0.08, duration: 0.5 }}>
            <span className="px">{String(i + 1).padStart(2, "0")} {p.key}</span>
            <p>{p.text}</p>
          </motion.div>
        ))}
      </section>
    </main>
  );
}
