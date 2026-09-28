import { ArchitectureMap } from "@/components/architecture/ArchitectureMap";
import styles from "./routes.module.css";

export default function ArchitectureRoute() {
  return (
    <main className={styles.page} data-testid="architecture-page">
      <div className={styles.pageHead}>
        <div>
          <span className="px muted">SYSTEM ARCHITECTURE</span>
          <h1 className={styles.pageTitle}>One process, <em>every decision</em> traced</h1>
        </div>
      </div>
      <ArchitectureMap />
    </main>
  );
}
