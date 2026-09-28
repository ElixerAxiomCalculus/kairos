/**
 * Acceptance flow over the recorded Pune demo (a real engine trace):
 * WAIT → RETRIEVE → multi-intent → evidence → V1 → late constraint (REFINE) → V2 → SUPPRESS → V3.
 */
describe("Kairos live workspace — recorded demo", () => {
  it("loads the landing screen and enters the replay", () => {
    cy.intercept("GET", "/api/**", { statusCode: 503, body: {} });
    cy.visit("/");
    cy.get("[data-testid=landing]").should("contain.text", "before");
    cy.screenshot("01-landing", { capture: "viewport" });
    cy.get("[data-testid=replay-demo]").click();
    cy.location("pathname").should("eq", "/live");
    cy.get("[data-testid=workspace]").should("exist");
    cy.get("[data-testid=connection-status]").should("contain.text", "RE"); // REPLAY or MOCK
  });

  it("shows WAIT on an unstable prefix, then RETRIEVE before the utterance ends", () => {
    cy.openDemo();
    cy.get("[data-testid=controller-state]").should("have.text", "WAIT");
    cy.screenshot("02-empty", { capture: "viewport" });
    cy.nextChunk(2); // "I need to" · "book a"
    cy.get("[data-testid=transcript-chunk]").should("have.length", 2);
    cy.get("[data-testid=controller-state]").should("have.text", "WAIT");
    cy.nextChunk(1); // "workshop venue in Pune"
    cy.get("[data-testid=controller-state]").should("have.text", "RETRIEVE");
    cy.get("[data-testid=retrieval-marker]").should("exist");
    // the utterance is still open (no answer yet) while retrieval has fired
    cy.get("[data-testid=answer-subtitle]").should("contain.text", "NO ANSWER VERSION YET");
    cy.screenshot("03-early-retrieve", { capture: "viewport" });
  });

  it("decomposes three intents, retrieves evidence, and produces grounded V1", () => {
    cy.openDemo();
    cy.nextChunk(6);
    cy.get("[data-testid=intent-node]").should("have.length", 3);
    cy.get("[data-testid=graph-node-chunk]").should("have.length.greaterThan", 3);
    cy.get("[data-testid=answer-subtitle]").should("contain.text", "ANSWER V01");
    cy.get("[data-testid=claim]").should("have.length.greaterThan", 3);
    cy.get("[data-testid=grounding-firewall]").should("have.attr", "data-status", "PASS");
    cy.wait(1000);
    cy.screenshot("04-v1", { capture: "viewport" });
  });

  it("handles the late constraint with REFINE and a delta update (V2)", () => {
    cy.openDemo();
    cy.get("[data-testid=jump-v1]").click();
    cy.get("[data-testid=claim]").then(($v1) => {
      const v1Texts = [...$v1].map((el) => el.textContent);
      cy.nextChunk(2); // "Actually, this is for an" · "international event."
      cy.get("[data-testid=answer-subtitle]").should("contain.text", "ANSWER V02");
      cy.get("[data-testid=delta-banner]").should("be.visible");
      cy.get("[data-testid=claims-invalidated]").invoke("text").should("match", /\d+ \/ \d+/);
      cy.get("[data-testid=delta-retrievals]").should("contain.text", "2 ops");
      cy.get("[data-testid=claim]").then(($v2) => {
        const v2Texts = [...$v2].map((el) => el.textContent);
        expect(v2Texts).to.not.deep.equal(v1Texts);
        // unaffected claims are carried verbatim
        expect(v2Texts.some((t) => v1Texts.includes(t))).to.equal(true);
      });
      cy.wait(800);
      cy.screenshot("05-v2-delta", { capture: "viewport" });
    });
  });

  it("suppresses retrieval for a presentation-only request (V3)", () => {
    cy.openDemo();
    cy.get("[data-testid=jump-v2]").click();
    cy.get("[data-testid=metric-retrievals]").invoke("text").then((before) => {
      cy.get("[data-testid=jump-suppress]").click();
      cy.get("[data-testid=controller-state]").should("have.text", "SUPPRESS");
      cy.get("[data-testid=jump-v3]").click();
      cy.get("[data-testid=answer-subtitle]").should("contain.text", "ANSWER V03");
      cy.get("[data-testid=suppression-sequence]").should("contain.text", "SKIPPED");
      // no retrieval event happened between V2 and V3
      cy.get("[data-testid=metric-retrievals]").invoke("text").should("eq", before);
      cy.wait(600);
      cy.screenshot("06-v3-suppress", { capture: "viewport" });
    });
  });

  it("opens the evidence inspector from a citation and navigates versions", () => {
    cy.openDemo();
    cy.get("[data-testid=jump-v3]").click();
    cy.get("[data-testid=citation-anchor]").first().click();
    cy.get("[data-testid=evidence-inspector]").should("be.visible");
    cy.get("[data-testid=inspector-passage]").should("not.be.empty");
    cy.screenshot("07-inspector", { capture: "viewport" });
    cy.get("[data-testid=version-1]").click();
    cy.get("[data-testid=answer-body]").should("have.attr", "data-version", "1");
  });

  it("surfaces contradictory evidence instead of picking one", () => {
    cy.openDemo("conflict-demo");
    cy.get("[data-testid=jump-v1]").click();
    cy.get("[data-testid=conflict-banner]").should("contain.text", "CONFLICT DETECTED");
    cy.get("[data-testid=claim]").should("have.length", 2);
    cy.get("[data-testid=graph-filter-CONFLICTS]").click();
    cy.wait(600);
    cy.screenshot("08-conflict", { capture: "viewport" });
  });

  it("opens event details from the timeline and the command palette", () => {
    cy.openDemo();
    cy.get("[data-testid=jump-v2]").click();
    cy.get("[data-testid=timeline-event][data-event-type=DeltaPlanComputed]").first().click({ force: true });
    cy.get("[data-testid=event-inspector]").should("contain.text", "DeltaPlanComputed");
    cy.screenshot("09-event-inspector", { capture: "viewport" });
    cy.get("body").type("{ctrl}k");
    cy.get("[data-testid=command-palette]").should("be.visible").find("input").type("benchmark mode{enter}");
    cy.location("pathname").should("eq", "/benchmark");
  });
});

describe("Benchmark and supporting views", () => {
  it("never shows targets as measurements", () => {
    cy.intercept("GET", "/api/**", { statusCode: 503, body: {} });
    cy.visit("/benchmark");
    cy.get("[data-testid=gate-table]").should("contain.text", "TARGET").and("contain.text", "MEASURED");
    cy.get("[data-testid=benchmark-provenance]").should("contain.text", "run");
    cy.screenshot("10-benchmark", { capture: "viewport" });
    cy.visit("/experiments");
    cy.get("[data-testid=experiment-D]").should("contain.text", "NOT RUN");
    cy.screenshot("11-experiments", { capture: "viewport" });
    cy.visit("/architecture");
    cy.get("[data-testid=architecture-map]").should("exist");
  });

  it("switches focus modes and theme", () => {
    cy.openDemo();
    cy.get("[data-testid=jump-v2]").click();
    cy.get("[data-testid=focus-evidence]").click();
    cy.get("[data-testid=workspace]").should("have.attr", "data-focus", "evidence");
    cy.wait(700);
    cy.screenshot("12-evidence-focus", { capture: "viewport" });
    cy.get("[data-testid=theme-toggle]").click();
    cy.get("html").should("have.attr", "data-theme", "light");
    cy.get("[data-testid=focus-split]").click();
    cy.wait(700);
    cy.screenshot("13-light-split", { capture: "viewport" });
  });
});
