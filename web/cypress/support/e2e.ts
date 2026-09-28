/// <reference types="cypress" />

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Cypress {
    interface Chainable {
      /** Open the workspace on the recorded Pune demo, paused at the start. */
      openDemo(script?: string): Chainable<void>;
      /** Advance one transcript chunk (plus the processing burst that follows it). */
      nextChunk(times?: number): Chainable<void>;
    }
  }
}

Cypress.Commands.add("openDemo", (script = "pune-demo") => {
  // no backend in e2e: /api fails, the app falls back to recorded engine traces (MOCK)
  cy.intercept("GET", "/api/**", { statusCode: 503, body: {} });
  cy.visit("/live?autoplay=0");
  cy.get("[data-testid=script-select]").should("exist");
  if (script !== "pune-demo") cy.get("[data-testid=script-select]").select(script);
  cy.get("[data-testid=play-toggle]").should("exist");
  cy.get("[data-testid=timeline-event]").should("have.length.greaterThan", 10);
});

Cypress.Commands.add("nextChunk", (times = 1) => {
  for (let i = 0; i < times; i++) cy.get("[data-testid=next-chunk]").click();
});

export {};
