/**
 * Live backend flow (needs `python -m kairos serve` on :8000). Run with:
 *   npx cypress run --spec cypress/e2e/live.cy.ts --config baseUrl=http://127.0.0.1:8000 --env LIVE=1
 */
describe("Live backend over WebSocket", () => {
  before(function () {
    if (!Cypress.env("LIVE")) this.skip();
  });

  it("streams a script through the real engine and renders every stage", () => {
    cy.visit("/live?autoplay=0");
    cy.get("[data-testid=mode-toggle]").should("contain.text", "GO LIVE").click();
    cy.get("[data-testid=connection-status]", { timeout: 15000 }).should("contain.text", "LIVE");
    cy.get("[data-testid=live-script-select]").select("pune-demo");
    cy.contains("button", "STREAM TO ENGINE").click();
    cy.get("[data-testid=controller-state]", { timeout: 20000 }).should("have.text", "RETRIEVE");
    cy.get("[data-testid=answer-subtitle]", { timeout: 30000 }).should("contain.text", "ANSWER V01");
    cy.get("[data-testid=answer-subtitle]", { timeout: 30000 }).should("contain.text", "ANSWER V02");
    cy.get("[data-testid=delta-banner]").should("be.visible");
    cy.get("[data-testid=answer-subtitle]", { timeout: 30000 }).should("contain.text", "ANSWER V03");
    cy.get("[data-testid=suppression-sequence]").should("contain.text", "SKIPPED");
    cy.wait(800);
    cy.screenshot("live-end-to-end", { capture: "viewport" });
  });
});
