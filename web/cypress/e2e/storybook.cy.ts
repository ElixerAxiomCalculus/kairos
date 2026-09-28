const stories = [
  ["retrievalcontroller--retrieve", "RETRIEVE"],
  ["retrievalcontroller--suppress", "SUPPRESS"],
  ["groundingfirewall--fail", "BLOCKED"],
  ["workspace--answer-delta-update", "DELTA UPDATE"],
  ["workspace--evidence-graph-conflict", "EVIDENCE GRAPH"],
  ["workspace--intent-graph-multi", "3 ACTIVE INTENTS"],
];
describe("Storybook stories render (opt-in: STORYBOOK=1, serve storybook-static on :6007)", () => {
  before(function () {
    if (!Cypress.env("STORYBOOK")) this.skip();
  });
  for (const [id, text] of stories) {
    it(id, () => {
      cy.visit(`http://localhost:6007/iframe.html?id=${id}&viewMode=story`);
      cy.contains(text, { timeout: 15000 }).should("exist");
      cy.wait(400);
      cy.screenshot(`sb-${id}`, { capture: "viewport" });
    });
  }
});
