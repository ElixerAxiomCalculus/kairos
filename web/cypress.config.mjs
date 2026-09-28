import { defineConfig } from "cypress";

export default defineConfig({
  e2e: {
    baseUrl: "http://localhost:4173",
    viewportWidth: 1440,
    viewportHeight: 900,
    video: false,
    screenshotsFolder: "cypress/screenshots",
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 10000,
    supportFile: "cypress/support/e2e.ts",
    specPattern: "cypress/e2e/**/*.cy.ts",
    setupNodeEvents(on) {
      on("before:browser:launch", (browser, launchOptions) => {
        if (browser.name === "electron") {
          launchOptions.preferences.width = 1440;
          launchOptions.preferences.height = 900;
        }
        return launchOptions;
      });
    },
  },
});
