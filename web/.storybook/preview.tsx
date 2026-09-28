import "@fontsource/silkscreen/400.css";
import "@fontsource/cormorant-garamond/400.css";
import "@fontsource/cormorant-garamond/500.css";
import "@fontsource/cormorant-garamond/500-italic.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/ibm-plex-mono/400.css";
import "../src/styles/global.css";

import * as Tooltip from "@radix-ui/react-tooltip";
import type { Preview } from "@storybook/react";

const preview: Preview = {
  globalTypes: {
    theme: {
      description: "Kairos theme",
      toolbar: { title: "Theme", items: ["dark", "light"], dynamicTitle: true },
    },
  },
  initialGlobals: { theme: "dark" },
  parameters: { layout: "fullscreen", backgrounds: { disable: true } },
  decorators: [
    (Story, ctx) => {
      document.documentElement.dataset.theme = (ctx.globals.theme as string) ?? "dark";
      return (
        <Tooltip.Provider>
          <div style={{ background: "var(--surface-base)", minHeight: "100vh", padding: 16, color: "var(--text-primary)" }}>
            <Story />
          </div>
        </Tooltip.Provider>
      );
    },
  ],
};

export default preview;
