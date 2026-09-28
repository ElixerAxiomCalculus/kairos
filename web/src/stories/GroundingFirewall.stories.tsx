import type { Meta, StoryObj } from "@storybook/react";
import { GroundingFirewall } from "@/components/telemetry/GroundingFirewall";
import { at, viewAt } from "@/test/recorded";
import type { GroundingResult } from "@/types/domain";

const meta: Meta<typeof GroundingFirewall> = {
  title: "GroundingFirewall",
  component: GroundingFirewall,
  decorators: [(Story) => <div style={{ width: 360 }}><Story /></div>],
};
export default meta;
type Story = StoryObj<typeof GroundingFirewall>;

const pass = viewAt("pune", at.version(1)).grounding.at(-1) ?? null;

/** Recorded: the firewall verdict for the Pune demo's first answer. */
export const PASS: Story = { args: { result: pass, transform: null } };

/**
 * Synthetic visual fixture (not an engine recording): the BLOCKED state for component review.
 * It mirrors what the firewall emits when a claim cites an ID that was never retrieved.
 */
const blocked: GroundingResult | null = pass
  ? {
      ...pass,
      status: "BLOCKED",
      unsupported: 1,
      invalidCitation: 1,
      fabricatedIdsBlocked: 1,
      emitted: pass.emitted - 1,
      results: [
        ...pass.results,
        { claim_id: "C99", verdict: "INVALID_CITATION", support: 0, anchors_ok: false, quote_ok: false,
          invalid_ids: ["D99:C01"], text: "(synthetic) claim citing an unknown document", evidence: [], repaired: false },
      ],
    }
  : null;
export const FAIL: Story = { args: { result: blocked, transform: null } };
