import type { Meta, StoryObj } from "@storybook/react";
import { RetrievalController } from "@/components/controller/RetrievalController";
import { at, viewAt } from "@/test/recorded";

const meta: Meta<typeof RetrievalController> = {
  title: "RetrievalController",
  component: RetrievalController,
  decorators: [(Story) => <div style={{ width: 260, height: 250, display: "grid" }}><Story /></div>],
};
export default meta;
type Story = StoryObj<typeof RetrievalController>;

const decision = (state: string) => {
  const v = viewAt("pune", at.decision(state));
  return { decision: v.controller, queryVersion: v.queryVersions.at(-1)?.version ?? 0 };
};

export const WAIT: Story = { args: decision("WAIT") };
export const RETRIEVE: Story = { args: decision("RETRIEVE") };
export const SUPPRESS: Story = { args: decision("SUPPRESS") };
export const REFINE: Story = { args: decision("REFINE") };
