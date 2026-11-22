import { expect, test, type Page, type WebSocketRoute } from "@playwright/test";
import type { InboundEvent, ThreadProjectionEvent } from "../src/lib/types";

const chatId = "answer-order";

async function openThread(
  page: Page,
  withHistory: boolean,
  immediateReply = false,
  options: { shortHistory?: boolean; beforeSend?: () => Promise<void> } = {},
) {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => localStorage.setItem("nanobot.locale", "en"));
  await page.route("**/api/**", (route) => route.fulfill({ status: 404, json: {} }));
  await page.route("**/api/commands", (route) => route.fulfill({ json: { commands: [] } }));
  const events: ThreadProjectionEvent[] = withHistory ? Array.from({ length: options.shortHistory ? 1 : 12 }, (_, index) => [
    { event: "user_message" as const, chat_id: chatId, turn_id: `old-${index}`, turn_seq: 1, text: `Earlier question ${index}`, starts_turn: true },
    { event: "message" as const, chat_id: chatId, turn_id: `old-${index}`, turn_seq: 2, text: (immediateReply ? `## Earlier answer ${index}\n\nFirst paragraph.\n\nSecond paragraph.\n\n` : `Earlier answer ${index}. `).repeat(options.shortHistory ? 1 : 8) },
    { event: "turn_end" as const, chat_id: chatId, turn_id: `old-${index}`, turn_seq: 3 },
  ]).flat() : [];
  await page.route("**/api/sessions/*/webui-thread?*", (route) => {
    const path = decodeURIComponent(new URL(route.request().url()).pathname);
    return route.fulfill({ json: {
      schemaVersion: 3, projection: "events", events: path.includes(`/sessions/websocket:${chatId}/`) ? events : [],
      has_more: false, has_pending_tool_calls: false,
    } });
  });
  let socket!: WebSocketRoute;
  let attached = false;
  const sent: Array<{ type: string; turn_id?: string }> = [];
  await page.routeWebSocket("**/answer-order-socket", (route) => {
    socket = route;
    route.onMessage((raw) => {
      const frame = JSON.parse(String(raw)) as { type: string; chat_id: string; turn_id?: string };
      if (frame.type === "attach") {
        attached = true;
        route.send(JSON.stringify({ event: "attached", chat_id: frame.chat_id }));
      } else if (frame.type === "message") {
        sent.push(frame);
        if (immediateReply) {
          const base = { chat_id: chatId, turn_id: frame.turn_id, stream_id: "fast" };
          route.send(JSON.stringify({ event: "delta", ...base, turn_seq: 1, text: "Immediate answer" }));
          route.send(JSON.stringify({ event: "stream_end", ...base, turn_seq: 2 }));
          route.send(JSON.stringify({ event: "turn_end", chat_id: chatId, turn_id: frame.turn_id, turn_seq: 3 }));
        }
      }
    });
  });
  await page.goto("/browser/fixture.html?case=answer-order&sendable=1");
  await expect.poll(() => attached).toBe(true);
  await expect(page.getByRole("status", { name: "Loading conversation…" })).toHaveCount(0);
  const input = page.getByRole("textbox", { name: "Message input", exact: true });
  await input.fill("My anchored question");
  await options.beforeSend?.();
  await input.press("Enter");
  await expect.poll(() => sent.find((frame) => frame.turn_id)?.turn_id).toBeTruthy();
  const turnId = sent.find((frame) => frame.turn_id)!.turn_id!;
  return {
    errors,
    turnId,
    emit: (...frames: InboundEvent[]) => frames.forEach((frame) => socket.send(JSON.stringify(frame))),
  };
}

test("moves the previous answer out of view smoothly when sending", async ({ page }) => {
  await page.clock.install({ time: new Date("2026-01-01T00:00:00Z") });
  await page.setViewportSize({ width: 1440, height: 800 });
  await openThread(page, true, false, {
    shortHistory: true,
    beforeSend: async () => {
      // Control animation time rather than requiring a minimum CI runner frame rate.
      await page.clock.pauseAt(new Date("2026-01-01T00:01:00Z"));
      await page.evaluate(() => {
        const samples: number[] = [];
        Object.assign(window, { __promptMotion: { samples } });
        document.querySelector("textarea")!.addEventListener("keydown", (event) => {
          if (event.key !== "Enter") return;
          const started = performance.now();
          const sample = () => {
            const viewport = document.querySelector<HTMLElement>('[data-testid="thread-message-region"]');
            if (viewport) samples.push(viewport.scrollTop);
            if (performance.now() - started < 400) requestAnimationFrame(sample);
          };
          sample();
        }, { once: true });
      });
    },
  });
  await page.clock.runFor(400);
  const positions = await page.evaluate(() => (window as unknown as { __promptMotion: { samples: number[] } }).__promptMotion.samples);
  const distinctPositions = new Set(positions.map((position) => Math.round(position)));
  expect(distinctPositions.size).toBeGreaterThan(3);
  const viewport = page.getByTestId("thread-message-region");
  const prompt = viewport.locator("[data-user-prompt-id]").filter({ hasText: "My anchored question" });
  const geometry = await prompt.evaluate((node) => {
    const viewport = node.closest('[data-testid="thread-message-region"]')!;
    const previous = node.previousElementSibling!;
    return {
      previousBottom: previous.getBoundingClientRect().bottom - viewport.getBoundingClientRect().top,
      promptTop: node.getBoundingClientRect().top - viewport.getBoundingClientRect().top,
    };
  });
  expect(geometry.previousBottom).toBeLessThanOrEqual(0);
  expect(geometry.promptTop).toBeGreaterThanOrEqual(0);
  expect(geometry.promptTop).toBeLessThanOrEqual(48);
});

test("anchors the prompt with reduced motion enabled", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 800 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  const { errors } = await openThread(page, true, false, { shortHistory: true });
  const viewport = page.getByTestId("thread-message-region");
  const prompt = viewport.locator("[data-user-prompt-id]").filter({ hasText: "My anchored question" });
  await expect.poll(() => prompt.evaluate((node) => {
    const region = node.closest('[data-testid="thread-message-region"]')!;
    return node.previousElementSibling!.getBoundingClientRect().bottom - region.getBoundingClientRect().top;
  })).toBeLessThanOrEqual(0);
  expect(errors).toEqual([]);
});

test("re-engages follow at the live tail with reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 800 });
  const { emit, errors, turnId } = await openThread(page, true);
  const viewport = page.getByTestId("thread-message-region");
  const prompt = viewport.locator("[data-user-prompt-id]").filter({ hasText: "My anchored question" });
  const promptOffset = () => prompt.evaluate((node) => {
    const region = node.closest('[data-testid="thread-message-region"]')!;
    return node.getBoundingClientRect().top - region.getBoundingClientRect().top;
  });
  const distanceFromBottom = () => viewport.evaluate((node) => node.scrollHeight - node.clientHeight - node.scrollTop);
  await expect.poll(promptOffset).toBeLessThanOrEqual(48);
  const initialOffset = await promptOffset();
  expect(initialOffset).toBeGreaterThanOrEqual(0);

  const base = { chat_id: chatId, turn_id: turnId, stream_id: "answer" };
  emit({ event: "delta", ...base, turn_seq: 1, text: `${"Growing answer line.\n\n".repeat(60)}END-FIRST` });
  await expect(viewport).toContainText("END-FIRST");
  // The submitted prompt stays anchored while the turn streams.
  expect(await promptOffset()).toBeCloseTo(initialOffset, 0);

  // Reaching the live tail by hand re-engages follow for the running turn.
  await viewport.hover();
  for (let pass = 0; pass < 40 && (await distanceFromBottom()) > 1; pass++) {
    await page.mouse.wheel(0, 1000);
    await page.waitForTimeout(25);
  }
  await expect.poll(distanceFromBottom).toBeLessThan(1);
  emit({ event: "delta", ...base, turn_seq: 2, text: `${"Followed growth.\n\n".repeat(40)}END-FOLLOW` });
  await expect(viewport).toContainText("END-FOLLOW");
  await expect.poll(distanceFromBottom).toBeLessThan(1);
  expect(await promptOffset()).toBeLessThan(0);

  // Reading older history stops follow again.
  await viewport.hover();
  await page.mouse.wheel(0, -600);
  await expect.poll(distanceFromBottom).toBeGreaterThan(300);
  const readingTop = await viewport.evaluate((node) => node.scrollTop);
  emit({ event: "delta", ...base, turn_seq: 3, text: "\n\nMore growth." });
  await expect(viewport).toContainText("More growth.");
  expect(await viewport.evaluate((node) => node.scrollTop)).toBeCloseTo(readingTop, 0);

  // The explicit bottom button returns to the tail and follows again.
  await page.getByRole("button", { name: "Scroll to bottom" }).click();
  await expect.poll(distanceFromBottom).toBeLessThan(1);
  emit({ event: "delta", ...base, turn_seq: 4, text: `${"Final growth.\n\n".repeat(30)}END-LAST` });
  await expect(viewport).toContainText("END-LAST");
  await expect.poll(distanceFromBottom).toBeLessThan(1);
  expect(errors).toEqual([]);
});

for (const width of [390, 1440]) for (const withHistory of [false, true]) {
  test(`anchors a submitted prompt without following streamed text at ${width}px, history=${withHistory}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    const { emit, errors, turnId } = await openThread(page, withHistory);
    const viewport = page.getByTestId("thread-message-region");
    const prompt = viewport.locator("[data-user-prompt-id]").filter({ hasText: "My anchored question" });
    await expect(prompt).toHaveCount(1);
    const promptOffset = () => prompt.evaluate((node) => {
      const viewport = node.closest('[data-testid="thread-message-region"]')!;
      return node.getBoundingClientRect().top - viewport.getBoundingClientRect().top;
    });
    if (withHistory) {
      await expect.poll(() => prompt.evaluate((node) => {
        const viewport = node.closest('[data-testid="thread-message-region"]')!;
        return node.previousElementSibling!.getBoundingClientRect().bottom - viewport.getBoundingClientRect().top;
      })).toBeLessThanOrEqual(0);
    } else {
      await expect.poll(promptOffset).toBeCloseTo(48, 0);
    }
    if (width === 1440 && withHistory) {
      const beforeResize = await viewport.evaluate((node) => node.scrollTop);
      await page.setViewportSize({ width: 390, height: 800 });
      await expect.poll(promptOffset).toBeLessThanOrEqual(48);
      expect(await promptOffset()).toBeGreaterThanOrEqual(0);
      expect(await viewport.evaluate((node) => node.scrollTop)).toBeGreaterThan(beforeResize);
    }
    const initialOffset = await promptOffset();
    const initialTop = await viewport.evaluate((node) => node.scrollTop);
    const distanceFromBottom = () => viewport.evaluate((node) => node.scrollHeight - node.clientHeight - node.scrollTop);
    const base = { chat_id: chatId, turn_id: turnId, stream_id: "answer" };
    emit({ event: "delta", ...base, turn_seq: 1, text: "The short opening." });
    await expect(page.getByText("The short opening.", { exact: true })).toBeVisible();
    await expect.poll(promptOffset).toBeCloseTo(initialOffset, 0);
    expect(await viewport.evaluate((node) => node.scrollTop)).toBeCloseTo(initialTop, 0);

    emit({ event: "delta", ...base, turn_seq: 2, text: `\n\n${"Long streamed answer line.\n\n".repeat(90)}END-FIRST-BATCH` });
    await expect(viewport).toContainText("END-FIRST-BATCH");
    expect(await viewport.evaluate((node) => node.scrollTop)).toBeCloseTo(initialTop, 0);
    await expect.poll(promptOffset).toBeCloseTo(initialOffset, 0);

    // Scrolling away mid-stream keeps the reading position stable.
    await viewport.hover();
    await page.mouse.wheel(0, 360);
    const stableTop = () => viewport.evaluate(async (node) => {
      let previous = node.scrollTop;
      let stable = 0;
      for (let i = 0; i < 600 && stable < 8; i++) {
        await new Promise((resolve) => requestAnimationFrame(resolve));
        if (node.scrollTop === previous) stable += 1;
        else stable = 0;
        previous = node.scrollTop;
      }
      return previous;
    });
    await expect.poll(() => viewport.evaluate((node) => node.scrollTop)).toBeGreaterThan(initialTop + 100);
    const readingTop = await stableTop();
    emit({ event: "delta", ...base, turn_seq: 3, text: `\n\n${"More streamed text.\n\n".repeat(30)}END-SECOND-BATCH` });
    await expect(viewport).toContainText("END-SECOND-BATCH");
    expect(await viewport.evaluate((node) => node.scrollTop)).toBeCloseTo(readingTop, 0);

    // Reaching the live tail by hand re-engages follow for the running turn.
    for (let pass = 0; pass < 40 && (await distanceFromBottom()) > 1; pass++) {
      await page.mouse.wheel(0, 1000);
      await page.waitForTimeout(25);
    }
    await expect.poll(distanceFromBottom).toBeLessThan(1);
    emit({ event: "delta", ...base, turn_seq: 4, text: `\n\n${"Followed after reaching the tail.\n\n".repeat(20)}END-THIRD-BATCH` });
    await expect(viewport).toContainText("END-THIRD-BATCH");
    await expect.poll(distanceFromBottom).toBeLessThanOrEqual(0.5);
    expect(await viewport.evaluate((node) => node.scrollTop)).toBeGreaterThan(readingTop + 100);

    // Scrolling back into history stops follow again.
    await page.mouse.wheel(0, -600);
    const historyTop = await stableTop();
    emit({ event: "delta", ...base, turn_seq: 5, text: `\n\n${"Kept in view after scrolling away.\n\n".repeat(20)}END-FOURTH-BATCH` });
    await expect(viewport).toContainText("END-FOURTH-BATCH");
    expect(await viewport.evaluate((node) => node.scrollTop)).toBeCloseTo(historyTop, 0);

    // The explicit bottom button returns to the tail and re-engages follow.
    await page.getByRole("button", { name: "Scroll to bottom" }).click();
    await expect.poll(distanceFromBottom).toBeLessThanOrEqual(0.5);
    emit({ event: "delta", ...base, turn_seq: 6, text: `\n\n${"New text after navigation.\n\n".repeat(20)}END-FIFTH-BATCH` });
    await expect(viewport).toContainText("END-FIFTH-BATCH");
    await expect.poll(distanceFromBottom).toBeLessThanOrEqual(0.5);
    emit({ event: "stream_end", ...base, turn_seq: 7 });
    emit({ event: "turn_end", chat_id: chatId, turn_id: turnId, turn_seq: 8 });
    await expect(page.getByRole("button", { name: "Stop response" })).toHaveCount(0);
    await expect.poll(distanceFromBottom).toBeLessThanOrEqual(0.5);
    expect(errors).toEqual([]);
  });
}

test("keeps the prompt anchored through immediate completion and lazy history layout", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 800 });
  let releaseMarkdown!: () => void;
  const markdownReady = new Promise<void>((resolve) => { releaseMarkdown = resolve; });
  await page.route("**/src/components/MarkdownTextRenderer.tsx*", async (route) => {
    await markdownReady;
    await route.continue();
  });
  const { errors } = await openThread(page, true, true, {
    shortHistory: true,
    beforeSend: () => page.locator("[data-assistant-message]").click(),
  });
  const viewport = page.getByTestId("thread-message-region");
  const prompt = viewport.locator("[data-user-prompt-id]").filter({ hasText: "My anchored question" });
  await expect(viewport).toContainText("Immediate answer");
  await expect(page.getByRole("button", { name: "Stop response" })).toHaveCount(0);
  const precedingBottom = () => prompt.evaluate((node) => {
    const viewport = node.closest('[data-testid="thread-message-region"]')!;
    return node.previousElementSibling!.getBoundingClientRect().bottom - viewport.getBoundingClientRect().top;
  });
  await expect.poll(precedingBottom).toBeLessThanOrEqual(0);
  releaseMarkdown();
  await expect(viewport.getByRole("heading", { name: "Earlier answer 0" })).toBeAttached();
  await expect.poll(precedingBottom).toBeLessThanOrEqual(0);
  await expect.poll(precedingBottom).toBeGreaterThan(-1);
  await expect.poll(() => prompt.evaluate((node) => {
    const offset = node.getBoundingClientRect().top
      - node.closest('[data-testid="thread-message-region"]')!.getBoundingClientRect().top;
    return offset >= 0 && offset <= 48;
  })).toBe(true);
  expect(errors).toEqual([]);
});

test("does not apply a previous conversation's anchor after switching away and back", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 800 });
  const { emit, errors, turnId } = await openThread(page, true);
  const viewport = page.getByTestId("thread-message-region");
  const base = { chat_id: chatId, turn_id: turnId, stream_id: "answer" };
  emit({ event: "delta", ...base, turn_seq: 1, text: `${"Long output before switching.\n\n".repeat(80)}BEFORE-SWITCH` });
  await expect(viewport).toContainText("BEFORE-SWITCH");
  await page.getByRole("button", { name: "Switch topic" }).click();
  await expect(page.getByText("My anchored question", { exact: true })).toHaveCount(0);
  emit({ event: "delta", ...base, turn_seq: 2, text: "\n\nOutput while away." });
  await expect(page.getByText("Output while away.", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Switch topic" }).click();
  await expect(page.getByText("My anchored question", { exact: true })).toHaveCount(1);
  await expect(viewport).toContainText("Output while away.");
  await expect.poll(() => viewport.evaluate((node) => node.clientHeight)).toBeLessThan(800);
  await expect.poll(() => page.locator(".thread-layout").evaluate((node) =>
    node.getAnimations().some((animation) => animation.playState === "running"),
  )).toBe(false);
  await expect.poll(() => viewport.evaluate((node) => node.scrollHeight - node.clientHeight - node.scrollTop))
    .toBeLessThanOrEqual(1);
  const reopenedTop = await viewport.evaluate((node) => node.scrollTop);
  emit({ event: "delta", ...base, turn_seq: 3, text: `\n\n${"Output after returning.\n\n".repeat(30)}AFTER-RETURN` });
  await expect(viewport).toContainText("AFTER-RETURN");
  expect(await viewport.evaluate((node) => node.scrollTop)).toBeCloseTo(reopenedTop, 0);
  expect(errors).toEqual([]);
});
