/**
 * Focused tests for the publish persistence safety in index.ts
 * =============================================================
 * Runs with `deno test supabase/functions/telegram-webhook/publish_persistence_test.ts`
 *
 * These pin the persistence invariants added for the remaining Phase-1 issues:
 *   - a Supabase '.update()' response error is NEVER silently swallowed:
 *     recordPerPlatformStatus / persistFinalStatus throw, so a write failure
 *     cannot masquerade as a clean publish;
 *   - a successful external publish whose per-platform DB record fails is
 *     reported as UNCONFIRMED (success === false) — never as fully published;
 *   - the PUBLISHING write-ahead marker precedes every external call; if it
 *     cannot be written, the external platform is never invoked;
 *   - ambiguous external failures stay PUBLISHING/unconfirmed (no FAILED
 *     write, never auto-retried);
 *   - partial platform failure derives PARTIAL_FAILURE and retries only the
 *     failed platform; repeated/concurrent approval of a claimed row publishes
 *     nothing.
 *
 * A mock Supabase client returns queued { data, error } outcomes for each
 * .from("posts").update(...).eq(...) call, exactly like the real client
 * (which resolves rather than throwing on response errors).
 */

import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2";
import {
  persistFinalStatus,
  publishAndRecord,
  recordPerPlatformStatus,
} from "./index.ts";
import {
  applyAttemptOutcomes,
  isClaimable,
  selectPublishTargets,
} from "./publish_logic.ts";

function assertEq(actual: unknown, expected: unknown, msg?: string): void {
  if (actual !== expected) {
    throw new Error(
      `${msg ?? "assertEq"}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
    );
  }
}

function assertDeepEq(actual: unknown, expected: unknown, msg?: string): void {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) {
    throw new Error(`${msg ?? "assertDeepEq"}:\n  expected ${e}\n  got      ${a}`);
  }
}

function assertThrows(
  fn: () => Promise<unknown>,
  fragment: string,
  msg?: string,
): Promise<void> {
  return fn()
    .then(() => {
      throw new Error(`${msg ?? "assertThrows"}: expected a rejection containing '${fragment}'`);
    })
    .catch((err) => {
      if (err instanceof Error && err.message.includes(fragment)) return;
      if (err instanceof Error) {
        throw new Error(
          `${msg ?? "assertThrows"}: rejection '${err.message}' did not contain '${fragment}'`,
        );
      }
      throw new Error(`${msg ?? "assertThrows"}: rejection was not an Error: ${String(err)}`);
    });
}

interface MockOutcome {
  data?: unknown;
  error?: { message: string } | null;
}

/**
 * Chain mock: from().update(patch).eq() resolves to the next queued outcome
 * (repeating the last once the queue is exhausted). Records every patch.
 */
function mockSupabase(
  outcomes: MockOutcome[],
  onUpdate?: (patch: Record<string, unknown>) => void,
): SupabaseClient {
  let call = 0;
  return {
    from: () => ({
      update: (patch: Record<string, unknown>) => {
        onUpdate?.(patch);
        const outcome = outcomes.length === 0
          ? { data: null, error: { message: "no mock outcome queued" } }
          : outcomes[Math.min(call, outcomes.length - 1)];
        call += 1;
        return {
          eq: () => Promise.resolve({ data: outcome.data ?? null, error: outcome.error ?? null }),
        };
      },
    }),
  } as unknown as SupabaseClient;
}

const ok = (): MockOutcome => ({ data: {} });
const failing = (message: string): MockOutcome => ({ error: { message } });

// ---------------------------------------------------------------------------
// Per-platform DB update failure after successful external publishing
// ---------------------------------------------------------------------------

Deno.test("publishAndRecord: live post whose DB record fails is UNCONFIRMED, never success", async () => {
  const patches: Record<string, unknown>[] = [];
  const db = mockSupabase([ok(), failing("connection reset")], (p) => patches.push(p));
  const result = await publishAndRecord(db, "p1", "LinkedIn", async () => "li-77");

  assertEq(result.success, false, "must not report success when the record failed");
  assertEq(result.unconfirmed, true, "stays unconfirmed (the post may be live)");
  assertEq(result.post_id, "li-77");
  // Write-ahead marker was recorded before the call; the PUBLISHED record was
  // attempted and FAILED — the durable state stays PUBLISHING (unconfirmed).
  assertEq(patches.length, 2);
  assertEq(patches[0].linkedin_status, "PUBLISHING");
  assertEq(patches[1].linkedin_status, "PUBLISHED");

  // A recovery/retry must never republish the unconfirmed platform.
  const { finalStatus } = applyAttemptOutcomes({ linkedin: "PENDING" }, [result]);
  assertEq(finalStatus, "PENDING");
});

// ---------------------------------------------------------------------------
// Final posts.status DB update failure
// ---------------------------------------------------------------------------

Deno.test("persistFinalStatus: response error throws — DB inconsistency is surfaced", async () => {
  const db = mockSupabase([failing("relation posts does not exist")]);
  await assertThrows(
    () => persistFinalStatus(db, "p1", "PUBLISHED"),
    "could not persist final status 'PUBLISHED' for post p1",
  );
});

Deno.test("persistFinalStatus: successful response resolves without throwing", async () => {
  const db = mockSupabase([ok()]);
  await persistFinalStatus(db, "p1", "PARTIAL_FAILURE"); // must not throw
});

// ---------------------------------------------------------------------------
// Successful DB persistence
// ---------------------------------------------------------------------------

Deno.test("publishAndRecord: successful external publish + durable record → success", async () => {
  const patches: Record<string, unknown>[] = [];
  const db = mockSupabase([ok(), ok()], (p) => patches.push(p));
  let runCalls = 0;

  const result = await publishAndRecord(db, "p1", "LinkedIn", async () => {
    runCalls += 1;
    return "li-42";
  });

  assertEq(result.success, true);
  assertEq(result.post_id, "li-42");
  assertEq(result.unconfirmed, undefined);
  assertEq(runCalls, 1, "external platform called exactly once");
  assertEq(patches[0].linkedin_status, "PUBLISHING", "write-ahead marker came first");
  assertEq(patches[1].linkedin_status, "PUBLISHED");
  assertEq(patches[1].linkedin_post_id, "li-42");
});

Deno.test("recordPerPlatformStatus: error throws; success resolves", async () => {
  await assertThrows(
    () => recordPerPlatformStatus(mockSupabase([failing("db down")]), "p1", "x", "PUBLISHED", "x-1"),
    "could not persist x status 'PUBLISHED' for post p1",
  );
  await recordPerPlatformStatus(mockSupabase([ok()]), "p1", "x", "PUBLISHED", "x-1");
});

Deno.test("publishAndRecord: write-ahead failure aborts BEFORE any external call", async () => {
  const db = mockSupabase([failing("db down")]);
  let runCalls = 0;

  const result = await publishAndRecord(db, "p1", "LinkedIn", async () => {
    runCalls += 1;
    return "li-1";
  });

  assertEq(result.success, false);
  assertEq(result.unconfirmed, undefined, "a clean failure, not unconfirmed");
  assertEq(runCalls, 0, "external platform must not be called without a durable marker");
  assertEq(result.error?.includes("could not record publish intent"), true);
});

// ---------------------------------------------------------------------------
// Partial platform failure
// ---------------------------------------------------------------------------

Deno.test("flow: one published + one definitively failed → PARTIAL_FAILURE, retry only the failed one", async () => {
  const li = await publishAndRecord(mockSupabase([ok(), ok()]), "p1", "LinkedIn", async () => "li-1");
  const x = await publishAndRecord(mockSupabase([ok(), ok()]), "p1", "X (Twitter)", async () => {
    throw new Error("X API 429 Too Many Requests");
  });
  assertEq(x.success, false);
  assertEq(x.unconfirmed, undefined, "a definitive 4xx is a clean, retryable failure");

  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PENDING", x: "PENDING" },
    [li, x],
  );
  assertEq(finalStatus, "PARTIAL_FAILURE");
  assertEq(finalStates.linkedin, "PUBLISHED");
  assertEq(finalStates.x, "FAILED");
  assertDeepEq(selectPublishTargets(finalStates), ["X (Twitter)"], "retries only the failed platform");
});

// ---------------------------------------------------------------------------
// Ambiguous external failure
// ---------------------------------------------------------------------------

Deno.test("flow: ambiguous external failure stays PUBLISHING/unconfirmed, never auto-retried", async () => {
  const patches: Record<string, unknown>[] = [];
  const db = mockSupabase([ok()], (p) => patches.push(p));

  const result = await publishAndRecord(db, "p1", "LinkedIn", async () => {
    throw new Error("API 500 Internal Server Error");
  });

  assertEq(result.success, false);
  assertEq(result.unconfirmed, true, "5xx may have been accepted → unconfirmed");
  assertEq(patches.length, 1, "only the write-ahead marker; no FAILED/PUBLISHED write");
  assertEq(patches[0].linkedin_status, "PUBLISHING", "marker stays PUBLISHING");

  const { finalStatus } = applyAttemptOutcomes({ linkedin: "PENDING", x: "PENDING" }, [result]);
  assertEq(finalStatus, "PENDING", "unconfirmed → remains claimable at top level");
  assertDeepEq(
    selectPublishTargets({ linkedin: "PUBLISHING", x: "PENDING" }),
    ["X (Twitter)"],
    "only the never-attempted platform is retried",
  );
});

// ---------------------------------------------------------------------------
// Repeated / concurrent approval stays idempotent
// ---------------------------------------------------------------------------

Deno.test("flow: second/concurrent approve of a claimed row publishes nothing", async () => {
  // First tap claimed the row: top-level + platforms → PUBLISHING.
  assertEq(isClaimable("PUBLISHING"), false, "a claimed row cannot be claimed again");
  assertDeepEq(selectPublishTargets({ linkedin: "PUBLISHING", x: "PUBLISHING" }), []);
  // Also after a resolve: everything PUBLISHED → nothing to do.
  assertDeepEq(selectPublishTargets({ linkedin: "PUBLISHED", x: "PUBLISHED" }), []);

  // Even if a second tap somehow reached publishAndRecord for an already
  // PUBLISHED platform, its record must not regress the platform to PENDING.
  const patches: Record<string, unknown>[] = [];
  const db = mockSupabase([failing("already published")], (p) => patches.push(p));
  await assertThrows(
    () => recordPerPlatformStatus(db, "p1", "linkedin", "PUBLISHED", "li-1"),
    "could not persist",
  );
  assertEq(patches.length, 1);
});