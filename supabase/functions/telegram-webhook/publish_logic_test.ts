/**
 * Unit tests for the publish decision logic (publish_logic.ts).
 *
 * Runs with `deno test supabase/functions/telegram-webhook/publish_logic_test.ts`
 * No external dependencies.
 *
 * These pin the idempotency invariants of the publish/retry flow:
 *   - a row is claimable only from PENDING / PARTIAL_FAILURE (concurrent and
 *     repeated approve taps that are already claimed / resolved are no-ops,
 *     so a draft is never published twice);
 *   - a platform is published only when its own status is PENDING / FAILED;
 *     a PUBLISHING (unconfirmed, in-flight) platform is NEVER auto-republished
 *     — stale PUBLISHING recovery or a crash between the external publish and
 *     its durable record cannot create a duplicate post;
 *   - the final top-level status is derived purely from the durable
 *     per-platform statuses, so partial retries and recovery converge.
 */

import {
  PLATFORMS,
  applyAttemptOutcomes,
  deriveFinalStatus,
  isAmbiguousPlatformError,
  isClaimable,
  isPublishable,
  normalizeStatus,
  selectPublishTargets,
  type AttemptResult,
  type PlatformState,
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

const denied = (status: string | null): boolean => status === "PUBLISHING" ||
  status === "PUBLISHED" || status === "REJECTED";

Deno.test("isClaimable: rows in PENDING or PARTIAL_FAILURE may be claimed", () => {
  assertEq(isClaimable("PENDING"), true);
  assertEq(isClaimable("pending"), true, "case-insensitive");
  assertEq(isClaimable("PARTIAL_FAILURE"), true);
  assertEq(isClaimable(null), false, "missing status is not claimable");
});

Deno.test("isClaimable: concurrent/repeated taps on claimed or resolved rows are no-ops", () => {
  for (const status of ["PUBLISHING", "PUBLISHED", "REJECTED"]) {
    assertEq(isClaimable(status), false, `${status} must not be claimable`);
  }
});

Deno.test("isPublishable: only PENDING / FAILED platforms may be (re)published", () => {
  assertEq(isPublishable("PENDING"), true);
  assertEq(isPublishable("FAILED"), true);
  assertEq(isPublishable(null), true, "missing per-platform status = never attempted");
  assertEq(isPublishable("PUBLISHED"), false, "already published");
  assertEq(isPublishable("PUBLISHING"), false, "in-flight/unconfirmed – never auto-republished");
});

Deno.test("normalizeStatus: unknown/missing maps to PENDING", () => {
  assertEq(normalizeStatus("PUBLISHED"), "PUBLISHED");
  assertEq(normalizeStatus(null), "PENDING");
  assertEq(normalizeStatus(undefined), "PENDING");
  assertEq(normalizeStatus("weird-value"), "PENDING");
  assertEq(normalizeStatus("FAILED"), "FAILED");
});

Deno.test("selectPublishTargets: fresh PENDING post publishes both platforms", () => {
  assertDeepEq(
    selectPublishTargets({ linkedin: "PENDING", x: "PENDING" }),
    PLATFORMS,
  );
});

Deno.test("selectPublishTargets: stale PUBLISHING recovery never republishes unconfirmed platform", () => {
  // Top-level recovered PENDING, but LinkedIn is still the unconfirmed attempt
  // of a previous (crashed) claim. Only X may be attempted.
  assertDeepEq(
    selectPublishTargets({ linkedin: "PUBLISHING", x: "PENDING" }),
    ["X (Twitter)"],
  );
  assertDeepEq(
    selectPublishTargets({ linkedin: "PUBLISHING", x: "PUBLISHING" }),
    [],
  );
});

Deno.test("selectPublishTargets: partial retry only touches non-PUBLISHED platforms", () => {
  // LinkedIn went out on an earlier round; X failed definitively → retry X only.
  assertDeepEq(
    selectPublishTargets({ linkedin: "PUBLISHED", x: "FAILED" }),
    ["X (Twitter)"],
  );
  // Both done → nothing left to publish.
  assertDeepEq(
    selectPublishTargets({ linkedin: "PUBLISHED", x: "PUBLISHED" }),
    [],
  );
  // FAILED platforms are retryable alongside never-attempted ones.
  assertDeepEq(
    selectPublishTargets({ linkedin: "FAILED", x: "PENDING" }),
    PLATFORMS,
  );
});

Deno.test("applyAttemptOutcomes: full success resolves to PUBLISHED", () => {
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PENDING", x: "PENDING" },
    [
      { platform: "LinkedIn", success: true, post_id: "li-1" },
      { platform: "X (Twitter)", success: true, post_id: "x-1" },
    ],
  );
  assertDeepEq(finalStates, { linkedin: "PUBLISHED", x: "PUBLISHED" });
  assertEq(finalStatus, "PUBLISHED");
});

Deno.test("applyAttemptOutcomes: partial success → PARTIAL_FAILURE, retryable", () => {
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PENDING", x: "PENDING" },
    [
      { platform: "LinkedIn", success: true, post_id: "li-1" },
      { platform: "X (Twitter)", success: false, error: "API 400" },
    ],
  );
  assertDeepEq(finalStates, { linkedin: "PUBLISHED", x: "FAILED" });
  assertEq(finalStatus, "PARTIAL_FAILURE");
  // Next round retries ONLY the failed platform.
  assertDeepEq(selectPublishTargets(finalStates), ["X (Twitter)"]);
});

Deno.test("applyAttemptOutcomes: total failure reverts to PENDING (re-claimable)", () => {
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PENDING", x: "PENDING" },
    [
      { platform: "LinkedIn", success: false, error: "API 403" },
      { platform: "X (Twitter)", success: false, error: "API 401" },
    ],
  );
  assertDeepEq(finalStates, { linkedin: "FAILED", x: "FAILED" });
  assertEq(finalStatus, "PENDING");
  // Total failure must be back to claimable for the next tap...
  assertEq(isClaimable(finalStatus), true);
  // ...and every platform is retryable again.
  assertDeepEq(selectPublishTargets(finalStates), PLATFORMS);
});

Deno.test("applyAttemptOutcomes: unconfirmed attempt keeps PUBLISHING → PENDING, never auto-retried", () => {
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PENDING", x: "PENDING" },
    [
      { platform: "LinkedIn", success: false, unconfirmed: true, error: "timeout" },
      { platform: "X (Twitter)", success: true, post_id: "x-2" },
    ],
  );
  assertDeepEq(finalStates, { linkedin: "PUBLISHING", x: "PUBLISHED" });
  assertEq(finalStatus, "PENDING", "unconfirmed → safe claimable state");
  // The unconfirmed platform must NOT be republished by a recovery retry.
  assertDeepEq(selectPublishTargets(finalStates), [], "no auto-retry of unconfirmed platform");
});

Deno.test("applyAttemptOutcomes: post published but record failed → unconfirmed, no duplicate", () => {
  // Crash between step 2 (external success, li-9 live) and step 3 (record).
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PENDING", x: "PENDING" },
    [{ platform: "LinkedIn", success: false, unconfirmed: true, post_id: "li-9", error: "record failed" }],
  );
  assertEq(finalStates.linkedin, "PUBLISHING");
  assertEq(finalStates.x, "PENDING");
  assertEq(finalStatus, "PENDING");
  // The unconfirmed LinkedIn may be live → NOT retried; X was never attempted
  // → the next round attempts only X.
  assertDeepEq(selectPublishTargets(finalStates), ["X (Twitter)"]);
});

Deno.test("applyAttemptOutcomes: skipped platforms keep their durable pre-round status", () => {
  // Partial retry where LinkedIn was already PUBLISHED (skipped, untouched)
  // and X is republished on this round.
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PUBLISHED", x: "FAILED" },
    [{ platform: "X (Twitter)", success: true, post_id: "x-3" }],
  );
  assertDeepEq(finalStates, { linkedin: "PUBLISHED", x: "PUBLISHED" });
  assertEq(finalStatus, "PUBLISHED");
});

Deno.test("applyAttemptOutcomes: recovery claim of all-done platforms heals to PUBLISHED", () => {
  // Both platforms already have a durable PUBLISHED status but the top-level
  // status was left as PARTIAL_FAILURE/PENDING (anomaly) — no attempts happen,
  // the derivation heals the top-level status.
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PUBLISHED", x: "PUBLISHED" },
    [],
  );
  assertEq(finalStatus, "PUBLISHED");
});

Deno.test("applyAttemptOutcomes: recovery claim with an unconfirmed platform stays PENDING", () => {
  // A stale claim left LinkedIn PUBLISHING (unconfirmed) and X was already
  // PUBLISHED: the report must stay claimable (PENDING) and NOT republish.
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: "PUBLISHING", x: "PUBLISHED" },
    [],
  );
  assertEq(finalStatus, "PENDING");
  assertEq(finalStates.linkedin, "PUBLISHING");
  assertEq(finalStates.x, "PUBLISHED");
});

Deno.test("deriveFinalStatus: statuses of both platforms and unconfirmed rule", () => {
  assertEq(deriveFinalStatus({ linkedin: "PUBLISHED", x: "PUBLISHED" }), "PUBLISHED");
  assertEq(deriveFinalStatus({ linkedin: "PUBLISHED", x: "FAILED" }), "PARTIAL_FAILURE");
  assertEq(deriveFinalStatus({ linkedin: "FAILED", x: "FAILED" }), "PENDING");
  assertEq(deriveFinalStatus({ linkedin: "PENDING", x: "PUBLISHED" }), "PARTIAL_FAILURE");
  // Any PUBLISHING (unconfirmed) dominates → PENDING (claims keep working,
  // but the unconfirmed platform is never auto-republished separately).
  assertEq(deriveFinalStatus({ linkedin: "PUBLISHING", x: "PUBLISHED" }), "PENDING");
  assertEq(deriveFinalStatus({ linkedin: "PUBLISHED", x: "PUBLISHING" }), "PENDING");
  assertEq(deriveFinalStatus({ linkedin: "PUBLISHING", x: "FAILED" }), "PENDING");
});

Deno.test("isAmbiguousPlatformError: 5xx / network / timeout are ambiguous", () => {
  assertEq(isAmbiguousPlatformError(new Error("API 500 Internal Server Error")), true);
  assertEq(isAmbiguousPlatformError(new Error("API 502 Bad Gateway")), true);
  assertEq(isAmbiguousPlatformError(new Error("fetch failed")), true);
  assertEq(isAmbiguousPlatformError(new Error("AbortError: The operation was aborted.")), true);
  assertEq(isAmbiguousPlatformError(new Error("Request timed out")), true);
  assertEq(isAmbiguousPlatformError(new TypeError("Failed to fetch")), true);
});

Deno.test("isAmbiguousPlatformError: 4xx is a definitive rejection (safe to retry)", () => {
  assertEq(isAmbiguousPlatformError(new Error("API 400 Bad Request")), false);
  assertEq(isAmbiguousPlatformError(new Error("API 401 Unauthorized")), false);
  assertEq(isAmbiguousPlatformError(new Error("API 403 Forbidden")), false);
  assertEq(isAmbiguousPlatformError(new Error("API 404 Not Found")), false);
  assertEq(isAmbiguousPlatformError(new Error("API 429 Too Many Requests")), false);
});

Deno.test("isAmbiguousPlatformError: unrecognized errors are treated as ambiguous", () => {
  // Unknown error shapes default to ambiguous — safer than assuming a failed
  // publish on an external side effect.
  assertEq(isAmbiguousPlatformError(new Error("boom")), true);
});

Deno.test("concurrent approval: only one claim wins; the loser's report is a no-op", () => {
  // Two approve taps arrive for the same post. The first claims (PENDING →
  // PUBLISHING); the second sees PUBLISHING and must NOT attempt anything.
  const firstClaim = isClaimable("PENDING");
  assertEq(firstClaim, true);
  // After the first claim the durable top-level status is PUBLISHING.
  assertEq(isClaimable("PUBLISHING"), false);
  assertEq(denied("PUBLISHING"), true);

  // Loser re-reads the claim table and finds no publishable targets.
  const attempts = selectPublishTargets({ linkedin: "PUBLISHING", x: "PUBLISHING" });
  assertDeepEq(attempts, []);
});

Deno.test("repeated approval: an already-resolved row is never republished", () => {
  // A post that reported PUBLISHED/PENDING-after-unconfirmed is tapped again.
  assertEq(isClaimable("PUBLISHED"), false);
  assertEq(denied("PUBLISHED"), true);
  // Even if somehow claimed, per-platform PUBLISHED means nothing is attempted.
  assertDeepEq(selectPublishTargets({ linkedin: "PUBLISHED", x: "PUBLISHED" }), []);
  const state: PlatformState = { linkedin: "PUBLISHED", x: "PUBLISHED" };
  const attempts: AttemptResult[] = [];
  const { finalStatus } = applyAttemptOutcomes(state, attempts);
  assertEq(finalStatus, "PUBLISHED");
});