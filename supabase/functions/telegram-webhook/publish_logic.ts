/**
 * PersonaPulse – publish decision logic (Telegram webhook)
 * =========================================================
 * Pure, unit-testable decision helpers for the approval/publish flow. Keeping
 * them free of I/O lets the idempotency invariants be verified directly:
 *
 *   - A row is claimable only from PENDING / PARTIAL_FAILURE. A stale
 *     PUBLISHING recovery returns to PENDING, so a later approve tap reclaims
 *     it — but the SECOND claim never re-publishes a platform that a former
 *     (crashed) claim already published or left unconfirmed.
 *   - A platform is published ONLY when its own per-platform status is
 *     PENDING (never attempted) or FAILED (definitively rejected). A
 *     platform marked PUBLISHING is an in-flight / unconfirmed attempt and is
 *     NEVER auto-republished — that is what prevents duplicate posts when a
 *     stale PUBLISHING record is recovered or when the process dies between
 *     the external publish and the durable record.
 *   - The final top-level status is derived purely from the durable
 *     per-platform statuses, so recovery, partial retries and concurrent
 *     taps all converge on the same outcome.
 */

export type PlatformName = "LinkedIn" | "X (Twitter)";
export type PlatformStatus = "PENDING" | "PUBLISHING" | "PUBLISHED" | "FAILED";
export type FinalStatus = "PUBLISHED" | "PARTIAL_FAILURE" | "PENDING";

export const PLATFORMS: PlatformName[] = ["LinkedIn", "X (Twitter)"];

export interface PlatformState {
  linkedin?: string | null;
  x?: string | null;
}

/** Normalize a stored (possibly missing/empty) per-platform status value. */
export function normalizeStatus(value: string | null | undefined): PlatformStatus {
  const s = (value ?? "").trim().toUpperCase();
  return (["PENDING", "PUBLISHING", "PUBLISHED", "FAILED"] as string[]).includes(s)
    ? (s as PlatformStatus)
    : "PENDING";
}

/**
 * A post row may be claimed for publishing only while its top-level status is
 * exactly PENDING or PARTIAL_FAILURE. The atomic conditional claim (status IN
 * [...] → PUBLISHING) rejects every other state, so a second/concurrent approve
 * tap on an already-claimed (PUBLISHING), resolved (PUBLISHED) or rejected
 * (REJECTED) row is a no-op — a draft is never published twice. A missing or
 * unknown status is deliberately NOT claimable.
 */
export function isClaimable(status: string | null | undefined): boolean {
  const s = (status ?? "").trim().toUpperCase();
  return s === "PENDING" || s === "PARTIAL_FAILURE";
}

/**
 * A platform may be (re)attempted only when its per-platform status is
 * PENDING (never touched) or FAILED (a definitive API rejection, safe to
 * retry). PUBLISHED is done; PUBLISHING is an in-flight/unconfirmed attempt
 * that must never be auto-republished. A missing status means the platform
 * was never attempted → publishable; any unknown value (safety) is not.
 */
export function isPublishable(status: string | null | undefined): boolean {
  if (status === null || status === undefined || String(status).trim() === "") {
    return true;
  }
  const s = String(status).trim().toUpperCase();
  return s === "PENDING" || s === "FAILED";
}

/**
 * The ordered list of platforms to publish on a given approve tap. Skipped
 * platforms keep their durable per-platform status untouched.
 */
export function selectPublishTargets(state: PlatformState): PlatformName[] {
  const targets: PlatformName[] = [];
  if (isPublishable(state.linkedin)) targets.push("LinkedIn");
  if (isPublishable(state.x)) targets.push("X (Twitter)");
  return targets;
}

export interface AttemptResult {
  platform: PlatformName;
  success: boolean;
  /** True when the outcome is unresolved: the platform may or may not have
   * been posted (crash/timeout between the external call and its record). */
  unconfirmed?: boolean;
  post_id?: string | null;
  /** Human-readable reason for the outcome (used in the Telegram report). */
  error?: string;
}

/**
 * Resolves one approve round into the new durable per-platform statuses and
 * the final top-level status. Platforms that were attempted take their
 * attempt outcome (unconfirmed → PUBLISHING, success → PUBLISHED, failure →
 * FAILED); platforms not attempted keep their pre-round status.
 */
export function applyAttemptOutcomes(
  claimState: PlatformState,
  attempted: AttemptResult[],
): { finalStates: PlatformState; finalStatus: FinalStatus } {
  const colFor = (p: PlatformName): "linkedin" | "x" =>
    p === "LinkedIn" ? "linkedin" : "x";

  const finalStates: PlatformState = {
    linkedin: normalizeStatus(claimState.linkedin),
    x: normalizeStatus(claimState.x),
  };
  for (const r of attempted) {
    finalStates[colFor(r.platform)] = r.unconfirmed
      ? "PUBLISHING"
      : r.success
        ? "PUBLISHED"
        : "FAILED";
  }
  return { finalStates, finalStatus: deriveFinalStatus(finalStates) };
}

/**
 * Derive the top-level post status from the durable per-platform statuses:
 *   - both PUBLISHED                    → PUBLISHED
 *   - any PUBLISHING (unconfirmed)      → PENDING (safe: stays claimable, but
 *     the unconfirmed platform is never auto-republished)
 *   - at least one PUBLISHED            → PARTIAL_FAILURE (retry the rest)
 *   - otherwise (all FAILED/PENDING)    → PENDING (total failure → re-claim)
 */
export function deriveFinalStatus(state: PlatformState): FinalStatus {
  const statuses = [normalizeStatus(state.linkedin), normalizeStatus(state.x)];
  if (statuses.every((s) => s === "PUBLISHED")) return "PUBLISHED";
  if (statuses.some((s) => s === "PUBLISHING")) return "PENDING";
  if (statuses.some((s) => s === "PUBLISHED")) return "PARTIAL_FAILURE";
  return "PENDING";
}

/**
 * Classifies a platform publisher error as AMBIGUOUS or definitive.
 *
 * A definitive failure (HTTP 4xx – bad request/forbidden/unprocessable) means
 * the platform rejected the post outright: safe AND correct to record as
 * FAILED and allow a retry. Everything else (HTTP 5xx, network failures,
 * timeouts/aborts) is ambiguous: the platform MAY have accepted the request
 * even though no response came back, so recording it as a plain FAILED and
 * auto-republishing could create a duplicate post.
 */
export function isAmbiguousPlatformError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  const statusMatch = /API ([45])\d\d/.exec(message);
  if (statusMatch) return statusMatch[1] === "5";
  // No explicit HTTP status: network failures, timeouts, aborts — and any
  // unrecognized error — default to ambiguous. Safer than assuming a failed
  // publish on an external side effect (the platform MAY have accepted it).
  return true;
}