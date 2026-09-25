/**
 * PersonaPulse – Telegram Webhook Handler
 * =========================================
 * Deno TypeScript Supabase Edge Function.
 * 
 *
 * Receives Telegram callback_query webhook events triggered when a user
 * taps the inline ✅ Approve or ❌ Reject buttons in the approval message.
 *
 * 
 * Workflow
 * --------
 *   approve_<post_id>  →  Atomic claim: conditional UPDATE
 *                         status IN (PENDING, PARTIAL_FAILURE) → PUBLISHING.
 *                         If no row matches (double tap / concurrent retry),
 *                         bail out – a draft is never published twice.
 *                      →  Publish ONLY platforms whose per-platform status
 *                         is not yet PUBLISHED (idempotent retry).
 *                      →  Every fetch() wrapped in AbortSignal.timeout(8000)
 *                      →  Persist per-platform status + post id
 *                      →  Final status: PUBLISHED | PARTIAL_FAILURE |
 *                         PENDING (revert on total failure – approve retries)
 *                      →  Edit Telegram message with execution report
 *                      →  Keyboards stay attached unless fully PUBLISHED
 *                         so the missing platform(s) can be retried.
 *
 *   reject_<post_id>   →  Atomic conditional UPDATE (only while PENDING)
 *                      →  Edit Telegram message: "❌ Post Rejected"
 *
 * Environment Variables (set in Supabase Dashboard → Edge Functions → Secrets)
 * -----------------------------------------------------------------------------
 *   SUPABASE_URL                – your project URL
 *   SUPABASE_SERVICE_ROLE_KEY   – service role key (bypasses RLS)
 *   TELEGRAM_BOT_TOKEN          – bot token from @BotFather
 *   TELEGRAM_CHAT_ID            – your personal Telegram chat ID (Layer 2)
 *   TELEGRAM_WEBHOOK_SECRET     – optional custom secret token; if omitted,
 *                                 derived via SHA-256 from TELEGRAM_BOT_TOKEN (Layer 1)
 *   LINKEDIN_ACCESS_TOKEN       – OAuth 2.0 access token
 *   LINKEDIN_AUTHOR_URN         – urn:li:person:XXXXXXXX
 *   X_API_KEY                   – Twitter/X API key
 *   X_API_SECRET                – Twitter/X API secret
 *   X_ACCESS_TOKEN              – Twitter/X access token
 *   X_ACCESS_SECRET             – Twitter/X access token secret
 */

import { createClient, SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2";
import {
  buildAuthorizationHeader,
  buildOAuthSignature,
  generateNonce,
} from "./oauth.ts";
import {
  PLATFORMS,
  applyAttemptOutcomes,
  isAmbiguousPlatformError,
  normalizeStatus,
  selectPublishTargets,
  type AttemptResult,
  type FinalStatus,
  type PlatformName,
  type PlatformState,
} from "./publish_logic.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TelegramChat {
  id: number | string;
  type?: string;
  title?: string;
  username?: string;
}

interface TelegramMessage {
  message_id: number;
  chat: TelegramChat;
  text?: string;
  caption?: string;
}

interface TelegramCallbackQuery {
  id: string;
  from: { id: number; username?: string };
  message?: TelegramMessage;
  data?: string;
}

interface TelegramUpdate {
  update_id: number;
  message?: TelegramMessage;
  edited_message?: TelegramMessage;
  channel_post?: TelegramMessage;
  edited_channel_post?: TelegramMessage;
  callback_query?: TelegramCallbackQuery;
}

interface PostRow {
  id: string;
  content: string;
  platform: string;
  status: string;
  linkedin_status?: string;
  x_status?: string;
  linkedin_post_id?: string | null;
  x_post_id?: string | null;
}

interface ReplyKeyboard {
  inline_keyboard: Array<Array<{ text: string; callback_data: string }>>;
}

// ---------------------------------------------------------------------------
// Constants & Security Helpers
// ---------------------------------------------------------------------------

const FETCH_TIMEOUT_MS = 8_000;  // AbortSignal.timeout cap per spec

/**
 * Reads environment secrets lazily, at the point of use. Keeps module load
 * side-effect free (importable for tests without env permissions) with no
 * runtime behavior change.
 */
function readEnv() {
  return {
    supabaseUrl: Deno.env.get("SUPABASE_URL")!,
    supabaseKey: Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
    telegramBotToken: Deno.env.get("TELEGRAM_BOT_TOKEN")!,
    telegramChatId: Deno.env.get("TELEGRAM_CHAT_ID"),
    telegramWebhookSecret: Deno.env.get("TELEGRAM_WEBHOOK_SECRET"),
    linkedinAccessToken: Deno.env.get("LINKEDIN_ACCESS_TOKEN")!,
    linkedinAuthorUrn: Deno.env.get("LINKEDIN_AUTHOR_URN")!,
    xApiKey: Deno.env.get("X_API_KEY")!,
    xApiSecret: Deno.env.get("X_API_SECRET")!,
    xAccessToken: Deno.env.get("X_ACCESS_TOKEN")!,
    xAccessSecret: Deno.env.get("X_ACCESS_SECRET")!,
  };
}

/**
 * Derives a valid Telegram secret token [a-zA-Z0-9_-]{1,256} from the bot token.
 * Telegram Bot API requires secret tokens to only contain [a-zA-Z0-9_-].
 * SHA-256 hex digest produces a 64-character string of [0-9a-f], perfectly compliant.
 */
async function deriveSecret(token: string): Promise<string> {
  const data = new TextEncoder().encode(token);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Returns the expected webhook secret:
 * 1. TELEGRAM_WEBHOOK_SECRET if explicitly configured.
 * 2. SHA-256 hex digest of TELEGRAM_BOT_TOKEN as fallback.
 */
async function getExpectedWebhookSecret(): Promise<string> {
  const configured = Deno.env.get("TELEGRAM_WEBHOOK_SECRET")?.trim();
  if (configured) {
    return configured;
  }
  const botToken = Deno.env.get("TELEGRAM_BOT_TOKEN")?.trim() ?? "";
  if (!botToken) {
    console.error("[Security] Neither TELEGRAM_WEBHOOK_SECRET nor TELEGRAM_BOT_TOKEN is set.");
    return "";
  }
  return await deriveSecret(botToken);
}

/**
 * Performs a constant-time string comparison to mitigate timing attacks.
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let mismatch = 0;
  for (let i = 0; i < a.length; i++) {
    mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return mismatch === 0;
}

/**
 * Extracts chat.id from incoming Telegram update.
 * Handles both standard message.chat.id and inline callback_query.message.chat.id,
 * plus edited messages and channel posts.
 */
function extractChatId(update: TelegramUpdate): string | null {
  if (update.callback_query?.message?.chat?.id !== undefined) {
    return String(update.callback_query.message.chat.id);
  }
  if (update.message?.chat?.id !== undefined) {
    return String(update.message.chat.id);
  }
  if (update.edited_message?.chat?.id !== undefined) {
    return String(update.edited_message.chat.id);
  }
  if (update.channel_post?.chat?.id !== undefined) {
    return String(update.channel_post.chat.id);
  }
  if (update.edited_channel_post?.chat?.id !== undefined) {
    return String(update.edited_channel_post.chat.id);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Main Handler
// ---------------------------------------------------------------------------

if (import.meta.main) {
  Deno.serve(async (req: Request): Promise<Response> => {
  // ── Layer 1: Secret Token Header Check ────────────────────────────────────
  // Read X-Telegram-Bot-Api-Secret-Token from incoming request before doing anything else
  const secretHeader = req.headers.get("X-Telegram-Bot-Api-Secret-Token");
  const expectedSecret = await getExpectedWebhookSecret();

  if (!secretHeader || !expectedSecret || !timingSafeEqual(secretHeader, expectedSecret)) {
    console.warn("[Security] Unauthorized: Invalid or missing X-Telegram-Bot-Api-Secret-Token header.");
    return new Response("Unauthorized", { status: 401 });
  }

  // Only accept POST requests from Telegram
  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  // Parse JSON payload
  let update: TelegramUpdate;
  try {
    update = await req.json();
  } catch {
    return new Response("Bad Request: invalid JSON", { status: 400 });
  }

  // ── Layer 2: Chat ID Check ────────────────────────────────────────────────
  // Extract chat.id from payload and compare strictly against TELEGRAM_CHAT_ID
  const expectedChatId = Deno.env.get("TELEGRAM_CHAT_ID")?.trim();
  const incomingChatId = extractChatId(update);

  if (!expectedChatId || !incomingChatId || incomingChatId !== expectedChatId) {
    console.warn(
      `[Security] Forbidden: Incoming chat.id '${incomingChatId}' does not match TELEGRAM_CHAT_ID '${expectedChatId}'.`
    );
    return new Response("Forbidden", { status: 403 });
  }

  const callbackQuery = update.callback_query;
  if (!callbackQuery) {
    // Not a callback query (e.g. regular text message from authorized chat) – acknowledge silently
    return new Response("OK", { status: 200 });
  }

  const { data: cbData, message, id: callbackId } = callbackQuery;
  if (!message || !cbData) {
    return new Response("OK", { status: 200 });
  }

  const chatId = Number(message.chat.id);
  const messageId = message.message_id;

  const env = readEnv();
  const supabase = createClient(env.supabaseUrl, env.supabaseKey);

  // ── Answer the callback query immediately (removes Telegram spinner) ─
  await answerCallbackQuery(callbackId);

  // ── Route based on callback_data ─────────────────────────────────────
  if (cbData.startsWith("reject_")) {
    const postId = cbData.replace("reject_", "");
    await handleReject(supabase, postId, chatId, messageId);

  } else if (cbData.startsWith("approve_")) {
    const postId = cbData.replace("approve_", "");
    await handleApprove(supabase, postId, chatId, messageId);

  } else {
    console.warn("[Webhook] Unknown callback_data:", cbData);
  }

  return new Response("OK", { status: 200 });
  });
}

// ---------------------------------------------------------------------------
// Reject Handler
// ---------------------------------------------------------------------------

async function handleReject(
  supabase: SupabaseClient,
  postId: string,
  chatId: number,
  messageId: number,
): Promise<void> {
  console.log(`[Webhook] Rejecting post: ${postId}`);

  // Only a draft still awaiting approval (PENDING) can be rejected. A
  // conditional UPDATE is atomic: if the row has been claimed or already
  // published (PUBLISHING / PUBLISHED / PARTIAL_FAILURE / REJECTED), no row
  // matches and the rejection is a no-op.
  const { data: rejected, error } = await supabase
    .from("posts")
    .update({ status: "REJECTED", updated_at: nowIso() })
    .eq("id", postId)
    .eq("status", "PENDING")
    .select("id")
    .maybeSingle();

  if (error || !rejected) {
    await editTelegramMessage(
      chatId,
      messageId,
      `⚠️ Draft \`${postId}\` could not be rejected: it is no longer awaiting approval.\n_It may already be processing or partially published – tap ✅ Approve & Publish to retry._`,
    );
    return;
  }

  await editTelegramMessage(
    chatId,
    messageId,
    `❌ *Post Rejected*\n\nDraft \`${postId}\` has been discarded.\n_No content was published._`,
  );
}

// ---------------------------------------------------------------------------
// Approve Handler
// ---------------------------------------------------------------------------

async function handleApprove(
  supabase: SupabaseClient,
  postId: string,
  chatId: number,
  messageId: number,
): Promise<void> {
  console.log(`[Webhook] Approving post: ${postId}`);

  // ── Atomic claim ─────────────────────────────────────────────────────
  // A conditional UPDATE claims the draft for THIS approval event only
  // (status IN (PENDING, PARTIAL_FAILURE) → PUBLISHING). If another event
  // already claimed it (double tap, concurrent webhook, or a retry racing
  // ahead), no row matches the WHERE clause and we bail out – so a draft is
  // never published (or re-published) twice. updated_at lets the scheduled
  // cleanup recover rows whose claim never resolved (see schema.sql).
  const { data: claimed, error: claimError } = await supabase
    .from("posts")
    .update({ status: "PUBLISHING", updated_at: nowIso() })
    .eq("id", postId)
    .in("status", ["PENDING", "PARTIAL_FAILURE"])
    .select("id, content, platform, linkedin_status, x_status, linkedin_post_id, x_post_id")
    .maybeSingle();

  if (claimError || !claimed) {
    console.error(
      "[Webhook] Could not claim draft for publishing:",
      claimError ?? `no row in claimable state (id=${postId})`,
    );
    await explainClaimFailure(supabase, postId, chatId, messageId);
    return;
  }

  const post = claimed as PostRow;

  // Parse platform-specific drafts from combined content field.
  const { linkedinText, xText } = parseDraftContent(post.content);

  // ‒ Publish ONLY platforms that are SAFE to (re)attempt ───────────────
  // selectPublishTargets publishes a platform only when its per-platform
  // status is PENDING (never attempted) or FAILED (definitively rejected).
  // PUBLISHED → already done; PUBLISHING → an in-flight / unconfirmed attempt
  // from this or an earlier claim, and it is NEVER auto-republished. That is
  // what prevents a duplicate post when a stale PUBLISHING record is
  // recovered (cleanup reverts top-level PUBLISHING → PENDING, but a platform
  // the previous claim was publishing is now left PUBLISHING and skipped) or
  // when the process dies between the external publish and the durable record.
  const attempted: AttemptResult[] = await Promise.all(
    selectPublishTargets({ linkedin: post.linkedin_status, x: post.x_status }).map(
      (platform) => publishAndRecord(
        supabase,
        postId,
        platform,
        () => platform === "LinkedIn"
          ? publishToLinkedIn(linkedinText)
          : publishToX(xText),
      ),
    ),
  );

  // ── Determine final status from the durable per-platform statuses ────
  // Never derived from in-memory results alone — the same outcome must hold
  // no matter which claim (first attempt, partial retry, or recovery) ran.
  const { finalStates, finalStatus } = applyAttemptOutcomes(
    { linkedin: post.linkedin_status, x: post.x_status },
    attempted,
  );

  // ── Final top-level status MUST be durable before any success is reported ──
  // If this write fails the row is left claiming (PUBLISHING) while isolated
  // per-platform records may already be PUBLISHED — an inconsistent database.
  // Never report that as a clean success: surface the failure, best-effort
  // revert the row to PENDING so a retry settles it from the durable per-
  // platform statuses (cleanup_stale_drafts is the scheduled fallback).
  try {
    await persistFinalStatus(supabase, postId, finalStatus);
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`[Webhook] Final status persistence failed for ${postId}:`, errorMsg);
    const { error: settleErr } = await supabase
      .from("posts")
      .update({ status: "PENDING", updated_at: nowIso() })
      .eq("id", postId);
    if (settleErr) {
      console.error(
        `[Webhook] Could not revert ${postId} to PENDING for retry; awaiting cleanup:`,
        settleErr.message,
      );
    }
    await editTelegramMessage(
      chatId,
      messageId,
      `⚠️ *Publishing could not be recorded*\n\nDraft \`${postId}\` was processed but its final status could not be saved to the database.\n_Check the logs. Tap ✅ Approve & Publish to retry._`,
      buildInlineKeyboard(postId),
    );
    return;
  }

  // ── Build and send execution report ─────────────────────────────────
  const report = buildExecutionReport(postId, post, attempted, finalStates, finalStatus);
  const fullyPublished = finalStatus === "PUBLISHED";
  await editTelegramMessage(
    chatId,
    messageId,
    report,
    fullyPublished ? undefined : buildInlineKeyboard(postId),
  );

  console.log(`[Webhook] Post ${postId} → ${finalStatus} (per-platform ${JSON.stringify(finalStates)})`);
}

/**
 * Explains why an approve tap could not claim the draft, distinguishing an
 * in-flight publish (keep the buttons, suggest waiting) from a resolved post
 * (remove the buttons – there is nothing left to approve).
 */
async function explainClaimFailure(
  supabase: SupabaseClient,
  postId: string,
  chatId: number,
  messageId: number,
): Promise<void> {
  let rowStatus: string | null = null;
  try {
    const { data: statusRow } = await supabase
      .from("posts")
      .select("status")
      .eq("id", postId)
      .maybeSingle();
    rowStatus = statusRow?.status ?? null;
  } catch {
    rowStatus = null;
  }

  if (rowStatus === "PUBLISHING") {
    await editTelegramMessage(
      chatId,
      messageId,
      `⏳ *Still publishing*\n\nDraft \`${postId}\` is currently being published.\n_No duplicate was created – wait a few minutes, then tap ✅ to retry if it stays stuck._`,
      buildInlineKeyboard(postId),
    );
    return;
  }

  await editTelegramMessage(
    chatId,
    messageId,
    `⚠️ *Already ${rowStatus ?? "processed"}*\n\nDraft \`${postId}\` could not be claimed for publishing.\n_No content was published or re-published._`,
  );
}

// ---------------------------------------------------------------------------
// LinkedIn Publisher (Edge Function version)
// ---------------------------------------------------------------------------

async function publishToLinkedIn(text: string): Promise<string | null> {
  const env = readEnv();
  const payload = {
    author: env.linkedinAuthorUrn,
    lifecycleState: "PUBLISHED",
    specificContent: {
      "com.linkedin.ugc.ShareContent": {
        shareCommentary: { text },
        shareMediaCategory: "NONE",
      },
    },
    visibility: {
      "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC",
    },
  };

  const resp = await fetch("https://api.linkedin.com/v2/ugcPosts", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.linkedinAccessToken}`,
      "Content-Type": "application/json",
      "X-Restli-Protocol-Version": "2.0.0",
    },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`LinkedIn API ${resp.status}: ${body.slice(0, 300)}`);
  }

  // LinkedIn identifies the created post in the `x-restli-id` RESPONSE HEADER
  // as a URN (e.g. urn:li:ugcPost:68447855235931240). Persist the returned
  // value for per-platform deduplication.
  return resp.headers.get("x-restli-id");
}

// ---------------------------------------------------------------------------
// X (Twitter) Publisher – OAuth 1.0a (HMAC-SHA1)
// ---------------------------------------------------------------------------

async function publishToX(text: string): Promise<string | null> {
  /**
   * X API v2 requires OAuth 1.0a for write operations.
   * This implementation manually signs the request using Web Crypto API
   * (available in Deno / Supabase Edge Runtime).
   */
  const env = readEnv();
  const url = "https://api.twitter.com/2/tweets";
  const method = "POST";

  const oauthParams: Record<string, string> = {
    oauth_consumer_key: env.xApiKey,
    oauth_nonce: generateNonce(),
    oauth_signature_method: "HMAC-SHA1",
    oauth_timestamp: String(Math.floor(Date.now() / 1000)),
    oauth_token: env.xAccessToken,
    oauth_version: "1.0",
  };

  const signature = await buildOAuthSignature(
    method,
    url,
    oauthParams,
    {},
    env.xApiSecret,
    env.xAccessSecret,
  );
  oauthParams["oauth_signature"] = signature;

  const authHeader = buildAuthorizationHeader(oauthParams);

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": authHeader,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`X API ${resp.status}: ${body.slice(0, 300)}`);
  }

  // X returns the new tweet id as data.id → persisted for deduplication.
  const data = await resp.json();
  const tweetId = data?.data?.id;
  return typeof tweetId === "string" ? tweetId : null;
}

// ---------------------------------------------------------------------------
// OAuth 1.0a Signing – see ./oauth.ts
// ---------------------------------------------------------------------------
// percentEncode / buildOAuthSignature / buildAuthorizationHeader are shared
// and unit-tested in oauth.ts (RFC 3986 encoding). They are imported above.

function nowIso(): string {
  return new Date().toISOString();
}

// ---------------------------------------------------------------------------
// Telegram API Helpers
// ---------------------------------------------------------------------------

async function answerCallbackQuery(callbackQueryId: string): Promise<void> {
  const env = readEnv();
  await fetch(
    `https://api.telegram.org/bot${env.telegramBotToken}/answerCallbackQuery`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ callback_query_id: callbackQueryId }),
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    },
  );
}

async function editTelegramMessage(
  chatId: number,
  messageId: number,
  text: string,
  keyboard?: ReplyKeyboard,
): Promise<void> {
  const env = readEnv();
  await fetch(
    `https://api.telegram.org/bot${env.telegramBotToken}/editMessageText`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        message_id: messageId,
        text: text.slice(0, 4096),
        parse_mode: "Markdown",
        // keep inline buttons for retry, or remove them once fully resolved
        reply_markup: keyboard ?? { inline_keyboard: [] },
      }),
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    },
  );
}

// ---------------------------------------------------------------------------
// Utility Helpers
// ---------------------------------------------------------------------------

function parseDraftContent(content: string): { linkedinText: string; xText: string } {
  /**
   * Content stored by agent.py in format:
   *   "LINKEDIN:\n<draft>\n\nX:\n<draft>"
   */
  const parts = content.split("\n\nX:\n");
  const linkedin = parts[0]?.replace(/^LINKEDIN:\n/, "").trim() ?? content;
  const x = parts[1]?.trim() ?? content;
  return { linkedinText: linkedin, xText: x };
}

/**
 * Publishes ONE platform and records its outcome, with a write-ahead marker:
 *
 *   1. durably set `<platform>_status = PUBLISHING` BEFORE the external call,
 *   2. call the platform,
 *   3. durably set PUBLISHED (+ post id) or FAILED as soon as the call resolves.
 *
 * If the process dies between any two steps, the per-platform status is left
 * PUBLISHING (unconfirmed). Because selectPublishTargets NEVER auto-republishes
 * a PUBLISHING platform, a recovery/retry cannot create a duplicate post —
 * the unavoidable at-most-once gap is replaced by "unconfirmed, wait for
 * manual verification", which is the safest legal outcome for an external API
 * with no request-idempotency keys.
 */
export async function publishAndRecord(
  supabase: SupabaseClient,
  postId: string,
  platform: PlatformName,
  run: () => Promise<string | null>,
): Promise<AttemptResult> {
  const column = platform === "LinkedIn" ? "linkedin" : "x";

  // ── Step 1: write-ahead in-flight marker (durable BEFORE side effects) ──
  try {
    await recordPerPlatformStatus(supabase, postId, column, "PUBLISHING", null);
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(`[Webhook] Could not mark ${platform} publish intent:`, errorMsg);
    return { platform, success: false, error: `could not record publish intent: ${errorMsg}` };
  }

  // ── Step 2: external publish ─────────────────────────────────────────
  let publishedId: string | null;
  try {
    publishedId = (await run()) ?? null;
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    if (isAmbiguousPlatformError(err)) {
      // Timeout / 5xx / network: the platform MAY have accepted the request.
      // Keep the durable PUBLISHING marker (unconfirmed) — never auto-retry.
      console.error(`[Webhook] ${platform} publish outcome UNCONFIRMED:`, errorMsg);
      return {
        platform,
        success: false,
        unconfirmed: true,
        error: `outcome unconfirmed (may have posted): ${errorMsg}`,
      };
    }
    // Definitive rejection (HTTP 4xx – invalid auth/payload): record FAILED,
    // which makes the platform safely retryable.
    console.error(`[Webhook] ${platform} publish failed:`, errorMsg);
    try {
      await recordPerPlatformStatus(supabase, postId, column, "FAILED", null);
    } catch (recordErr) {
      console.error(`[Webhook] Could not record ${platform} failure:`, recordErr);
    }
    return { platform, success: false, error: errorMsg };
  }

  // ── Step 3: durable record of the confirmed post ─────────────────────
  try {
    await recordPerPlatformStatus(supabase, postId, column, "PUBLISHED", publishedId);
    return { platform, success: true, post_id: publishedId };
  } catch (err) {
    // The post IS live but its record failed; the per-platform status remains
    // 'PUBLISHING' from step 1 — unconfirmed, so a recovery never republishes
    // it (that would duplicate the live post).
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error(
      `[Webhook] ${platform} was published (${publishedId}) but its record failed:`,
      errorMsg,
    );
    return {
      platform,
      success: false,
      unconfirmed: true,
      post_id: publishedId,
      error: `published (${publishedId ?? "n/a"}) but record failed: ${errorMsg}`,
    };
  }
}

export async function recordPerPlatformStatus(
  supabase: SupabaseClient,
  postId: string,
  column: "linkedin" | "x",
  status: "PUBLISHING" | "PUBLISHED" | "FAILED",
  platformPostId: string | null,
): Promise<void> {
  // The Supabase client does NOT throw on response errors — inspect them, or a
  // failed per-platform write would be swallowed and the publish reported as a
  // clean success against a database that never recorded it.
  const { error } = await supabase
    .from("posts")
    .update({
      [`${column}_status`]: status,
      [`${column}_post_id`]: platformPostId,
      updated_at: nowIso(),
    } as Record<string, unknown>)
    .eq("id", postId);
  if (error) {
    throw new Error(
      `could not persist ${column} status '${status}' for post ${postId}: ${error.message}`,
    );
  }
}

/**
 * Persists the final top-level post status. Captures and checks the Supabase
 * response so a failed write surfaces (thrown up to handleApprove) instead of
 * silently claiming success while the database is left inconsistent.
 */
export async function persistFinalStatus(
  supabase: SupabaseClient,
  postId: string,
  status: FinalStatus,
): Promise<void> {
  const { error } = await supabase
    .from("posts")
    .update({ status, updated_at: nowIso() })
    .eq("id", postId);
  if (error) {
    throw new Error(
      `could not persist final status '${status}' for post ${postId}: ${error.message}`,
    );
  }
}

function buildInlineKeyboard(postId: string): ReplyKeyboard {
  return {
    inline_keyboard: [
      [
        { text: "✅ Approve & Publish", callback_data: `approve_${postId}` },
        { text: "❌ Reject", callback_data: `reject_${postId}` },
      ],
    ],
  };
}

function buildExecutionReport(
  postId: string,
  post: PostRow,
  attempted: AttemptResult[],
  finalStates: PlatformState,
  finalStatus: FinalStatus,
): string {
  const statusEmoji = finalStatus === "PUBLISHED" ? "✅" : "⚠️";
  const statusLabel = finalStatus === "PUBLISHED"
    ? "Successfully Published"
    : finalStatus === "PARTIAL_FAILURE"
      ? "Partial Failure"
      : finalStatus === "PENDING" && attempted.some((r) => r.unconfirmed)
        ? "Published Unconfirmed"
        : "Publishing Failed";

  const attemptedById = new Map(attempted.map((r) => [r.platform, r]));
  const lineFor = (platform: PlatformName): string => {
    const status = normalizeStatus(finalStates[platform === "LinkedIn" ? "linkedin" : "x"]);
    const attempt = attemptedById.get(platform);
    const storedId = platform === "LinkedIn"
      ? post.linkedin_post_id
      : post.x_post_id;
    const ref = attempt?.post_id ?? storedId ?? undefined;
    if (status === "PUBLISHED") {
      return `  ✅ *${platform}*: Published${ref ? ` (\`${ref}\`)` : ""}`;
    }
    if (status === "PUBLISHING") {
      // Unconfirmed: the post may be live (crashed after publish, or a failed
      // record write). We surfaced it loudly and never auto-republish it —
      // that would duplicate the post.
      return `  ⚠️ *${platform}*: Post may have gone out (cannot confirm)${ref ? ` (\`${ref}\`)` : ""}\n      _Verify manually — not auto-retried (would duplicate)._`;
    }
    return `  ❌ *${platform}*: Failed\n      \`${(attempt?.error ?? "").slice(0, 120)}\``;
  };

  const lines = PLATFORMS.map(lineFor);

  const sections = [
    `${statusEmoji} *${statusLabel}*`,
    "",
    `🆔 Draft ID: \`${postId}\``,
    "",
    "📊 *Results:*",
    ...lines,
    "",
    `_Status: ${finalStatus}_`,
  ];

  if (finalStatus !== "PUBLISHED") {
    sections.push("", "_Tap ✅ Approve & Publish to retry the missing platform(s)._");
  }

  return sections.join("\n");
}
