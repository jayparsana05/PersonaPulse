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
 *   approve_<post_id>  →  Fetch draft from Supabase
 *                      →  Publish to LinkedIn + X in parallel (Promise.allSettled)
 *                      →  Every fetch() wrapped in AbortSignal.timeout(8000)
 *                      →  Update Supabase status: PUBLISHED | PARTIAL_FAILURE
 *                      →  Edit Telegram message with execution report
 *
 *   reject_<post_id>   →  Update Supabase status: REJECTED
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
}

interface PublishResult {
  platform: string;
  success: boolean;
  error?: string;
}

// ---------------------------------------------------------------------------
// Constants & Security Helpers
// ---------------------------------------------------------------------------

const FETCH_TIMEOUT_MS = 8_000;  // AbortSignal.timeout cap per spec

const env = {
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

  await supabase
    .from("posts")
    .update({ status: "REJECTED" })
    .eq("id", postId);

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

  // ── Fetch the pending draft from Supabase ────────────────────────────
  const { data: rows, error } = await supabase
    .from("posts")
    .select("id, content, platform, status")
    .eq("id", postId)
    .eq("status", "PENDING")
    .limit(1);

  if (error || !rows || rows.length === 0) {
    console.error("[Webhook] Failed to fetch post:", error);
    await editTelegramMessage(
      chatId, messageId,
      `⚠️ *Error*: Could not find PENDING draft \`${postId}\`.\nIt may have already been processed.`,
    );
    return;
  }

  const post = rows[0] as PostRow;

  // Parse platform-specific drafts from combined content field
  const { linkedinText, xText } = parseDraftContent(post.content);

  // ── Publish to both platforms in parallel ────────────────────────────
  const [linkedinResult, xResult] = await Promise.allSettled([
    publishToLinkedIn(linkedinText),
    publishToX(xText),
  ]);

  const results: PublishResult[] = [
    evaluateSettled(linkedinResult, "LinkedIn"),
    evaluateSettled(xResult, "X (Twitter)"),
  ];

  // ── Determine final status ───────────────────────────────────────────
  const allSuccess = results.every((r) => r.success);
  const anySuccess = results.some((r) => r.success);
  const finalStatus = allSuccess ? "PUBLISHED"
    : anySuccess ? "PARTIAL_FAILURE"
      : "PARTIAL_FAILURE";

  await supabase
    .from("posts")
    .update({ status: finalStatus })
    .eq("id", postId);

  // ── Build and send execution report ─────────────────────────────────
  const report = buildExecutionReport(postId, results, finalStatus);
  await editTelegramMessage(chatId, messageId, report);

  console.log(`[Webhook] Post ${postId} → ${finalStatus}`);
}

// ---------------------------------------------------------------------------
// LinkedIn Publisher (Edge Function version)
// ---------------------------------------------------------------------------

async function publishToLinkedIn(text: string): Promise<void> {
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
}

// ---------------------------------------------------------------------------
// X (Twitter) Publisher – OAuth 1.0a (HMAC-SHA1)
// ---------------------------------------------------------------------------

async function publishToX(text: string): Promise<void> {
  /**
   * X API v2 requires OAuth 1.0a for write operations.
   * This implementation manually signs the request using Web Crypto API
   * (available in Deno / Supabase Edge Runtime).
   */
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

  const signature = await buildOAuthSignature(method, url, oauthParams, {});
  oauthParams["oauth_signature"] = signature;

  const authHeader = "OAuth " + Object.entries(oauthParams)
    .map(([k, v]) => `${encodeURIComponent(k)}="${encodeURIComponent(v)}"`)
    .join(", ");

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
}

// ---------------------------------------------------------------------------
// OAuth 1.0a Signing Helpers (Web Crypto / Deno-compatible)
// ---------------------------------------------------------------------------

function generateNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes)).replace(/[^a-zA-Z0-9]/g, "");
}

async function buildOAuthSignature(
  method: string,
  url: string,
  oauthParams: Record<string, string>,
  bodyParams: Record<string, string>,
): Promise<string> {
  const allParams = { ...oauthParams, ...bodyParams };
  const sortedKeys = Object.keys(allParams).sort();
  const paramString = sortedKeys
    .map((k) => `${encodeURIComponent(k)}=${encodeURIComponent(allParams[k])}`)
    .join("&");

  const baseString = [
    method.toUpperCase(),
    encodeURIComponent(url),
    encodeURIComponent(paramString),
  ].join("&");

  const signingKey = `${encodeURIComponent(env.xApiSecret)}&${encodeURIComponent(env.xAccessSecret)}`;

  const keyData = new TextEncoder().encode(signingKey);
  const msgData = new TextEncoder().encode(baseString);
  const cryptoKey = await crypto.subtle.importKey("raw", keyData, { name: "HMAC", hash: "SHA-1" }, false, ["sign"]);
  const signatureBuffer = await crypto.subtle.sign("HMAC", cryptoKey, msgData);
  return btoa(String.fromCharCode(...new Uint8Array(signatureBuffer)));
}

// ---------------------------------------------------------------------------
// Telegram API Helpers
// ---------------------------------------------------------------------------

async function answerCallbackQuery(callbackQueryId: string): Promise<void> {
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
): Promise<void> {
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
        reply_markup: { inline_keyboard: [] },  // remove buttons after action
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

function evaluateSettled(
  result: PromiseSettledResult<void>,
  platform: string,
): PublishResult {
  if (result.status === "fulfilled") {
    return { platform, success: true };
  }
  const errorMsg = result.reason instanceof Error
    ? result.reason.message
    : String(result.reason);
  console.error(`[Webhook] ${platform} publish failed:`, errorMsg);
  return { platform, success: false, error: errorMsg };
}

function buildExecutionReport(
  postId: string,
  results: PublishResult[],
  finalStatus: string,
): string {
  const statusEmoji = finalStatus === "PUBLISHED" ? "✅" : "⚠️";
  const statusLabel = finalStatus === "PUBLISHED"
    ? "Successfully Published"
    : "Partial Failure – check logs";

  const lines = results.map((r) =>
    r.success
      ? `  ✅ *${r.platform}*: Published`
      : `  ❌ *${r.platform}*: Failed\n      \`${(r.error ?? "").slice(0, 120)}\``
  );

  return [
    `${statusEmoji} *${statusLabel}*`,
    "",
    `🆔 Draft ID: \`${postId}\``,
    "",
    "📊 *Results:*",
    ...lines,
    "",
    `_Status: ${finalStatus}_`,
  ].join("\n");
}
