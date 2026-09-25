/**
 * PersonaPulse – OAuth 1.0a helpers (X / Twitter API v2)
 * ======================================================
 * Single source of truth for the percent-encoding and request signing used
 * by the Telegram webhook's X (Twitter) publisher.
 *
 * Deno-compatible (Web Crypto API, no external deps).
 */

/**
 * RFC 3986/Section 3.6 "percent-encoding" for OAuth 1.0a.
 *
 * Unreserved characters (ALPHA / DIGIT / "-" / "." / "_" / "~") are left
 * unchanged; everything else is percent-encoded with UPPERCASE hex digits.
 *
 * JavaScript's `encodeURIComponent` is NOT RFC 3986 compliant: it leaves
 * `!'()*` unencoded. This helper is the ONLY encoder the signing and header
 * code may use, so the signature base string and the Authorization header
 * are always byte-identical to what a conforming OAuth client produces.
 */
export function percentEncode(value: string): string {
  return encodeURIComponent(value).replace(
    /[!'()*]/g,
    (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`,
  );
}

/**
 * OAuth nonce: strictly [a-zA-Z0-9] so it is always header-safe and never
 * needs quoting/encoding inside the Authorization header.
 */
export function generateNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes)).replace(/[^a-zA-Z0-9]/g, "");
}

/**
 * Builds the OAuth 1.0a HMAC-SHA1 signature base string and signs it with
 * the combined consumer/token secret, returning the base64 signature.
 *
 * - Parameters are percent-encoded with `percentEncode`, sorted by encoded
 *   key, and joined as `key=value&...`.
 * - The base string is `METHOD&percentEncode(url)&percentEncode(paramString)`.
 * - The signing key is `percentEncode(consumerSecret)&percentEncode(tokenSecret)`.
 *
 * `oauthParams` must NOT contain `oauth_signature`; it is added by the caller
 * after signing. `bodyParams` is merged into the parameter set (form params).
 */
export async function buildOAuthSignature(
  method: string,
  url: string,
  oauthParams: Record<string, string>,
  bodyParams: Record<string, string>,
  consumerSecret: string,
  tokenSecret: string,
): Promise<string> {
  const allParams = { ...oauthParams, ...bodyParams };
  const sortedKeys = Object.keys(allParams).sort();
  const paramString = sortedKeys
    .map((k) => `${percentEncode(k)}=${percentEncode(allParams[k])}`)
    .join("&");

  const baseString = [
    method.toUpperCase(),
    percentEncode(url),
    percentEncode(paramString),
  ].join("&");

  const signingKey = `${percentEncode(consumerSecret)}&${percentEncode(tokenSecret)}`;

  const keyData = new TextEncoder().encode(signingKey);
  const msgData = new TextEncoder().encode(baseString);
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    keyData,
    { name: "HMAC", hash: "SHA-1" },
    false,
    ["sign"],
  );
  const signatureBuffer = await crypto.subtle.sign("HMAC", cryptoKey, msgData);
  return btoa(String.fromCharCode(...new Uint8Array(signatureBuffer)));
}

/**
 * Builds the `Authorization: OAuth ...` header value using the same RFC 3986
 * encoding as the signature base string (keys and values are quoted).
 */
export function buildAuthorizationHeader(oauthParams: Record<string, string>): string {
  return "OAuth " + Object.entries(oauthParams)
    .map(([k, v]) => `${percentEncode(k)}="${percentEncode(v)}"`)
    .join(", ");
}