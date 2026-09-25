/**
 * Unit tests for the X / Twitter OAuth 1.0a helpers (oauth.ts).
 *
 * Runs with `deno test supabase/functions/telegram-webhook/oauth_test.ts`
 * No external dependencies – Web Crypto API only.
 */

import {
  buildAuthorizationHeader,
  buildOAuthSignature,
  generateNonce,
  percentEncode,
} from "./oauth.ts";

function assertEq(actual: unknown, expected: unknown, msg?: string): void {
  if (actual !== expected) {
    throw new Error(
      `${msg ?? "assertEq"}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
    );
  }
}

function assertTrue(cond: boolean, msg?: string): void {
  if (!cond) {
    throw new Error(msg ?? "assertTrue failed");
  }
}

Deno.test("percentEncode leaves RFC 3986 unreserved characters unchanged", () => {
  assertEq(percentEncode("AZaz09-._~"), "AZaz09-._~");
});

Deno.test("percentEncode encodes space, +, comma and ! (unlike encodeURIComponent)", () => {
  assertEq(
    percentEncode("Hello Ladies + Gentlemen, a signed OAuth request!"),
    "Hello%20Ladies%20%2B%20Gentlemen%2C%20a%20signed%20OAuth%20request%21",
  );
  // encodeURIComponent would have left !, ' ( ) and * unencoded.
  assertEq(percentEncode("a!b'c(d)e*f"), "a%21b%27c%28d%29e%2Af");
});

Deno.test("percentEncode never uses '+' for spaces and emits uppercase hex", () => {
  assertEq(percentEncode("a b"), "a%20b");
  assertEq(percentEncode("\n"), "%0A");
  assertEq(percentEncode("é"), "%C3%A9");
});

Deno.test("generateNonce is URL/header-safe and stable in shape", () => {
  for (let i = 0; i < 100; i++) {
    const nonce = generateNonce();
    assertTrue(/^[a-zA-Z0-9]+$/.test(nonce), `nonce unsafe: ${nonce}`);
    assertTrue(nonce.length > 8, "nonce too short");
  }
});

Deno.test(
  "buildOAuthSignature reproduces the Twitter documented example (RFC 3986)",
  async () => {
    const signature = await buildOAuthSignature(
      "POST",
      "https://api.twitter.com/1/statuses/update.json",
      {
        oauth_consumer_key: "xvz1evFS4wEEPTGEFPHBog",
        oauth_nonce: "kYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg",
        oauth_signature_method: "HMAC-SHA1",
        oauth_timestamp: "1318622958",
        oauth_token: "370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb",
        oauth_version: "1.0",
      },
      {
        include_entities: "true",
        status: "Hello Ladies + Gentlemen, a signed OAuth request!",
      },
      "kAcSOqF21Fu85e7zjz7ZN2U4ZRhfV3WpwPAoE3Z7kBw",
      "LswwdoUaIvS8ltyTt5jkRh4J50vUPVVHtR2YPi5kE",
    );
    assertEq(signature, "tnnArxj06cWHq44gCs1OSKk/jLY=", "known-answer signature");
  },
);

Deno.test("buildAuthorizationHeader percent-encodes keys and values", () => {
  const header = buildAuthorizationHeader({
    oauth_consumer_key: "xvz1evFS4wEEPTGEFPHBog",
    oauth_signature: "t=sig&v!",
    oauth_token: "tok/~en",
  });
  assertTrue(header.startsWith('OAuth oauth_consumer_key="xvz1evFS4wEEPTGEFPHBog", '));
  assertTrue(header.includes('oauth_signature="t%3Dsig%26v%21"'), "signature must be RFC3986 encoded");
  assertTrue(header.includes('oauth_token="tok%2F~en"'), "~ stays, / is encoded");
});