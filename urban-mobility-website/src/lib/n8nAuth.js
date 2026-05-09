/**
 * Server-side n8n authentication helper.
 *
 * Logs in via POST /rest/login, caches the session cookie for 55 minutes
 * (n8n sessions expire after 60 min), and returns the cookie string
 * for use in subsequent authenticated requests.
 */

const N8N_URL = process.env.N8N_URL ?? "http://localhost:5678";
const N8N_EMAIL = process.env.N8N_EMAIL ?? "";
const N8N_PASSWORD = process.env.N8N_PASSWORD ?? "";

// Module-level token cache
let cachedCookie = null;
let cacheExpiry = 0; // epoch ms

/**
 * Authenticate with n8n and return a session cookie string.
 * The cookie is cached for 55 minutes to avoid re-authenticating on every call.
 *
 * @returns {Promise<string>} The n8n session cookie value
 * @throws {Error} If n8n is unreachable or credentials are invalid
 */
export async function getN8nToken() {
  // Return cached cookie if still valid
  if (cachedCookie && Date.now() < cacheExpiry) {
    return cachedCookie;
  }

  if (!N8N_EMAIL || !N8N_PASSWORD) {
    throw new Error("N8N_EMAIL and N8N_PASSWORD must be set in .env.local");
  }

  const res = await fetch(`${N8N_URL}/rest/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      emailAddress: N8N_EMAIL,
      password: N8N_PASSWORD,
    }),
    signal: AbortSignal.timeout(5000),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`n8n login failed (${res.status}): ${text}`);
  }

  // n8n returns the auth token in a Set-Cookie header (n8n-auth)
  // AND/OR in the response body as { data: { ... } }
  const setCookie = res.headers.get("set-cookie") ?? "";
  const data = await res.json().catch(() => ({}));

  // Extract the n8n-auth cookie value
  const cookieMatch = setCookie.match(/n8n-auth=([^;]+)/);
  const cookieValue = cookieMatch ? cookieMatch[1] : null;

  // Some n8n versions return a token in the response body instead
  const bodyToken = data?.data?.token ?? data?.token ?? null;

  if (cookieValue) {
    cachedCookie = `n8n-auth=${cookieValue}`;
    cacheExpiry = Date.now() + 55 * 60 * 1000; // 55 minutes
    return cachedCookie;
  }

  if (bodyToken) {
    cachedCookie = bodyToken;
    cacheExpiry = Date.now() + 55 * 60 * 1000;
    return cachedCookie;
  }

  // If login succeeded (200) but no cookie/token, use basic auth fallback
  throw new Error("n8n login returned 200 but no session cookie or token found");
}

/**
 * Build the appropriate auth headers for an n8n API request.
 * @returns {Promise<Record<string, string>>}
 */
export async function getN8nAuthHeaders() {
  const token = await getN8nToken();

  // If the token looks like a cookie string (contains "n8n-auth="), pass it as Cookie header
  if (token.startsWith("n8n-auth=")) {
    return { Cookie: token };
  }

  // Otherwise treat it as a Bearer token
  return { Authorization: `Bearer ${token}` };
}
