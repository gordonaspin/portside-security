export const ssr = false;
export const prerender = true;

export async function load({ fetch, url }) {
  // Skip ALL logic during prerender
  if (import.meta.env.PRERENDER) {
    return {};
  }

  // Allow login page without auth
  if (url.pathname.startsWith("/login")) {
    return {};
  }

  // Normal runtime auth
  const res = await fetch("/whoami", { credentials: "include" });

  if (!res.ok) {
    return {
      status: 302,
      redirect: "/login"
    };
  }

  return { user: await res.json() };
}
