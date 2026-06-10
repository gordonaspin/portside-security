export const ssr = false;
export const prerender = false;

export async function load({ fetch, url }) {
  // Allow login page without auth
  if (url.pathname.startsWith("/login")) {
    return {};
  }

  const res = await fetch("/whoami", { credentials: "include" });

  if (!res.ok) {
    return {
      status: 302,
      redirect: "/login"
    };
  }

  return { user: await res.json() };
}