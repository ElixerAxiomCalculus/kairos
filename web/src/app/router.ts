import { useSyncExternalStore } from "react";

/** Minimal history router (5 routes don't justify a routing library). */
export type Route = "/" | "/live" | "/benchmark" | "/architecture" | "/experiments";
const ROUTES: Route[] = ["/", "/live", "/benchmark", "/architecture", "/experiments"];

function current(): Route {
  const path = window.location.pathname.replace(/\/+$/, "") || "/";
  return (ROUTES as string[]).includes(path) ? (path as Route) : "/";
}

const listeners = new Set<() => void>();
window.addEventListener("popstate", () => listeners.forEach((l) => l()));

export function navigate(to: Route): void {
  if (to === current()) return;
  window.history.pushState({}, "", to + window.location.search);
  listeners.forEach((l) => l());
  window.scrollTo(0, 0);
}

export function useRoute(): Route {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    current,
    () => "/" as Route,
  );
}
