import { useCallback, useEffect, useState } from "react";
import { Activity } from "./Activity";
import { Admin } from "./Admin";
import { Gate, rememberSignedIn } from "./Gate";
import { Studio } from "./Studio";
import { Note } from "./components";
import {
  api,
  ApiError,
  setUnauthorizedHandler,
  type Budget,
  type User,
} from "./api";

type View = "studio" | "admin" | "activity";

// How often an administrator's nav badge re-reads the waiting count.
const PENDING_POLL_MS = 60_000;

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);
  const [unreachable, setUnreachable] = useState(false);
  const [view, setView] = useState<View>("studio");
  const [budget, setBudget] = useState<Budget | null>(null);
  const [expired, setExpired] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);

  // A session is an httpOnly cookie, so the only way to know whether one is
  // valid is to ask. A 401 here is the normal signed-out case; anything else
  // means the server is unreachable, which is a different thing and used to be
  // reported identically.
  useEffect(() => {
    let live = true;
    api
      .me()
      .then((u) => {
        if (!live) return;
        setUser(u);
        setUnreachable(false);
      })
      .catch((e) => {
        if (!live) return;
        setUser(null);
        setUnreachable(!(e instanceof ApiError));
      })
      .finally(() => live && setChecked(true));
    return () => {
      live = false;
    };
  }, []);

  // Any 401 from anywhere means the session ended under the user: the cookie
  // expired, or an admin suspended the account. Without this they stay inside
  // the signed-in shell while every action returns "Not authenticated".
  useEffect(() => {
    setUnauthorizedHandler(() => {
      setUser(null);
      setExpired(true);
    });
    return () => setUnauthorizedHandler(null);
  }, []);

  // Admins get a count of accounts waiting on the Accounts tab, so a pending
  // signup is visible without opening it. Non-admins never ask: the route would
  // refuse them, and the count is theirs to see only in the admin view.
  useEffect(() => {
    if (user) rememberSignedIn();
  }, [user]);

  // Whoever signs in next starts on the Studio, not on the last person's tab.
  useEffect(() => {
    setView("studio");
  }, [user?.id]);

  const isAdmin = Boolean(user?.is_admin);
  useEffect(() => {
    if (!isAdmin) {
      setPendingCount(0);
      return;
    }
    let live = true;
    const refresh = () => {
      api
        .adminPending()
        .then((list) => live && setPendingCount(list.length))
        .catch(() => undefined);
    };
    refresh();
    const timer = window.setInterval(refresh, PENDING_POLL_MS);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [isAdmin]);

  const loadBudget = useCallback(() => {
    api
      .budget()
      .then(setBudget)
      .catch(() => setBudget(null));
  }, []);

  // Refetched after every submitted render, not only on sign-in. The remaining
  // cap was previously frozen for a whole working session, which is exactly
  // when it matters.
  useEffect(() => {
    if (!user) return;
    loadBudget();
  }, [user, view, loadBudget]);

  if (!checked) {
    return (
      <div className="gate">
        <p className="caption" role="status">
          Checking your session
        </p>
      </div>
    );
  }

  if (!user) {
    return (
      <Gate
        onSignedIn={(u) => {
          setUser(u);
          setExpired(false);
          setUnreachable(false);
        }}
        serverDown={unreachable}
        expired={expired}
      />
    );
  }

  const over = budget ? budget.month_user_usd >= budget.per_user_cap_usd : false;

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <header className="titlebar">
        <div className="titlebar-inner">
          <div className="mark">
            <span className="mark-names">
              <h1 className="app-name">Render Studio</h1>
              <span className="app-sub">AI rendering for architects</span>
            </span>
          </div>

          <div className="sheet">
            <div className="sheet-cell">
              <span className="sheet-key">Signed in</span>
              <span className="sheet-val">{user.name}</span>
            </div>
            {budget && (
              <div className="sheet-cell">
                <span className="sheet-key">Left this month</span>
                <span className={`sheet-val ${over ? "is-over" : ""}`}>
                  ${Math.max(0, budget.per_user_cap_usd - budget.month_user_usd).toFixed(2)}
                </span>
              </div>
            )}
          </div>
        </div>

        <nav className="nav" aria-label="Sections">
          <button
            aria-current={view === "studio" ? "page" : undefined}
            onClick={() => setView("studio")}
          >
            Studio
          </button>
          {user.is_admin && (
            <button
              aria-current={view === "admin" ? "page" : undefined}
              onClick={() => setView("admin")}
            >
              Accounts
              {pendingCount > 0 && (
                <span className="status status--count">
                  <span className="sr-only">, </span>
                  {pendingCount}
                  <span className="sr-only"> waiting for approval</span>
                </span>
              )}
            </button>
          )}
          {user.is_admin && (
            <button
              aria-current={view === "activity" ? "page" : undefined}
              onClick={() => setView("activity")}
            >
              Activity
            </button>
          )}
          <span className="spacer" />
          <button
            onClick={() =>
              api
                .logout()
                .catch(() => undefined)
                .finally(() => setUser(null))
            }
          >
            Sign out
          </button>
        </nav>
      </header>

      {user.engine === "fake" && (
        <p className="fake-banner" role="status">
          {user.is_admin
            ? "Fake engine: placeholder images, nothing is sent to fal.ai."
            : "Test mode: placeholder images, nothing is sent out."}
        </p>
      )}

      <main className="shell" id="main">
        {budget?.warning && (
          <Note kind="error" live>
            {budget.warning}
          </Note>
        )}

        {view === "studio" ? (
          <Studio onSpent={loadBudget} budget={budget} me={user.email} />
        ) : view === "activity" && user.is_admin ? (
          <Activity />
        ) : view === "admin" && user.is_admin ? (
          <Admin me={user} onPendingChange={setPendingCount} />
        ) : (
          <Studio onSpent={loadBudget} budget={budget} me={user.email} />
        )}
      </main>
    </>
  );
}
