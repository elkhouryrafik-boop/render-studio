import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ApiError, api, describeError, type User } from "./api";
import { Field, Note, Section } from "./components";

type Mode = "login" | "code" | "signup";

/**
 * What a signup that needs approval leaves behind. The password is kept only so
 * the page can sign the person in the moment approval lands. It lives in
 * component state and nowhere else: not storage, not the URL, not a log line.
 */
interface Waiting {
  email: string;
  password: string;
  since: number;
}

const APPROVAL_POLL_MS = 15_000;
const ELAPSED_TICK_MS = 30_000;

function askedAgo(since: number, now: number): string {
  const minutes = Math.floor((now - since) / 60_000);
  if (minutes < 1) return "Asked just now";
  return minutes === 1 ? "Asked 1 minute ago" : `Asked ${minutes} minutes ago`;
}

/**
 * The way in.
 *
 * The access code is checked before the signup form is shown, purely so nobody
 * fills in three fields and is then told the code was wrong. It grants nothing:
 * signup checks it again server-side, because a gate the client can skip is
 * not a gate.
 *
 * Signing in does not need the code. It is the door for people who already have
 * an account, and asking for a shared secret on every login would guarantee it
 * ends up written on a monitor.
 */
const SIGNED_IN_FLAG = "render.hasSignedIn";

function hasSignedIn(): boolean {
  try {
    return window.localStorage.getItem(SIGNED_IN_FLAG) === "1";
  } catch {
    return true;
  }
}

/** Remembered so the next visit opens on "Sign in". Storage may be blocked. */
export function rememberSignedIn() {
  try {
    window.localStorage.setItem(SIGNED_IN_FLAG, "1");
  } catch {
    /* private window: the gate simply opens on Create an account again */
  }
}

export function Gate({
  onSignedIn,
  serverDown,
  expired,
}: {
  onSignedIn: (user: User) => void;
  serverDown: boolean;
  expired: boolean;
}) {
  // A browser that has never signed in here opens on "Create an account".
  const [mode, setMode] = useState<Mode>(() => (hasSignedIn() ? "login" : "code"));
  const [code, setCode] = useState("");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Set when a signup comes back needing approval. While it is set the page is
  // the waiting panel, and the poll below is what eventually signs the person in.
  const [waiting, setWaiting] = useState<Waiting | null>(null);
  // The approval lookup said this address is no longer waiting (unknown or
  // suspended). Polling stops and the panel says so.
  const [gone, setGone] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  // The parent passes a fresh callback on most renders. Reading it through a ref
  // keeps the poll from restarting every time the parent re-renders.
  const signedIn = useRef(onSignedIn);
  useEffect(() => {
    signedIn.current = onSignedIn;
  }, [onSignedIn]);

  // A local copy so the retry below can clear it: an ApiError coming back means
  // the server is answering again, which the parent cannot learn from here
  // without a successful sign-in.
  const [down, setDown] = useState(serverDown);
  useEffect(() => setDown(serverDown), [serverDown]);

  // Poll for approval. Pending answers are free on the server, so this is cheap.
  // Any network blip is ignored and the next tick tries again.
  useEffect(() => {
    if (!waiting || gone) return;
    let live = true;
    const check = async () => {
      let state: "pending" | "active" | "unknown";
      try {
        state = (await api.approval(waiting.email)).status;
      } catch {
        return;
      }
      if (!live || state === "pending") return;
      if (state === "unknown") {
        setGone(true);
        return;
      }
      // Approved. Sign in with the held credentials and hand off exactly as a
      // normal sign-in does.
      try {
        const user = await api.login(waiting.email, waiting.password);
        if (!live) return;
        setWaiting(null);
        signedIn.current(user);
      } catch (e) {
        if (!live) return;
        setWaiting(null);
        setError(describeError(e));
      }
    };
    const timer = window.setInterval(check, APPROVAL_POLL_MS);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [waiting, gone]);

  // Only the elapsed-time line reads the clock, so it needs a slow tick, not a
  // render per second.
  useEffect(() => {
    if (!waiting) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), ELAPSED_TICK_MS);
    return () => window.clearInterval(timer);
  }, [waiting]);

  const fail = (e: unknown) => setError(describeError(e));

  // Switching mode always clears the last failure. A wrong password left its
  // red box sitting over the access-code form, which reads as the code being
  // wrong before it has even been typed.
  const go = (next: Mode) => {
    setMode(next);
    setError(null);
  };

  const useAnotherEmail = () => {
    setWaiting(null);
    setGone(false);
    setEmail("");
    setPassword("");
    setError(null);
    setMode("signup");
  };

  const retry = async () => {
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await api.me());
    } catch (e) {
      // The server answered, it just does not know us. That is the ordinary
      // signed-out case, so show the form again.
      if (e instanceof ApiError) setDown(false);
      else fail(e);
    } finally {
      setBusy(false);
    }
  };

  const submitLogin = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await api.login(email, password));
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  const submitCode = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.checkAccessCode(code);
      setMode("signup");
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  const submitSignup = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await api.signup({ access_code: code, email, name, password });
      if (result.needs_approval) {
        setGone(false);
        setWaiting({ email: email.trim(), password, since: Date.now() });
        setPassword("");
      } else {
        onSignedIn(await api.me());
      }
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="gate">
      <div className="gate-panel">
        <div className="mark">
          <span className="mark-names">
            <h1 className="app-name">Render Studio</h1>
            <span className="app-sub">AI rendering for architects</span>
          </span>
        </div>

        {down ? (
          <div className="stack">
            <Note kind="error" live>
              Cannot reach the render server. Signing in will not work until it answers.
            </Note>
            <p className="hint">
              It is usually restarting and comes back on its own within a minute. If it stays
              quiet, tell whoever runs the office server.
            </p>
            {error && (
              <Note kind="error" live>
                {error}
              </Note>
            )}
            <button
              type="button"
              className="btn btn-primary btn-block"
              onClick={retry}
              disabled={busy}
            >
              {busy && <span className="spin" />}
              Try again
            </button>
          </div>
        ) : waiting ? (
          <div className="stack">
            <Section title="Waiting for approval" />
            <div className="sheet">
              <div className="sheet-cell">
                <span className="sheet-key">Account</span>
                <span className="sheet-val">{waiting.email}</span>
              </div>
            </div>
            <p className="hint">
              An administrator sees this account in Accounts and approves it there. This page
              signs you in as soon as they do.
            </p>
            <p className="caption" role="status">
              {askedAgo(waiting.since, now)}. This page checks every 15 seconds.
            </p>
            {gone && (
              <Note kind="warn" live>
                This address is no longer waiting for approval. Ask an administrator what
                happened, or use a different email.
              </Note>
            )}
            {error && (
              <Note kind="error" live>
                {error}
              </Note>
            )}
            <button type="button" className="btn btn-ghost btn-block" onClick={useAnotherEmail}>
              Use a different email
            </button>
          </div>
        ) : (
          <div className="stack">
            <div
              className="switcher switcher--even"
              role="group"
              aria-label="Sign in, or create an account"
            >
              <button
                type="button"
                aria-pressed={mode === "login"}
                onClick={() => go("login")}
              >
                Sign in
              </button>
              <button
                type="button"
                aria-pressed={mode !== "login"}
                onClick={() => go(mode === "signup" ? "signup" : "code")}
              >
                Create an account
              </button>
            </div>

            {expired && mode === "login" && !error && (
              <Note kind="warn" live>
                Your session ended. Sign in again.
              </Note>
            )}
            {error && (
              <Note kind="error" live>
                {error}
              </Note>
            )}

            {mode === "login" && (
              <form className="stack" onSubmit={submitLogin}>
                <Field label="Email">
                  {(id) => (
                    <input
                      id={id}
                      type="email"
                      autoComplete="username"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  )}
                </Field>
                <Field label="Password">
                  {(id) => (
                    <input
                      id={id}
                      type="password"
                      autoComplete="current-password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                  )}
                </Field>
                <button className="btn btn-primary btn-block" disabled={busy}>
                  {busy && <span className="spin" />}
                  Sign in
                </button>
              </form>
            )}

            {mode === "code" && (
              <form className="stack" onSubmit={submitCode}>
                <Field
                  label="Office access code"
                  hint="Ask an administrator for the office code. Your account still needs approval before you can render."
                >
                  {(id) => (
                    <input
                      id={id}
                      type="password"
                      autoComplete="off"
                      value={code}
                      onChange={(e) => setCode(e.target.value)}
                      required
                      autoFocus
                    />
                  )}
                </Field>
                <button className="btn btn-primary btn-block" disabled={busy}>
                  {busy && <span className="spin" />}
                  Continue
                </button>
              </form>
            )}

            {mode === "signup" && (
              <form className="stack" onSubmit={submitSignup}>
                <p className="hint">
                  Office code accepted. Fill this in and an admin approves the account before
                  it can render.
                </p>
                <Field label="Name">
                  {(id) => (
                    <input
                      id={id}
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                      autoFocus
                    />
                  )}
                </Field>
                <Field label="Email">
                  {(id) => (
                    <input
                      id={id}
                      type="email"
                      autoComplete="username"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  )}
                </Field>
                <Field label="Password" hint="At least 10 characters.">
                  {(id) => (
                    <input
                      id={id}
                      type="password"
                      autoComplete="new-password"
                      minLength={10}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                  )}
                </Field>
                <button className="btn btn-primary btn-block" disabled={busy}>
                  {busy && <span className="spin" />}
                  Create account
                </button>
              </form>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
