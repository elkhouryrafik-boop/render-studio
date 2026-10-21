import { useCallback, useEffect, useState } from "react";
import { api, describeError, type AdminControls, type User } from "./api";
import { Empty, Note, Section, Spec } from "./components";

interface Spend {
  month_total_usd: number;
  monthly_cap_usd: number;
  kill_switch: boolean;
  per_user: {
    user_id: number;
    email: string;
    name: string;
    status: string;
    spent_usd: number;
    cap_usd: number;
  }[];
}

type Message = { kind: "error" | "good" | "warn"; text: string };

/**
 * Approving people, and controlling what they can spend.
 *
 * Pending accounts are first because that is the queue with someone waiting at
 * the other end of it.
 */
export function Admin({
  me,
  onPendingChange,
}: {
  me: User;
  /** Told the waiting count whenever the list is read, so the nav badge stays true. */
  onPendingChange?: (count: number) => void;
}) {
  const [users, setUsers] = useState<User[]>([]);
  const [spend, setSpend] = useState<Spend | null>(null);
  const [controls, setControls] = useState<AdminControls | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [working, setWorking] = useState(false);
  const [message, setMessage] = useState<Message | null>(null);

  // Allowance inputs are controlled, keyed by user id. `defaultValue` kept
  // whatever had been typed after a refetch, so a figure the server clamped or
  // refused still sat in the box looking as though it had been saved.
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  // Clearing a cap is destructive and was silent. It is confirmed in place
  // rather than the moment focus leaves the field.
  const [clearing, setClearing] = useState<User | null>(null);

  const say = (kind: Message["kind"], text: string) => setMessage({ kind, text });
  const fail = (e: unknown) => say("error", describeError(e));

  const load = useCallback(() => {
    // The server is the truth for every allowance, so a reload drops the
    // drafts along with it.
    api
      .adminUsers()
      .then((list) => {
        setUsers(list);
        setDrafts({});
      })
      .catch((e) => say("error", describeError(e)))
      .finally(() => setLoaded(true));
    api
      .spend()
      .then(setSpend)
      .catch((e) => say("error", describeError(e)));
    api
      .adminControls()
      .then(setControls)
      .catch((e) => say("error", describeError(e)));
  }, []);

  useEffect(load, [load]);

  const act = async (fn: () => Promise<unknown>, text: string) => {
    setWorking(true);
    setMessage(null);
    try {
      await fn();
      say("good", text);
      load();
    } catch (e) {
      fail(e);
    } finally {
      setWorking(false);
    }
  };

  const setStop = async (on: boolean) => {
    setWorking(true);
    setMessage(null);
    try {
      setControls(await api.setKillSwitch(on));
      // Stopping is the red wash, opening again the green one.
      say(on ? "error" : "good", on ? "Spending is stopped for everyone." : "Spending is open again.");
    } catch (e) {
      fail(e);
    } finally {
      setWorking(false);
    }
  };

  const pending = users.filter((u) => u.status === "pending");
  const rest = users.filter((u) => u.status !== "pending");

  useEffect(() => {
    onPendingChange?.(pending.length);
    // The callback is a state setter in the parent; only the count matters here.
  }, [pending.length]);
  const spentBy = (id: number) => spend?.per_user.find((p) => p.user_id === id);

  const draftFor = (u: User) =>
    drafts[u.id] ?? (u.budget_monthly_usd == null ? "" : String(u.budget_monthly_usd));

  const setDraft = (id: number, value: string) =>
    setDrafts((current) => ({ ...current, [id]: value }));

  const dropDraft = (id: number) =>
    setDrafts((current) => {
      const next = { ...current };
      delete next[id];
      return next;
    });

  const commitBudget = (u: User) => {
    const raw = draftFor(u).trim();
    const current = u.budget_monthly_usd ?? null;

    if (raw === "") {
      if (current === null) {
        dropDraft(u.id);
        return;
      }
      setClearing(u);
      return;
    }

    // An empty or malformed number used to arrive at the server as a cap of
    // zero, which stops that person rendering entirely.
    const value = Number(raw);
    if (!Number.isFinite(value) || value < 0) {
      say(
        "error",
        `An allowance is a figure of zero dollars or more. "${raw}" is not one, so ${u.name} is unchanged.`,
      );
      dropDraft(u.id);
      return;
    }
    if (value === current) {
      dropDraft(u.id);
      return;
    }
    act(() => api.setBudget(u.id, value), `${u.name} is capped at $${value.toFixed(2)} a month.`);
  };

  return (
    <div className="stack">
      {message && (
        <Note kind={message.kind} live>
          {message.text}
        </Note>
      )}

      {clearing && (
        <Note
          kind="warn"
          live
          actions={
            <div className="actions">
              <button
                type="button"
                className="btn btn-sm btn-danger"
                disabled={working}
                onClick={() => {
                  const u = clearing;
                  setClearing(null);
                  act(() => api.setBudget(u.id, null), `${u.name} is back on the office default.`);
                }}
              >
                Clear the cap
              </button>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => {
                  dropDraft(clearing.id);
                  setClearing(null);
                }}
              >
                Keep it
              </button>
            </div>
          }
        >
          Clearing {clearing.name}&rsquo;s allowance discards their own figure of $
          {(clearing.budget_monthly_usd ?? 0).toFixed(2)} and puts them back on the office
          default.
        </Note>
      )}

      <section className="panel stop-panel" aria-label="Stop all spending">
        <Section title="Stop all spending" note={controls ? (controls.kill_switch ? "on" : "off") : "reading"} />
        <div className="stop-row">
          <p className="stop-text" id="stop-title">
            {controls?.kill_switch
              ? "Spending is stopped. Nobody can send a render until you open it again."
              : "Spending is open. Turn this on to stop every render in the office at once; it takes effect on the next request."}
          </p>
          <button
            type="button"
            role="switch"
            aria-checked={controls?.kill_switch ?? false}
            aria-label="Stop all spending"
            aria-describedby="stop-title"
            className={`switch ${controls?.kill_switch ? "is-on" : ""}`}
            disabled={working || controls == null}
            onClick={() => void setStop(!controls?.kill_switch)}
          >
            <span className="switch-knob" aria-hidden="true" />
            <span className="switch-word">{controls?.kill_switch ? "Stopped" : "Open"}</span>
          </button>
        </div>
      </section>

      <section className="panel">
        <Section
          title="Waiting for approval"
          note={pending.length ? `${pending.length}` : "none"}
        />
        <div className="stack">
          {pending.length === 0 ? (
            <Empty title="Nobody is waiting">
              Anyone who signs up with the office code appears here. Approving them is what lets
              them render.
            </Empty>
          ) : (
            <div className="table-scroll">
              <table className="table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Asked</th>
                    <th>
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {pending.map((u) => (
                    <tr key={u.id}>
                      <td>{u.name}</td>
                      <td className="caption">{u.email}</td>
                      <td className="caption">{(u.created_at ?? "").slice(0, 10) || "unknown"}</td>
                      <td>
                        <div className="actions">
                          <button
                            type="button"
                            className="btn btn-sm btn-primary"
                            disabled={working}
                            onClick={() => act(() => api.approve(u.id), `${u.name} approved.`)}
                          >
                            Approve
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="caption">
            Email addresses are not verified and nothing is sent to them. Anyone holding the
            office code can type any address, so approving is the check, not the address.
          </p>
        </div>
      </section>

      <section className="panel">
        <Section title="This month" note="office" />
        {spend ? (
          <Spec
            rows={[
              ["Spent", `$${spend.month_total_usd.toFixed(2)}`],
              ["Cap", `$${spend.monthly_cap_usd.toFixed(2)}`],
            ]}
            total={[
              "Remaining",
              `$${Math.max(0, spend.monthly_cap_usd - spend.month_total_usd).toFixed(2)}`,
            ]}
            over={spend.month_total_usd >= spend.monthly_cap_usd}
          />
        ) : (
          // Zeroes here read as an office cap of zero, which is a very
          // different sentence from "not read yet".
          <p className="caption" role="status">
            Reading this month&rsquo;s ledger.
          </p>
        )}
      </section>

      <section className="panel">
        <Section title="Accounts" note={`${rest.length}`} />
        <div className="stack">
          {!loaded ? (
            <p className="caption" role="status">
              Reading the account list.
            </p>
          ) : rest.length === 0 ? (
            <Empty title="No accounts yet">
              Approved accounts land here, with what each has spent this month and the cap they
              are spending against.
            </Empty>
          ) : (
            <div className="table-scroll">
              <table className="table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Status</th>
                    <th className="num">Spent</th>
                    <th className="num">Allowance</th>
                    <th>
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rest.map((u) => {
                    const s = spentBy(u.id);
                    return (
                      <tr key={u.id}>
                        <td>
                          {u.name}
                          {u.is_admin && <span className="caption"> admin</span>}
                        </td>
                        <td className="caption">{u.email}</td>
                        <td>
                          <span
                            className={`status ${u.status === "active" ? "completed" : "failed"}`}
                          >
                            {u.status}
                          </span>
                        </td>
                        <td className="num">${(s?.spent_usd ?? 0).toFixed(2)}</td>
                        <td className="num">
                          <input
                            type="number"
                            step="0.5"
                            min="0"
                            aria-label={`Monthly allowance for ${u.name}, in dollars`}
                            value={draftFor(u)}
                            placeholder="office default"
                            onChange={(e) => setDraft(u.id, e.target.value)}
                            onBlur={() => commitBudget(u)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") e.currentTarget.blur();
                            }}
                          />
                        </td>
                        <td>
                          <div className="actions">
                            {u.status === "active" ? (
                              <button
                                type="button"
                                className="btn btn-sm btn-danger"
                                disabled={u.id === me.id || working}
                                title={
                                  u.id === me.id
                                    ? "Suspending yourself would lock you out."
                                    : undefined
                                }
                                onClick={() =>
                                  act(
                                    () => api.suspend(u.id),
                                    `${u.name} suspended and signed out everywhere.`,
                                  )
                                }
                              >
                                Suspend
                              </button>
                            ) : (
                              <button
                                type="button"
                                className="btn btn-sm"
                                disabled={working}
                                onClick={() =>
                                  act(() => api.reinstate(u.id), `${u.name} reinstated.`)
                                }
                              >
                                Reinstate
                              </button>
                            )}
                            {u.is_admin ? (
                              <button
                                type="button"
                                className="btn btn-sm"
                                disabled={working}
                                onClick={() =>
                                  act(
                                    () => api.revokeAdmin(u.id),
                                    `${u.name} is no longer an admin.`,
                                  )
                                }
                              >
                                Drop admin
                              </button>
                            ) : (
                              <button
                                type="button"
                                className="btn btn-sm"
                                disabled={working}
                                onClick={() =>
                                  act(() => api.grantAdmin(u.id), `${u.name} is now an admin.`)
                                }
                              >
                                Make admin
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <p className="caption">
            A blank allowance means the office default. Zero means that person cannot render at
            all. Suspending signs them out immediately rather than when their session expires.
          </p>
        </div>
      </section>
    </div>
  );
}
