"""Settings, loaded from environment. The fal.ai key never leaves this process.

Design constraint: FAL_KEY is server-side
only. It is never returned by an API route, never rendered into the UI, and never
written to a log line. `Settings.safe_dump()` exists so debug output can show the
config without leaking the key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets ------------------------------------------------------------
    # SecretStr so an accidental repr()/print() shows "**********" not the key.
    fal_key: SecretStr = Field(default=SecretStr(""))

    # The shared code that gates the signup form. Not an authentication factor
    # and never treated as one — it decides who may *ask* for an account, and
    # approval decides who may spend money.
    office_access_code: SecretStr = Field(default=SecretStr(""))

    # --- Accounts -----------------------------------------------------------
    # The first signup on this address is created active and admin. Without it
    # there is nobody to approve the first account, which is a deadlock worth
    # designing out rather than documenting a manual SQL fix for.
    admin_email: str = ""

    # New accounts wait for an administrator. Set to 0 only if the access code
    # alone is considered enough.
    signup_requires_approval: bool = True

    # Comma-separated email domains whose signups are active at once, for an
    # office that trusts its own domain. Matched exactly and case-insensitively:
    # "example.test" does not admit "sub.example.test" or "example.test.evil.net".
    # The admin bootstrap rule is checked first and is unaffected by this list.
    auto_approve_domains: str = ""

    # Argon2 handles slow hashing; this stops the trivially guessable.
    password_min_length: int = 10

    # Cookie `Secure` flag. Off on plain HTTP, because a Secure cookie is simply
    # never sent over http:// and nobody could sign in. Flip to 1 the day a
    # certificate exists — that is the only application change HTTPS needs.
    cookie_secure: bool = False
    session_days: int = 30

    # --- Budget guardrails --------------------------------------------------
    # fal.ai has NO native spend cap (verified). These are the only
    # ceilings that exist. The app enforces all of them before submitting.
    budget_monthly_usd: float = 50.0
    budget_per_user_usd: float = 25.0
    budget_per_job_usd: float = 1.00
    kill_switch: bool = False

    # --- Engine -------------------------------------------------------------
    # "fal" submits to fal.ai and spends money. "fake" routes every call through
    # app/fake_engine.py: no network, no key, no spend, same job lifecycle. Use
    # "fake" for development and end-to-end runs; never for a client deliverable.
    render_engine: Literal["fal", "fake"] = "fal"

    # How long the fake engine takes to finish one request, in seconds. Queued for
    # the first half, in progress until this, then completed. 0 finishes at once.
    fake_engine_delay_s: float = 4.0

    # A job still in flight this long after it was created is failed by the
    # poller, and its budget reservation is released. 480 s is eight minutes,
    # which is well past a normal render and short of the day-long pending rows
    # the office used to accumulate.
    job_timeout_s: int = 480

    # --- Paths --------------------------------------------------------------
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/render_studio.db")
    routing_path: Path = Path("./routing.yaml")

    @field_validator(
        "kill_switch", "cookie_secure", "signup_requires_approval", mode="before"
    )
    @classmethod
    def _parse_flag(cls, v: object) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return bool(v)

    @field_validator("admin_email", mode="before")
    @classmethod
    def _normalise_admin_email(cls, v: object) -> str:
        return str(v or "").strip().lower()

    def auto_approve_domain_set(self) -> frozenset[str]:
        """AUTO_APPROVE_DOMAINS as a set of lowercase domains, blanks dropped."""
        return frozenset(
            d.strip().lower().lstrip("@")
            for d in self.auto_approve_domains.split(",")
            if d.strip()
        )

    def require_key(self) -> str:
        """Return the key, or fail loudly. Called only at the fal.ai boundary."""
        key = self.fal_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "FAL_KEY is not set. Copy .env.example to .env and add your key "
                "from https://fal.ai/dashboard/keys — never commit that file."
            )
        return key

    def require_access_code(self) -> str:
        """The office access code, or fail loudly.

        An empty code would gate nothing, so the signup route refuses to operate
        rather than quietly letting the whole network create accounts.
        """
        code = self.office_access_code.get_secret_value()
        if not code:
            raise RuntimeError(
                "OFFICE_ACCESS_CODE is not set. Signup is disabled until it is — "
                "an empty code would let anyone on the network create an account."
            )
        return code

    def safe_dump(self) -> dict[str, object]:
        """Config minus the secrets. Use this for any logging or debug output."""
        d = self.model_dump()
        d["fal_key"] = "SET" if self.fal_key.get_secret_value() else "MISSING"
        d["office_access_code"] = (
            "SET" if self.office_access_code.get_secret_value() else "MISSING"
        )
        return d

    def inputs_dir(self) -> Path:
        return self.data_dir / "inputs"

    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    def skp_cache_dir(self) -> Path:
        """Where a parsed+baked SKP scene is cached, keyed by content hash.

        The parse+bake is the slow part of SKP conversion (minutes, not
        seconds) — everything after it is fast, so the cache is what makes a
        second conversion of the same file instant instead of another wait.
        """
        return self.data_dir / "skp_cache"

    def depth_model_dir(self) -> Path:
        """Where `scripts/fetch_depth_model.py` puts the ONNX depth model."""
        return self.data_dir / "models" / "depth-anything-v2-small"

    def depth_cache_dir(self) -> Path:
        """Estimated depth PNGs, named by source hash + model so a repeat upload is free."""
        return self.data_dir / "depth_cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.inputs_dir().mkdir(parents=True, exist_ok=True)
    s.outputs_dir().mkdir(parents=True, exist_ok=True)
    return s
