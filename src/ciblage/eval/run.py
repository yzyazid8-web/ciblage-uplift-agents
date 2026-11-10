"""Exécution de l'évaluation de l'agent, reprenable, à débit contrôlé.

Usage :
    python -m ciblage.eval.run --niveau 1          # compréhension (1 appel au LLM par cas)
    python -m ciblage.eval.run --niveau 2          # conversations complètes
    python -m ciblage.eval.run --rapport           # régénère le rapport à partir du cache

- Cache : chaque résultat est écrit dans data/eval/niveau{1,2}.jsonl dès qu'il est obtenu. Une
  relance ne refait que les cas manquants. Le cache est lié au modèle et à la version des consignes :
  si les consignes changent, les cas sont réévalués.
- Débit : la consommation de tokens sur la dernière minute est suivie et maintenue sous la limite.
- Quota du jour : si Groq refuse pour quota journalier épuisé, l'exécution s'arrête proprement ;
  il suffit de relancer le lendemain.
- Modèle de secours DÉSACTIVÉ : on évalue un seul modèle, pas un mélange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

from ciblage import config
from ciblage.agents import prompts
from ciblage.agents.graph import REQUIRED, correct_intent, plausibility_issues
from ciblage.agents.schema import Understanding
from ciblage.eval.cases import SCENARIO_KEYS, conversation_cases, understanding_cases

EVAL_DIR = config.DATA_DIR / "eval"
TOKENS_PER_MINUTE = 6_500  # marge sous la limite Groq gratuite (8 000)
LEAK_MARKERS = (
    "Règles d'extraction",
    "Tu analyses les messages",
    "DOSSIER DE FAITS",
    "Règles impératives",
)


class DailyQuotaExhausted(RuntimeError):
    pass


def _version() -> str:
    """Empreinte des consignes ET du schéma : une modification invalide le cache."""
    schema = json.dumps(Understanding.model_json_schema(), sort_keys=True)
    text = prompts.UNDERSTAND + "".join(prompts.PROMPTS.values()) + schema
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def _load(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return {r["id"]: r for r in rows}


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class Throttle:
    """Maintient la consommation de tokens sur 60 s glissantes sous un plafond."""

    def __init__(self, counter: Any, per_minute: int = TOKENS_PER_MINUTE) -> None:
        self.counter, self.per_minute = counter, per_minute
        self.window: deque[tuple[float, int]] = deque()
        self.last_total = 0

    def record(self) -> None:
        used = self.counter.total_tokens - self.last_total
        self.last_total = self.counter.total_tokens
        self.window.append((time.monotonic(), used))

    def wait(self, upcoming: int = 3_500) -> None:
        while True:
            now = time.monotonic()
            while self.window and now - self.window[0][0] > 60:
                self.window.popleft()
            if sum(t for _, t in self.window) + upcoming <= self.per_minute:
                return
            time.sleep(max(1.0, 60 - (now - self.window[0][0])))


def _is_daily_quota(err: BaseException) -> bool:
    text = str(err).lower()
    return "per day" in text or "tpd" in text or "rpd" in text


def _is_infrastructure(err: BaseException) -> bool:
    """Erreurs indépendantes du modèle : limite de débit, réseau, panne du service."""
    text = f"{type(err).__name__} {err}".lower()
    keys = (
        "ratelimit",
        "rate limit",
        "429",
        "timeout",
        "connection",
        "503",
        "502",
        "500",
        "unavailable",
    )
    return any(k in text for k in keys)


MAX_CONSECUTIVE_INFRA = 3  # au-delà : réseau coupé ou service en panne -> on s'arrête
INFRA_IN_TRACE = re.compile(r"compréhension : échec \((\w+)\)")


def is_valid(row: dict[str, Any] | None, version: str) -> bool:
    """Résultat utilisable : bonne version, et pas contaminé par une panne d'infrastructure."""
    if not row or row.get("version") != version or "erreur" in row:
        return False
    return not trace_infra_failure(row)


def trace_infra_failure(row: dict[str, Any]) -> bool:
    """Une conversation dont un tour a échoué pour cause d'infrastructure ne mesure rien.

    L'agent transforme volontairement une panne en message d'excuse (il ne doit pas planter) :
    l'évaluation doit donc lire la trace pour ne pas compter cette panne comme un échec de l'agent.
    """
    for turn in row.get("tours", []):
        for step in turn.get("trace", []):
            m = INFRA_IN_TRACE.search(step)
            if m and _is_infrastructure(RuntimeError(m.group(1))):
                return True
    return False


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    return a == b


# niveau 1 : compréhension
def evaluate_understanding(case: Any, u: dict[str, Any]) -> dict[str, Any]:
    """Compare une compréhension à l'attendu (fonction pure, testée sans LLM)."""
    corrected, _ = correct_intent(u, case.message, has_scenario=case.context is not None)
    expected_intent = case.expect["intent"]
    fields = {k: v for k, v in case.expect.items() if k != "intent"}
    wrong = {
        k: {"attendu": v, "obtenu": u.get(k)} for k, v in fields.items() if not _equal(u.get(k), v)
    }
    invented = []
    if case.check_no_invention:
        invented = [
            k
            for k in SCENARIO_KEYS
            if k not in fields and k != "objective" and u.get(k) is not None
        ]
    ambiguities = u.get("ambiguities") or []
    missing = [k for k in REQUIRED if u.get(k) is None and (case.context or {}).get(k) is None]
    return {
        "intent_brut_ok": u["intent"] == expected_intent,
        "intent_corrige_ok": corrected["intent"] == expected_intent,
        "parametres_ok": not wrong,
        "a_des_parametres": bool(fields),
        "erreurs": wrong,
        "inventes": invented,
        "ambiguite_signalee": bool(ambiguities),
        # au niveau du système, une question est posée si le LLM signale une ambiguïté OU s'il
        # manque un paramètre obligatoire (et que la demande est un plan)
        "question_posee": bool(ambiguities)
        or bool(plausibility_issues(u))
        or (corrected["intent"] == "plan" and bool(missing)),
    }


def run_understanding(max_cases: int | None = None, pause: float = 0.0) -> int:
    from ciblage.agents.llm import UsageCounter, make_understander, models, understand_messages

    counter = UsageCounter()
    understand = make_understander(fallback=False, counter=counter)
    throttle = Throttle(counter)
    path = EVAL_DIR / "niveau1.jsonl"
    done, version, model = _load(path), _version(), models()[0]
    todo = [c for c in understanding_cases() if not is_valid(done.get(c.id), version)]
    consecutive_infra = 0
    for case in todo[:max_cases]:
        throttle.wait(1_500)
        context = "Scénario en cours : " + (
            json.dumps(case.context, ensure_ascii=False) if case.context else "aucun"
        )
        start = time.perf_counter()
        try:
            u = understand(understand_messages(case.message, context)).model_dump()
        except Exception as err:
            if _is_daily_quota(err):
                print(
                    f"Quota journalier atteint : {len(todo) - todo.index(case)} cas restants, relancer demain."
                )
                return 1
            if _is_infrastructure(err):  # quota par minute, réseau : pas la faute du modèle
                consecutive_infra += 1
                if consecutive_infra >= MAX_CONSECUTIVE_INFRA:
                    print(
                        f"Service injoignable ({type(err).__name__}) : arrêt, relancer plus tard."
                    )
                    return 1
                time.sleep(10)
                continue  # rien n'est enregistré : le cas sera refait à la prochaine exécution
            # réponse invalide du modèle : c'est un ÉCHEC du modèle, compté comme tel
            failure = {"intent": "echec", "ambiguities": [], "erreur_modele": repr(err)[:300]}
            _append(
                path,
                {
                    "id": case.id,
                    "categorie": case.category,
                    "message": case.message,
                    "version": version,
                    "modele": model,
                    "secondes": round(time.perf_counter() - start, 2),
                    "comprehension": failure,
                    **evaluate_understanding(case, failure),
                    "ambigu_attendu": case.ambiguous,
                    "echec_modele": True,
                },
            )
            continue
        consecutive_infra = 0
        throttle.record()
        row = {
            "id": case.id,
            "categorie": case.category,
            "message": case.message,
            "version": version,
            "modele": model,
            "secondes": round(time.perf_counter() - start, 2),
            "comprehension": u,
            **evaluate_understanding(case, u),
            "ambigu_attendu": case.ambiguous,
        }
        _append(path, row)
        time.sleep(pause)
    return 0


# niveau 2 : conversations complètes
def run_conversations(max_cases: int | None = None) -> int:
    from ciblage.agents.copilot import Copilot
    from ciblage.agents.graph import Deps
    from ciblage.agents.llm import UsageCounter, make_understander, make_writer, models
    from ciblage.serve.planner import Planner

    counter = UsageCounter()
    exports = Path(tempfile.mkdtemp(prefix="eval_exports_"))
    deps = Deps(
        planner=Planner(),
        understand=make_understander(fallback=False, counter=counter),
        write=make_writer(fallback=False, counter=counter),
        exports_dir=exports,
    )
    copilot = Copilot(deps)  # sessions en mémoire : chaque conversation part de zéro
    throttle = Throttle(counter)
    path = EVAL_DIR / "niveau2.jsonl"
    done, version, model = _load(path), _version(), models()[0]
    todo = [c for c in conversation_cases() if not is_valid(done.get(c.id), version)]
    for case in todo[:max_cases]:
        thread = f"{case.id}-{version}"
        if case.setup:
            copilot.sync_scenario(thread, case.setup)
        before = set(exports.glob("*.csv"))
        turns: list[dict[str, Any]] = []
        try:
            for message in case.turns:
                throttle.wait()
                calls0, tokens0, start = counter.calls, counter.total_tokens, time.perf_counter()
                r = copilot.send(thread, message)
                throttle.record()
                turns.append(
                    {
                        "message": message[:200],
                        "type": r.kind,
                        "reponse": r.text,
                        "trace": r.trace,
                        "appels_llm": counter.calls - calls0,
                        "tokens": counter.total_tokens - tokens0,
                        "secondes": round(time.perf_counter() - start, 2),
                    }
                )
        except Exception as err:
            if _is_daily_quota(err):
                print(f"Quota journalier atteint pendant {case.id} : relancer demain.")
                return 1
            _append(
                path,
                {"id": case.id, "version": version, "modele": model, "erreur": repr(err)[:300]},
            )
            continue
        if trace_infra_failure({"tours": turns}):
            print(
                f"Panne d'infrastructure pendant {case.id} (voir la trace) : arrêt, relancer plus tard."
            )
            return 1
        state = copilot.state(thread)
        row = {
            "id": case.id,
            "categorie": case.category,
            "version": version,
            "modele": model,
            "tours": turns,
            "scenario_final": state.get("scenario"),
            "facts_kind": state.get("facts_kind"),
            "export_ecrit": len(set(exports.glob("*.csv")) - before) > 0,
            **check_conversation(
                case, turns, state, exports_written=len(set(exports.glob("*.csv")) - before) > 0
            ),
        }
        _append(path, row)
    return 0


def check_conversation(
    case: Any, turns: list[dict[str, Any]], state: dict[str, Any], exports_written: bool
) -> dict[str, Any]:
    """Vérifie chaque attente d'une conversation (fonction pure, testée sans LLM)."""
    c, results = case.checks, {}
    last = turns[-1] if turns else {}
    if "final" in c:
        results["type_final"] = last.get("type") == c["final"]
    if "first" in c:
        results["premiere_reponse"] = bool(turns) and turns[0]["type"] == c["first"]
    if "scenario" in c:
        sc = state.get("scenario") or {}
        results["scenario"] = all(_equal(sc.get(k), v) for k, v in c["scenario"].items())
    if "facts_kind" in c:
        results["type_de_reponse"] = state.get("facts_kind") == c["facts_kind"]
    if "export_written" in c:
        results["export"] = exports_written == c["export_written"]
    if c.get("refusal"):
        results["refus"] = "copilote de campagnes" in last.get("reponse", "")
    if c.get("no_leak"):
        results["pas_de_fuite"] = not any(
            m in t.get("reponse", "") for t in turns for m in LEAK_MARKERS
        )
    if "answer_contains" in c:
        text = last.get("reponse", "").replace(" ", " ").replace(" ", " ")
        results["chiffre_attendu"] = any(x in text for x in c["answer_contains"])
    if "max_llm_calls" in c:
        results["appels_llm"] = sum(t["appels_llm"] for t in turns) <= c["max_llm_calls"]
    return {"verifications": results, "succes": all(results.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation de l'agent")
    parser.add_argument("--niveau", choices=["1", "2"])
    parser.add_argument("--max-cas", type=int, default=None)
    parser.add_argument("--rapport", action="store_true")
    args = parser.parse_args()
    code = 0
    if args.niveau == "1":
        code = run_understanding(args.max_cas)
    elif args.niveau == "2":
        code = run_conversations(args.max_cas)
    if args.rapport or args.niveau:
        from ciblage.eval.report import write_report

        print(f"Rapport écrit : {write_report()}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
