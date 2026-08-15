#!/usr/bin/env python3
"""Generate version-pinned API digests from the installed packages.

The pfit skills (/pfit-new, /pfit-check, /pfit-jax) generate JAX +
diffrax + optax code. Relying on the model's recollection of those APIs is the
main source of subtle generation bugs, because the installed versions are
pinned to an old, mutually-compatible set (see requirements.txt) whose API
differs from the current upstream docs.

This script introspects the *installed* packages and writes compact markdown
digests that the skills read before generating or validating code. Everything
in the digests is derived from the live objects, so it can never drift from
what the code will actually run against.

Run (from the repo root):

    ./venv/bin/python3 tools/gen_api_context.py

Files are only rewritten when their content changes, so re-running on an
unchanged environment is a no-op and leaves the git tree clean.
"""

import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "gen_api_context.yaml"

logger = logging.getLogger("gen_api_context")


def setup_logging(log_file: Path) -> None:
    """Attach a stdout handler and a file handler, both at DEBUG level."""
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    # pyswarms attaches a handler to the root logger on import; without this
    # every record would be emitted twice.
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_versions(package_names: list[str]) -> dict[str, str]:
    """Return {package: version} for every importable tracked package."""
    versions = {}
    for name in package_names:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
            logger.debug("%s == %s", name, versions[name])
        except ImportError:
            logger.warning("%s is not installed; omitted from the manifest", name)
    return versions


def short_signature(obj, max_len: int = 400) -> str:
    """Render a callable's signature, stripped of noisy type annotations.

    jaxtyping annotations expand to multi-line unions that are useless in a
    digest, so parameters are reduced to `name=default`.
    """
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError) as exc:
        logger.debug("no signature for %r (%s)", obj, exc)
        return "(signature unavailable)"

    parts = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{name}")
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{name}")
        elif param.default is inspect.Parameter.empty:
            parts.append(name)
        else:
            default = param.default
            rendered = getattr(default, "__name__", None) or repr(default)
            if len(rendered) > 30:
                rendered = "<default>"
            parts.append(f"{name}={rendered}")
    text = "(" + ", ".join(parts) + ")"
    return text if len(text) <= max_len else text[: max_len - 4] + " ...)"


def first_doc_line(obj) -> str:
    """First non-empty line of a docstring, or an empty string."""
    doc = inspect.getdoc(obj) or ""
    for line in doc.split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def header(title: str, versions: dict[str, str], relevant: list[str]) -> list[str]:
    """Build the shared digest header, including the pinned version stamp."""
    stamp = ", ".join(f"{k}=={versions[k]}" for k in relevant if k in versions)
    # Deliberately no generation timestamp: the digests must be a pure function
    # of the installed environment so re-running the generator is a true no-op
    # (see tests/test_api_digest.py::test_generator_is_idempotent). The version
    # stamp, not a date, is what tells you whether the digest is stale.
    return [
        f"# {title}",
        "",
        f"**Pinned to: {stamp}**",
        "",
        "Auto-generated from the installed packages by "
        "`tools/gen_api_context.py`. Do not edit by hand — edit "
        "`tools/gen_api_context.yaml` and regenerate.",
        "",
        "If the versions above differ from the installed ones, this digest is "
        "stale: regenerate it before trusting it.",
        "",
    ]


def render_notes(notes: list[str]) -> list[str]:
    """Render the curated gotcha notes from the YAML config."""
    lines = ["## Gotchas (curated — these are the ones that bite)", ""]
    for note in notes:
        collapsed = " ".join(note.split())
        lines.append(f"- {collapsed}")
    lines.append("")
    return lines


def write_if_changed(path: Path, lines: list[str]) -> bool:
    """Write the digest only when its content differs. Returns True if written."""
    content = "\n".join(lines).rstrip() + "\n"
    if path.exists() and path.read_text() == content:
        logger.info("unchanged: %s", path.relative_to(REPO_ROOT))
        return False
    path.write_text(content)
    logger.info("wrote: %s (%d lines)", path.relative_to(REPO_ROOT), len(lines))
    return True


# ---------------------------------------------------------------------------
# diffrax
# ---------------------------------------------------------------------------

def classify_solver(cls, diffrax_mod) -> tuple[str, bool]:
    """Return (kind, is_adaptive) for a solver class.

    Order matters: many explicit RK solvers (Heun, Midpoint, Ralston) ALSO
    subclass an SDE base, so the deterministic-ODE classification must be
    tested first or they get mislabelled as SDE-only.

    is_adaptive reports whether the solver provides an error estimate. A
    non-adaptive solver cannot be driven by `PIDController`, which the
    framework's `_integrate_system` always uses.
    """
    bases = {b.__name__ for b in cls.__mro__}
    adaptive = issubclass(cls, diffrax_mod.AbstractAdaptiveSolver)

    if "AbstractWrappedSolver" in bases:
        kind = "wrapper"
    elif "AbstractImplicitSolver" in bases:
        kind = "implicit (stiff)"
    elif "AbstractERK" in bases or "AbstractRungeKutta" in bases:
        kind = "explicit RK (non-stiff)"
    elif {"AbstractSRK", "AbstractStratonovichSolver", "AbstractItoSolver"} & bases:
        kind = "SDE only"
    else:
        kind = "other"

    # Dual-purpose solvers are usable for ODEs but are advertised for SDEs too.
    if kind.startswith("explicit") and (
        {"AbstractStratonovichSolver", "AbstractItoSolver"} & bases
    ):
        kind += ", SDE-capable"
    return kind, adaptive


def build_diffrax(cfg: dict, versions: dict[str, str]) -> list[str]:
    import diffrax

    conf = cfg["diffrax"]
    validated = set(conf["validated_solvers"])
    lines = header(
        "diffrax API digest", versions, ["diffrax", "jax", "equinox", "optimistix"]
    )

    lines += [
        "## ODE solvers",
        "",
        "The `INTEGRATOR` field in `GRADIENT_OPT/SETTINGS` must name one of these "
        "classes exactly. It is substituted for `diffrax.SOLVER_CLASS()` in "
        "`_integrate_system`. Anything not in this table does not exist in this "
        "version and will raise `AttributeError` at import time.",
        "",
        "`validated` marks solvers already exercised by this framework's examples; "
        "the others exist and are usable but are untested here.",
        "",
        "Only **adaptive** solvers are listed. `_integrate_system` always drives "
        "the solve with `diffrax.PIDController(rtol, atol)`, which requires an "
        "error estimate, so non-adaptive solvers are unusable and are omitted "
        "rather than listed as options.",
        "",
        "When in doubt, pick an **implicit (stiff)** solver: on a non-stiff system "
        "it merely costs more per step, whereas an explicit solver on a stiff "
        "system fails outright. See the gotchas below.",
        "",
        "| Solver | Kind | Validated | Summary |",
        "|---|---|---|---|",
    ]

    solvers = []
    excluded = []
    for name in sorted(dir(diffrax)):
        obj = getattr(diffrax, name)
        if not inspect.isclass(obj) or not issubclass(obj, diffrax.AbstractSolver):
            continue
        if conf.get("exclude_abstract", True) and name.startswith("Abstract"):
            continue
        if inspect.isabstract(obj):
            continue
        kind, adaptive = classify_solver(obj, diffrax)

        # Filter to solvers that are actually usable as an INTEGRATOR: adaptive
        # (PIDController needs an error estimate), deterministic (the term is an
        # ODETerm), and constructible as a bare `diffrax.<Name>()`.
        if conf.get("adaptive_only", True) and not adaptive:
            excluded.append((name, "not adaptive"))
            continue
        if conf.get("exclude_sde_only", True) and kind == "SDE only":
            excluded.append((name, "SDE only"))
            continue
        if conf.get("require_zero_arg_construction", True):
            try:
                obj()
            except Exception:
                excluded.append((name, "needs constructor arguments"))
                continue
        solvers.append((name, obj))

    for name, obj in solvers:
        kind, _ = classify_solver(obj, diffrax)
        mark = "yes" if name in validated else ""
        lines.append(f"| `{name}` | {kind} | {mark} | {first_doc_line(obj)} |")
    logger.info("diffrax: documented %d usable solvers (%d excluded)",
                len(solvers), len(excluded))
    for name, reason in excluded:
        logger.debug("excluded %s (%s)", name, reason)

    if excluded:
        by_reason: dict[str, list[str]] = {}
        for name, reason in excluded:
            by_reason.setdefault(reason, []).append(name)
        lines += [
            "",
            f"**Excluded as unusable with this framework** ({len(excluded)} "
            "solvers). These exist in diffrax but must never be named as an "
            "`INTEGRATOR`:",
            "",
        ]
        for reason, names in sorted(by_reason.items()):
            lines.append(f"- _{reason}_: " + ", ".join(f"`{n}`" for n in sorted(names)))

    missing = validated - {n for n, _ in solvers}
    if missing:
        logger.error(
            "validated_solvers names not present in diffrax %s: %s",
            versions.get("diffrax"), sorted(missing),
        )

    # RESULTS — the failure codes the loss function must mask on.
    lines += ["", "## `RESULTS` codes", ""]
    try:
        name_to_item = diffrax.RESULTS._name_to_item
        index_to_message = diffrax.RESULTS._index_to_message
        lines += [
            "`_integrate_system` returns `sol.result`. Every code below other than "
            "`successful` means the trajectory is NOT trustworthy. The loss "
            "function must map failed solves to `constants['error_loss']`.",
            "",
            "| Code | Meaning |",
            "|---|---|",
        ]
        for member_name, item in name_to_item.items():
            message = index_to_message[item._value] if hasattr(item, "_value") else ""
            message = " ".join(message.split())
            if member_name == "successful":
                message = "Solve completed normally. The ONLY code that means the "\
                          "trajectory is usable."
            if len(message) > 200:
                message = message[:197].rstrip() + "..."
            lines.append(f"| `RESULTS.{member_name}` | {message} |")
        logger.info("diffrax: documented %d RESULTS codes", len(name_to_item))
    except Exception as exc:
        logger.error(
            "could not enumerate RESULTS (equinox internals changed?): %s", exc
        )
        lines.append("_RESULTS enumeration unavailable — inspect `diffrax.RESULTS`._")

    # diffeqsolve / SaveAt / step size controllers / adjoints
    lines += ["", "## Core call signatures", ""]
    for label, obj in [
        ("diffrax.diffeqsolve", diffrax.diffeqsolve),
        ("diffrax.SaveAt", diffrax.SaveAt),
        ("diffrax.ODETerm", diffrax.ODETerm),
    ]:
        lines += [f"```python", f"{label}{short_signature(obj)}", "```", ""]

    lines += ["## Step size controllers", "", "| Class | Signature |", "|---|---|"]
    for name in sorted(dir(diffrax)):
        obj = getattr(diffrax, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, diffrax.AbstractStepSizeController)
            and not name.startswith("Abstract")
        ):
            sig = short_signature(obj.__init__, max_len=220).replace("|", "\\|")
            lines.append(f"| `{name}` | `{sig}` |")

    lines += ["", "## Adjoints (gradient propagation through the solve)", "",
              "| Class | Summary |", "|---|---|"]
    for name in sorted(dir(diffrax)):
        obj = getattr(diffrax, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, diffrax.AbstractAdjoint)
            and not name.startswith("Abstract")
        ):
            lines.append(f"| `{name}` | {first_doc_line(obj)} |")
    lines += [
        "",
        "`diffeqsolve` defaults to `RecursiveCheckpointAdjoint` when `adjoint` is "
        "not passed; the framework does not override it.",
        "",
    ]

    lines += render_notes(conf["notes"])
    return lines


# ---------------------------------------------------------------------------
# optax
# ---------------------------------------------------------------------------

def build_optax(cfg: dict, versions: dict[str, str]) -> list[str]:
    import optax

    conf = cfg["optax"]
    lines = header("optax API digest (gradient stage)", versions, ["optax", "jax"])

    for section, key in [
        ("Optimizers", "optimizers"),
        ("Learning-rate schedules", "schedules"),
        ("Utilities", "utilities"),
    ]:
        lines += [f"## {section}", "", "| Name | Signature (defaults shown) |", "|---|---|"]
        for name in conf.get(key, []):
            obj = getattr(optax, name, None)
            if obj is None:
                logger.error("optax.%s does not exist in optax %s — check the config",
                             name, versions.get("optax"))
                continue
            sig = short_signature(obj, max_len=300).replace("|", "\\|")
            lines.append(f"| `optax.{name}` | `{sig}` |")
            logger.debug("optax.%s %s", name, sig)
        lines.append("")

    lines += render_notes(conf["notes"])
    return lines


# ---------------------------------------------------------------------------
# jax
# ---------------------------------------------------------------------------

def build_jax(cfg: dict, versions: dict[str, str]) -> list[str]:
    import jax
    import jax.numpy as jnp

    conf = cfg["jax"]
    lines = header(
        "jax / jax.numpy API digest (translation rules)", versions, ["jax", "jaxlib", "numpy"]
    )

    lines += [
        "## Transforms",
        "",
        "| Name | Signature |",
        "|---|---|",
    ]
    for name in conf.get("transforms", []):
        obj = getattr(jax, name, None)
        if obj is None:
            logger.error("jax.%s not found", name)
            continue
        sig = short_signature(obj, max_len=260).replace("|", "\\|")
        lines.append(f"| `jax.{name}` | `{sig}` |")
    lines.append("")

    lines += [
        "## `jax.numpy` functions available for translation",
        "",
        "The generated code translates numpy pseudocode to these. A numpy "
        "function absent from this list should be checked against the installed "
        "`jax.numpy` before use — several numpy APIs have no jnp equivalent.",
        "",
    ]
    for group, names in conf.get("functions", {}).items():
        lines += [f"**{group.replace('_', ' ')}**", "", "| Name | Signature |", "|---|---|"]
        for name in names:
            obj = getattr(jnp, name, None)
            if obj is None:
                logger.error("jnp.%s not found in jax %s", name, versions.get("jax"))
                continue
            sig = short_signature(obj, max_len=200).replace("|", "\\|")
            lines.append(f"| `jnp.{name}` | `{sig}` |")
        lines.append("")

    lines += render_notes(conf["notes"])
    return lines


# ---------------------------------------------------------------------------
# population optimizers (scipy DE + pyswarms)
# ---------------------------------------------------------------------------

def build_population(cfg: dict, versions: dict[str, str]) -> list[str]:
    import scipy.optimize
    import pyswarms

    conf = cfg["population"]
    lines = header(
        "Population (zero-order) optimizer digest", versions, ["scipy", "pyswarms"]
    )

    lines += [
        "## `scipy.optimize.differential_evolution`",
        "",
        "Driven by `lib/algorithms/DE/classes.py`. Only the arguments the "
        "framework sets are exposed through user_input.yaml today; the rest are scipy "
        "defaults shown here.",
        "",
        "```python",
        f"differential_evolution{short_signature(scipy.optimize.differential_evolution, max_len=900)}",
        "```",
        "",
    ]

    try:
        solver_cls = scipy.optimize._differentialevolution.DifferentialEvolutionSolver
        available = sorted(set(solver_cls._binomial) | set(solver_cls._exponential))
        lines += [
            "Valid `strategy` values in this scipy version:",
            "",
            "`" + "`, `".join(available) + "`",
            "",
        ]
        logger.info("scipy DE: %d strategies documented", len(available))
    except Exception as exc:
        logger.warning("could not enumerate DE strategies: %s", exc)

    lines += ["## pyswarms handler strategies", ""]
    try:
        from pyswarms.backend.handlers import (
            BoundaryHandler, VelocityHandler, OptionsHandler,
        )
        lines += ["| Handler | Valid strategies |", "|---|---|"]
        for cls in (BoundaryHandler, VelocityHandler, OptionsHandler):
            inst = cls(strategy=None)
            names = sorted(inst.strategies.keys())
            lines.append(f"| `{cls.__name__}` | `" + "`, `".join(names) + "` |")
            logger.debug("%s strategies: %s", cls.__name__, names)
        lines.append("")
    except Exception as exc:
        logger.warning("could not enumerate pyswarms strategies: %s", exc)

    lines += render_notes(conf["notes"])
    return lines


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def build_manifest(versions: dict[str, str], digests: list[str]) -> list[str]:
    lines = [
        "# API digest manifest",
        "",
        "Generated by `tools/gen_api_context.py` from the packages installed in "
        "`./venv`.",
        "",
        "## Pinned versions",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for name, version in versions.items():
        lines.append(f"| {name} | {version} |")
    lines += [
        "",
        "## Digests",
        "",
    ]
    for name in digests:
        lines.append(f"- [{name}]({name})")
    lines += [
        "",
        "## Staleness",
        "",
        "These digests describe the packages installed in `./venv`. If the "
        "versions above no longer match, regenerate with:",
        "",
        "```bash",
        "./venv/bin/python3 tools/gen_api_context.py",
        "```",
        "",
        "The generator only rewrites files whose content changed, so running it "
        "on an unchanged environment leaves the git tree clean.",
        "",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to the YAML config (default: alongside this script)")
    parser.add_argument("--log-file", type=Path,
                        default=Path(__file__).with_suffix(".log"),
                        help="Log file path (default: gen_api_context.log)")
    args = parser.parse_args()

    setup_logging(args.log_file)
    logger.info("reading config: %s", args.config)
    cfg = yaml.safe_load(args.config.read_text())

    output_dir = REPO_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output directory: %s", output_dir.relative_to(REPO_ROOT))

    versions = get_versions(cfg["tracked_packages"])

    builders = {
        "diffrax.md": build_diffrax,
        "optax.md": build_optax,
        "jax.md": build_jax,
        "population_optimizers.md": build_population,
    }

    written = 0
    produced = []
    for filename, builder in builders.items():
        try:
            lines = builder(cfg, versions)
        except Exception as exc:
            logger.error("failed to build %s: %s", filename, exc, exc_info=True)
            continue
        written += write_if_changed(output_dir / filename, lines)
        produced.append(filename)

    written += write_if_changed(output_dir / "MANIFEST.md",
                                build_manifest(versions, produced))

    logger.info("done: %d digest(s) produced, %d file(s) updated",
                len(produced), written)
    if len(produced) != len(builders):
        logger.error("some digests failed to build; see errors above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
