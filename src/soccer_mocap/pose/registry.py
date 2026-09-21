"""Backend registry.

Backends register themselves by name here, and everything else - the CLI, the
config file, the demo app dropdown - refers to them by that name only. Adding
a system to the project is therefore a two-line change plus the backend
itself.

Registration is lazy: the module records a factory, not an instance, so
listing what exists never imports torch.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import BackendUnavailableError
from .base import BackendInfo, PoseBackend

_REGISTRY: dict[str, type[PoseBackend]] = {}


def register(backend_cls: type[PoseBackend]) -> type[PoseBackend]:
    """Register a backend class. Usable as a decorator."""
    key = backend_cls.info.key
    if key in _REGISTRY and _REGISTRY[key] is not backend_cls:
        raise ValueError(f"Backend key {key!r} is already registered.")
    _REGISTRY[key] = backend_cls
    return backend_cls


def registered() -> dict[str, type[PoseBackend]]:
    """Every registered backend class, keyed by name."""
    _load_builtins()
    return dict(_REGISTRY)


def describe() -> list[BackendInfo]:
    """Metadata for every registered backend, implemented or not."""
    return [cls.info for cls in registered().values()]


def availability() -> dict[str, tuple[bool, str]]:
    """``{key: (usable, reason)}`` for every registered backend."""
    return {key: cls.available() for key, cls in registered().items()}


def get_backend(key: str, **options: Any) -> PoseBackend:
    """Construct a backend by name.

    Raises:
        BackendUnavailableError: if the name is unknown, or its dependencies
            are missing. The message names what to install.
    """
    backends = registered()
    if key not in backends:
        raise BackendUnavailableError(
            key,
            "no backend is registered under that name",
            f"Choose one of: {', '.join(sorted(backends))}",
        )

    backend_cls = backends[key]
    usable, reason = backend_cls.available()
    if not usable:
        raise BackendUnavailableError(key, reason, backend_cls.info.install_hint or None)
    return backend_cls(**options)


def default_backend_key() -> str:
    """The first usable backend, preferring the lightest.

    Used when a config says ``backend: auto``. The order encodes a judgement:
    for a demo, something that runs beats something that is accurate.
    """
    preference = ("mediapipe", "rtmpose", "sapiens", "posepipeline")
    checks = availability()
    for key in preference:
        if checks.get(key, (False, ""))[0]:
            return key
    usable = [key for key, (ok, _) in checks.items() if ok]
    if usable:
        return usable[0]
    raise BackendUnavailableError(
        "auto",
        "no pose backend is installed",
        "mamba env create -f environment.yml && mamba activate soccer-mocap",
    )


_BUILTINS_LOADED = False


def _load_builtins() -> None:
    """Import the bundled backend modules so their registrations run."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from . import backends  # noqa: F401  (import triggers registration)


# Convenience re-export so callers can write `registry.BackendInfo`.
__all__ = [
    "BackendInfo",
    "availability",
    "default_backend_key",
    "describe",
    "get_backend",
    "register",
    "registered",
]


def make_stub(
    key: str,
    label: str,
    layout_name: str,
    summary: str,
    speed: str,
    multi_person: bool,
    needs_gpu: bool,
    install_hint: str,
    homepage: str,
    extras: dict[str, Any] | None = None,
) -> type[PoseBackend]:
    """Build and register a placeholder backend.

    Saves each planned integration from repeating the same class body while
    it is still unimplemented.
    """
    from ..skeleton import get_layout
    from .base import StubBackend

    stub_cls = type(
        f"{key.title().replace('_', '')}Backend",
        (StubBackend,),
        {
            "info": BackendInfo(
                key=key,
                label=label,
                layout=get_layout(layout_name),
                summary=summary,
                speed=speed,
                multi_person=multi_person,
                needs_gpu=needs_gpu,
                install_hint=install_hint,
                homepage=homepage,
                implemented=False,
                extras=extras or {},
            )
        },
    )
    return register(stub_cls)
