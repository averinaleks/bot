"""Compatibility helpers for working with multiple Pydantic versions.

This module normalises the parts of the Pydantic API that changed between
version 1 and 2 so the rest of the codebase can rely on ``model_validate`` and
``model_dump`` regardless of the installed dependency.  When Pydantic is not
available the lightweight stubs used in tests are returned instead.
"""
from __future__ import annotations

from typing import Any, Callable

from services.stubs import create_pydantic_stub

try:  # pragma: no cover - simply exercising the import branch
    from pydantic import BaseModel as _BaseModel, ValidationError as _ValidationError
    from pydantic import Field as _Field
except Exception:  # pragma: no cover - executed when pydantic is unavailable
    _BaseModel, _ConfigDict, _ValidationError = create_pydantic_stub()

    def _Field(default: Any = None, **_: Any) -> Any:
        return default
else:
    try:  # pragma: no cover - ConfigDict only exists in Pydantic v2
        from pydantic import ConfigDict as _ConfigDict
    except ImportError:  # pragma: no cover - executed on Pydantic v1
        def _ConfigDict(**kwargs: Any) -> dict[str, Any]:
            return dict(**kwargs)

    def _ensure_method(
        name: str,
        factory: Callable[[type[_BaseModel]], Callable[..., Any]],
    ) -> None:
        if hasattr(_BaseModel, name):
            return
        setattr(_BaseModel, name, factory(_BaseModel))  # type: ignore[attr-defined]

    def _make_model_validate(_: type[_BaseModel]) -> Callable[..., Any]:
        def _model_validate(cls: type[_BaseModel], data: Any, *args: Any, **kwargs: Any) -> _BaseModel:
            return cls.parse_obj(data)

        return classmethod(_model_validate)

    def _make_model_validate_json(_: type[_BaseModel]) -> Callable[..., Any]:
        def _model_validate_json(
            cls: type[_BaseModel], data: Any, *args: Any, **kwargs: Any
        ) -> _BaseModel:
            return cls.parse_raw(data, *args, **kwargs)

        return classmethod(_model_validate_json)

    def _make_model_dump(_: type[_BaseModel]) -> Callable[..., Any]:
        def _model_dump(self: _BaseModel, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return self.dict(*args, **kwargs)

        return _model_dump

    def _make_model_dump_json(_: type[_BaseModel]) -> Callable[..., Any]:
        def _model_dump_json(self: _BaseModel, *args: Any, **kwargs: Any) -> str:
            return self.json(*args, **kwargs)

        return _model_dump_json

    _ensure_method("model_validate", _make_model_validate)
    _ensure_method("model_validate_json", _make_model_validate_json)
    _ensure_method("model_dump", _make_model_dump)
    _ensure_method("model_dump_json", _make_model_dump_json)

BaseModel = _BaseModel
ConfigDict = locals().get("_ConfigDict")  # type: ignore[assignment]
if ConfigDict is None:  # pragma: no cover - defensive safeguard
    def _fallback_config_dict(**kwargs: Any) -> dict[str, Any]:
        return dict(**kwargs)

    ConfigDict = _fallback_config_dict
ValidationError = _ValidationError
Field = _Field

__all__ = ["BaseModel", "ConfigDict", "ValidationError", "Field"]
