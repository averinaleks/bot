"""Compatibility helpers for working with multiple Pydantic versions.

This module normalises the parts of the Pydantic API that changed between
version 1 and 2 so the rest of the codebase can rely on ``model_validate`` and
``model_dump`` regardless of the installed dependency.  When Pydantic is not
available the lightweight stubs used in tests are returned instead.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, cast

from services.stubs import create_pydantic_stub

_runtime_base_model: type[Any]
_runtime_config_dict: Optional[Any] = None
_runtime_validation_error: type[Exception]
_runtime_field: Callable[..., Any]

try:  # pragma: no cover - simply exercising the import branch
    from pydantic import BaseModel as _ImportedBaseModel, ValidationError as _ImportedValidationError
    from pydantic import Field as _ImportedField
except Exception:  # pragma: no cover - executed when pydantic is unavailable
    _runtime_base_model, _runtime_config_dict, _runtime_validation_error = create_pydantic_stub()

    def _field_stub(default: Any = None, **_: Any) -> Any:
        return default

    _runtime_field = _field_stub
else:
    _runtime_base_model = _ImportedBaseModel
    _runtime_validation_error = _ImportedValidationError
    _runtime_field = cast(Callable[..., Any], _ImportedField)

    try:  # pragma: no cover - ConfigDict only exists in Pydantic v2
        from pydantic import ConfigDict as _ImportedConfigDict
    except ImportError:  # pragma: no cover - executed on Pydantic v1
        def _legacy_config_dict(**kwargs: Any) -> dict[str, Any]:
            return dict(**kwargs)

        _runtime_config_dict = _legacy_config_dict
    else:
        _runtime_config_dict = _ImportedConfigDict

    def _ensure_method(
        name: str,
        factory: Callable[[type[Any]], object],
    ) -> None:
        if hasattr(_runtime_base_model, name):
            return
        setattr(_runtime_base_model, name, factory(_runtime_base_model))  # type: ignore[attr-defined]

    def _make_model_validate(_: type[Any]) -> object:
        def _model_validate(
            cls: type[Any], data: Any, *args: Any, **kwargs: Any
        ) -> Any:
            return cls.parse_obj(data)

        return classmethod(_model_validate)

    def _make_model_validate_json(_: type[Any]) -> object:
        def _model_validate_json(
            cls: type[Any], data: Any, *args: Any, **kwargs: Any
        ) -> Any:
            return cls.parse_raw(data, *args, **kwargs)

        return classmethod(_model_validate_json)

    def _make_model_dump(_: type[Any]) -> object:
        def _model_dump(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return self.dict(*args, **kwargs)

        return _model_dump

    def _make_model_dump_json(_: type[Any]) -> object:
        def _model_dump_json(self: Any, *args: Any, **kwargs: Any) -> str:
            return self.json(*args, **kwargs)

        return _model_dump_json

    _ensure_method("model_validate", _make_model_validate)
    _ensure_method("model_validate_json", _make_model_validate_json)
    _ensure_method("model_dump", _make_model_dump)
    _ensure_method("model_dump_json", _make_model_dump_json)

BaseModel = _runtime_base_model
ConfigDict: Any = _runtime_config_dict
if ConfigDict is None:  # pragma: no cover - defensive safeguard
    def _fallback_config_dict(**kwargs: Any) -> dict[str, Any]:
        return dict(**kwargs)

    ConfigDict = _fallback_config_dict
ValidationError = _runtime_validation_error
Field = _runtime_field

__all__ = ["BaseModel", "ConfigDict", "ValidationError", "Field"]
