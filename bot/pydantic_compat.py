"""Compatibility helpers for working with multiple Pydantic versions.

This module normalises the parts of the Pydantic API that changed between
version 1 and 2 so the rest of the codebase can rely on ``model_validate`` and
``model_dump`` regardless of the installed dependency.  When Pydantic is not
available the lightweight stubs used in tests are returned instead.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, cast

from services.stubs import create_pydantic_stub

BaseModel: type[Any]
ValidationError: type[Exception]
ConfigDict: Any
Field: Callable[..., Any]

try:  # pragma: no cover - simply exercising the import branch
    from pydantic import BaseModel as _PydanticBaseModel, ValidationError as _PydanticValidationError
    from pydantic import Field as _pydantic_field
except Exception:  # pragma: no cover - executed when pydantic is unavailable
    BaseModel, ConfigDict, ValidationError = create_pydantic_stub()

    def Field(default: Any = None, **_: Any) -> Any:
        return default
else:
    BaseModel = _PydanticBaseModel
    ValidationError = _PydanticValidationError
    Field = cast(Callable[..., Any], _pydantic_field)

    _config_dict = getattr(import_module("pydantic"), "ConfigDict", None)
    if _config_dict is None:
        def _fallback_config_dict(**kwargs: Any) -> dict[str, Any]:
            return dict(**kwargs)

        ConfigDict = _fallback_config_dict
    else:
        ConfigDict = cast(Any, _config_dict)

    def _ensure_method(
        name: str,
        factory: Callable[[type[_PydanticBaseModel]], object],
    ) -> None:
        if hasattr(_PydanticBaseModel, name):
            return
        setattr(_PydanticBaseModel, name, factory(_PydanticBaseModel))  # type: ignore[attr-defined]

    def _make_model_validate(_: type[_PydanticBaseModel]) -> object:
        def _model_validate(
            cls: type[_PydanticBaseModel], data: Any, *args: Any, **kwargs: Any
        ) -> _PydanticBaseModel:
            return cls.parse_obj(data)

        return classmethod(_model_validate)

    def _make_model_validate_json(_: type[_PydanticBaseModel]) -> object:
        def _model_validate_json(
            cls: type[_PydanticBaseModel], data: Any, *args: Any, **kwargs: Any
        ) -> _PydanticBaseModel:
            return cls.parse_raw(data, *args, **kwargs)

        return classmethod(_model_validate_json)

    def _make_model_dump(_: type[_PydanticBaseModel]) -> Callable[..., Any]:
        def _model_dump(self: _PydanticBaseModel, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return self.dict(*args, **kwargs)

        return _model_dump

    def _make_model_dump_json(_: type[_PydanticBaseModel]) -> Callable[..., Any]:
        def _model_dump_json(self: _PydanticBaseModel, *args: Any, **kwargs: Any) -> str:
            return self.json(*args, **kwargs)

        return _model_dump_json

    _ensure_method("model_validate", _make_model_validate)
    _ensure_method("model_validate_json", _make_model_validate_json)
    _ensure_method("model_dump", _make_model_dump)
    _ensure_method("model_dump_json", _make_model_dump_json)

__all__ = ["BaseModel", "ConfigDict", "ValidationError", "Field"]
