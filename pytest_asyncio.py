"""Minimal plugin to run async tests without pytest-asyncio."""

import asyncio
import types

def pytest_pyfunc_call(pyfuncitem):
    testfunction = pyfuncitem.obj
    if isinstance(testfunction, types.FunctionType) and asyncio.iscoroutinefunction(testfunction):
        asyncio.run(testfunction(**pyfuncitem.funcargs))
        return True
    return None
