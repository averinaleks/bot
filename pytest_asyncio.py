"""Minimal plugin to run async tests without pytest-asyncio."""

import asyncio
import types


def pytest_pyfunc_call(pyfuncitem):
    """Run coroutine test functions with their fixture arguments."""
    testfunction = pyfuncitem.obj
    if isinstance(testfunction, types.FunctionType) and asyncio.iscoroutinefunction(testfunction):
        # Only pass arguments the test function expects to avoid autouse fixtures.
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(testfunction(**kwargs))
        return True
    return None
