import ast
import astunparse
import re
from typing import List

from KVCOMM.tools.coding.executor_utils import function_with_timeout
from KVCOMM.tools.coding.executor_types import ExecuteResult, Executor
import multiprocessing as mp
import textwrap
import traceback
import queue
import io
from contextlib import redirect_stdout


def _extract_python_candidate(text: str) -> str | None:
    """Return executable python snippet or None if input is not code."""
    if text is None:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    # Prefer explicit fenced python blocks.
    py_blocks = re.findall(r"```python\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if py_blocks:
        candidate = py_blocks[-1].strip()
        return candidate or None

    # Generic fenced block fallback.
    fenced_blocks = re.findall(r"```\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if fenced_blocks:
        candidate = fenced_blocks[-1].strip()
        if candidate.lower().startswith("python\n"):
            candidate = candidate.split("\n", 1)[1].strip()
        return candidate or None

    # No fenced block: only execute if the whole text is valid Python.
    try:
        ast.parse(raw)
    except SyntaxError:
        return None
    return raw


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left               
    except:
        call_str = ast_parsed.body[0].test               

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)












def execute_code_get_return(code: str, timeout: int = 5):
    """在子进程执行 code，超过 timeout 秒直接终止，并返回 answer 变量（或错误信息）。"""
    code_to_run = _extract_python_candidate(code)
    if code_to_run is None:
        return "Skipped: non-python content"

    def _runner(q):
        namespace = {}
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(textwrap.dedent(code_to_run), namespace, namespace)
            res = namespace.get("answer")

            if callable(res):
                res = res()

            if res is None:
                printed = buf.getvalue().strip()
                if printed:
                    lines = [line for line in printed.splitlines() if line.strip()]
                    res = lines[-1] if lines else printed
                else:
                    res = "Executed: no answer variable"

            q.put(res)
        except Exception:
            q.put(f"Error: {traceback.format_exc().splitlines()[-1]}")

    q = mp.Queue()
    p = mp.Process(target=_runner, args=(q,))
    p.start()

    try:

        result = q.get(timeout=timeout)
    except queue.Empty:
        p.terminate()
        p.join()
        return f"Timeout (> {timeout}s)"
    else:
        p.join()
        return result

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:

        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]


        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                success_tests.append(tests[i])
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests.append(f"{tests[i]} # output: {output}")
                is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """

        code = f"""{func}

{test}

check({name})
    """
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False

