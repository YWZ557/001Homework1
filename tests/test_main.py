"""最小验证：python -m unittest discover -s tests

只覆盖「基本要求」相关的关键行为，不做过度测试。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.py").write_text("def f(x):\n    return eval(x)\n", encoding="utf-8")
        self.tools = main.Tools(str(self.root))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_read_file_returns_numbered_content(self):
        out = self.tools.read_file("a.py")
        self.assertIn("1: def f(x):", out)
        self.assertIn("共 2 行", out)

    def test_missing_file_returns_error_text(self):
        self.assertTrue(self.tools.read_file("nope.py").startswith("[错误]"))

    def test_path_traversal_is_blocked(self):
        self.assertIn("只允许访问工作区", self.tools.read_file("../secret.txt"))

    def test_absolute_path_outside_workspace_is_blocked(self):
        outside = Path(tempfile.gettempdir()) / "dsh_h1_probe.txt"
        self.assertIn("只允许访问工作区", self.tools.read_file(str(outside)))

    def test_list_dir_lists_entries(self):
        self.assertIn("a.py", self.tools.list_dir("."))

    def test_search_code_finds_pattern(self):
        self.assertIn("a.py:2:", self.tools.search_code("eval"))

    def test_unknown_tool_is_reported(self):
        self.assertIn("未知工具", self.tools.call("nope", {}))

    def test_bad_arguments_do_not_raise(self):
        self.assertIn("[错误]", self.tools.call("read_file", {"unexpected": 1}))

    def test_schemas_describe_all_tools(self):
        names = {s["function"]["name"] for s in self.tools.schemas()}
        self.assertEqual({"read_file", "list_dir", "search_code"}, names)


class MemoryTests(unittest.TestCase):
    def test_trim_drops_orphan_tool_messages(self):
        mem = main.Memory(max_messages=3, system="sys")
        for i in range(5):
            mem.add("user", f"u{i}")
            mem.add("tool", f"r{i}", tool_call_id=f"id{i}")
        payload = mem.payload()
        self.assertEqual("system", payload[0]["role"])
        self.assertNotEqual("tool", payload[1]["role"])

    def test_clear_keeps_system(self):
        mem = main.Memory(system="sys")
        mem.add("user", "hi")
        mem.clear()
        self.assertEqual(1, len(mem.payload()))


class AgentLoopTests(unittest.TestCase):
    """基本要求：输入 → 推理 → 工具调用 → 输出。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "bad.py").write_text(
            "def f(x):\n    try:\n        return eval(x)\n    except:\n        pass\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _agent(self, max_rounds: int = 6) -> main.Agent:
        return main.Agent(main.MockLLM(), main.Tools(str(self.root)),
                          max_rounds=max_rounds, verbose=False)

    def test_agent_calls_tool_then_answers(self):
        result = self._agent().run("审查 bad.py 的代码质量")
        self.assertGreaterEqual(result.tool_calls, 1)
        self.assertGreaterEqual(result.rounds, 2)
        self.assertIn("代码审查报告", result.answer)
        self.assertIn("eval", result.answer)

    def test_steps_record_full_loop(self):
        kinds = [kind for kind, _ in self._agent().run("审查 bad.py").steps]
        self.assertIn("tool_result", kinds)
        self.assertEqual("answer", kinds[-1])

    def test_memory_carries_context_across_turns(self):
        agent = self._agent()
        agent.run("审查 bad.py")
        before = len(agent.memory.messages)
        agent.run("项目里有哪些文件")
        self.assertGreater(len(agent.memory.messages), before)

    def test_second_turn_does_not_reuse_old_tool_result(self):
        agent = self._agent()
        self.assertIn("代码审查报告", agent.run("审查 bad.py").answer)
        self.assertIn("项目结构", agent.run("项目里有哪些文件和目录").answer)

    def test_empty_input_is_handled(self):
        self.assertIn("输入为空", self._agent().run("   ").answer)

    def test_tool_error_is_reported_gracefully(self):
        self.assertIn("错误", self._agent().run("审查 missing.py").answer)

    def test_max_rounds_guard(self):
        self.assertIn("最大推理轮数", self._agent(max_rounds=1).run("搜索项目里所有 TODO").answer)


class ConfigTests(unittest.TestCase):
    def test_offline_without_key(self):
        self.assertTrue(main.Config(api_key="").offline())
        self.assertFalse(main.Config(api_key="sk-x").offline())

    def test_dotenv_does_not_override_existing_env(self):
        with tempfile.TemporaryDirectory() as d:
            dotenv = Path(d) / ".env"
            dotenv.write_text("LLM_API_KEY=from-file\nLLM_MODEL=m1\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"LLM_MODEL": "from-env"}, clear=False):
                os.environ.pop("LLM_API_KEY", None)
                main._load_dotenv(str(dotenv))
                self.assertEqual("from-file", os.environ["LLM_API_KEY"])
                self.assertEqual("from-env", os.environ["LLM_MODEL"])

    def test_cli_single_shot_runs_offline(self):
        """基本要求：可通过命令行交互（这里验证单轮问答入口真的能跑通）。"""
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"LLM_API_KEY": ""}, clear=False):
                code = main.main(["--ask", "项目里有哪些文件", "--quiet", "--workspace", d])
        self.assertEqual(0, code)


if __name__ == "__main__":
    unittest.main()
