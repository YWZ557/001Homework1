"""Code Agent —— 代码助手 Agent（Homework 1）

必须实现：Agent 循环（输入→推理→工具调用→输出）、至少一种工具、命令行交互
推荐实现：LLM 原生 API、上下文记忆、错误处理与重试

单文件实现，零第三方依赖；未配置 LLM_API_KEY 时自动走离线兜底模式。

用法：
    python main.py                              # 交互式会话
    python main.py --ask "审查 demo/sample.py"   # 单轮问答
    python main.py --rounds 4                   # 最大推理轮数
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# 1. 配置
# ============================================================================


@dataclass
class Config:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    workspace: str = "."
    max_rounds: int = 6
    timeout: int = 60
    retries: int = 3

    @classmethod
    def load(cls, dotenv: str = ".env") -> "Config":
        _load_dotenv(dotenv)
        return cls(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", cls.base_url),
            model=os.environ.get("LLM_MODEL", cls.model),
            workspace=os.environ.get("AGENT_WORKSPACE", "."),
            max_rounds=int(os.environ.get("AGENT_MAX_ROUNDS", "6")),
            timeout=int(os.environ.get("LLM_TIMEOUT", "60")),
            retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
        )

    def offline(self) -> bool:
        return not self.api_key


def _load_dotenv(path: str) -> None:
    """不覆盖已存在的环境变量，因此 .env 不会顶掉命令行或系统设置。"""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


# ============================================================================
# 2. 工具：统一描述 + 统一异常隔离
# ============================================================================


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: object

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}


class Tools:
    """工具注册表。

    任何异常都被转成 `[错误] ...` 文本回灌给模型，让模型自己纠正，
    因此单个工具出错不会让整个会话崩溃。
    """

    SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}

    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()
        self._tools: dict[str, Tool] = {}
        self.add(
            "read_file", "读取文件内容并带行号返回。分析任何代码前都应先调用它。",
            {"type": "object", "properties": {
                "path": {"type": "string", "description": "相对工作区的文件路径"}},
             "required": ["path"]},
            self.read_file,
        )
        self.add(
            "list_dir", "列出工作区内某个目录的子目录与文件，用于定位要分析的文件。",
            {"type": "object", "properties": {
                "path": {"type": "string", "description": "目录路径，默认 '.'"}},
             "required": []},
            self.list_dir,
        )
        self.add(
            "search_code", "在工作区代码中按正则搜索，返回 文件:行号:内容。",
            {"type": "object", "properties": {
                "pattern": {"type": "string", "description": "正则表达式，如 'TODO'"}},
             "required": ["pattern"]},
            self.search_code,
        )

    def add(self, name: str, description: str, parameters: dict, func) -> None:
        self._tools[name] = Tool(name, description, parameters, func)

    def schemas(self) -> list:
        return [t.schema() for t in self._tools.values()]

    def describe(self) -> str:
        return "\n".join(f"  - {t.name}: {t.description}" for t in self._tools.values())

    def call(self, name: str, arguments) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"[错误] 未知工具 {name}；可用：{', '.join(self._tools)}"
        if isinstance(arguments, str):     # 模型有时会把参数发成 JSON 字符串
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return f"[错误] 参数不是合法 JSON：{arguments}"
        try:
            return str(tool.func(**arguments))
        except TypeError as exc:
            return f"[错误] 参数不匹配：{exc}"
        except Exception as exc:  # noqa: BLE001 - 工具异常必须被隔离
            return f"[错误] {name} 执行失败：{type(exc).__name__}: {exc}"

    # -- 三个工具的实现 ----------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """按路径分量比较来限制工作区，避免 `../` 与「同名前缀兄弟目录」越权。"""
        target = (self.workspace / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if os.path.commonpath([str(self.workspace), str(target)]) != str(self.workspace):
            raise PermissionError(f"只允许访问工作区内的路径：{self.workspace}")
        return target

    def read_file(self, path: str) -> str:
        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return f"[错误] {exc}"
        if not target.is_file():
            return f"[错误] 文件不存在：{target}"
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"[错误] 读取失败：{exc}"
        body = "\n".join(f"{i:>4}: {line}" for i, line in enumerate(lines, 1))
        return f"文件：{target}（共 {len(lines)} 行）\n{body}"

    def list_dir(self, path: str = ".") -> str:
        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return f"[错误] {exc}"
        if not target.is_dir():
            return f"[错误] 目录不存在：{target}"
        entries = [p for p in sorted(target.iterdir())
                   if not p.name.startswith(".") and p.name not in self.SKIP_DIRS]
        body = "\n".join(f"[DIR ] {p.name}/" if p.is_dir() else f"[FILE] {p.name}" for p in entries)
        return f"目录：{target}\n{body or '（空目录）'}"

    def search_code(self, pattern: str, max_results: int = 30) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"[错误] 正则表达式非法：{exc}"
        hits = []
        for file in sorted(self.workspace.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in (".py", ".md", ".txt", ".json", ".js", ".ts"):
                continue
            if any(part.startswith(".") or part in self.SKIP_DIRS for part in file.parts):
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{file.relative_to(self.workspace)}:{lineno}: {line.strip()[:120]}")
                    if len(hits) >= max_results:
                        return f"共 {len(hits)} 处匹配（已截断）：\n" + "\n".join(hits)
        if not hits:
            return f"未找到匹配 `{pattern}` 的内容"
        return f"共 {len(hits)} 处匹配：\n" + "\n".join(hits)


# ============================================================================
# 3. 记忆
# ============================================================================


@dataclass
class Memory:
    max_messages: int = 20
    system: str = ""
    messages: list = field(default_factory=list)

    def add(self, role: str, content: str, **extra) -> None:
        self.messages.append({"role": role, "content": content, **extra})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        # 开头的孤儿 tool 消息缺少对应的 tool_calls，会让 API 直接报错
        while self.messages and self.messages[0]["role"] == "tool":
            self.messages.pop(0)

    def payload(self) -> list:
        return [{"role": "system", "content": self.system}] + self.messages

    def clear(self) -> None:
        self.messages.clear()


# ============================================================================
# 4. LLM 接入：OpenAI 兼容协议 + 离线兜底 + 重试
# ============================================================================

SYSTEM_PROMPT = """你是一个代码助手 Agent，擅长代码审查、代码解释与重构建议。

## 工作方式
1. 先用一两句话说明你的判断思路；
2. 需要了解代码时**必须调用 read_file 等工具获取真实内容**，禁止凭空猜测代码；
3. 拿到工具结果后再给出结论。

## 可用工具
read_file（读文件，带行号）、list_dir（看目录）、search_code（正则搜索）

## 输出格式
【代码审查】结论（含风险等级 高/中/低）→ 问题清单（位置 / 问题 / 影响 / 建议）→ 改进示例
【代码解释】功能概述 → 逐段讲解 → 注意事项

## 约束
用中文回答，代码与标识符保持原文；只基于工具返回的真实内容作答；不要修改文件。
"""


@dataclass
class Reply:
    content: str = ""
    calls: list = field(default_factory=list)   # [{"id", "name", "arguments"}]


class OpenAICompatLLM:
    """OpenAI 兼容 /chat/completions，适配 OpenAI / DeepSeek / 通义千问 / 智谱。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.endpoint = cfg.base_url.rstrip("/") + "/chat/completions"

    def chat(self, messages: list, tools: list) -> Reply:
        payload = {"model": self.cfg.model, "messages": messages,
                   "temperature": 0.2, "tools": tools, "tool_choice": "auto"}
        last = ""
        for attempt in range(1, self.cfg.retries + 1):
            try:
                return self._post(payload)
            except Exception as exc:  # noqa: BLE001 - 统一退避重试
                last = str(exc)
                if attempt < self.cfg.retries:
                    delay = min(2 ** (attempt - 1), 8)
                    print(f"[llm] 第 {attempt} 次调用失败：{last}；{delay}s 后重试…")
                    time.sleep(delay)
        raise RuntimeError(f"重试 {self.cfg.retries} 次仍失败：{last}")

    def _post(self, payload: dict) -> Reply:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.cfg.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc

        message = body["choices"][0]["message"]
        calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": raw.get("id", f"call_{len(calls)}"),
                          "name": fn.get("name", ""), "arguments": args})
        return Reply(content=message.get("content") or "", calls=calls)


class MockLLM:
    """离线兜底：无 API Key 时用规则复现完整的「推理 → 工具调用 → 总结」循环。

    它先识别问题里的文件路径并调用 read_file，拿到真实内容后做关键字静态检查，
    因此没有 Key 也能给出真实的审查结果，而不是空转。
    """

    PATH_RE = re.compile(r"[\w./\\-]+\.(?:py|js|ts|java|go|md|json)\b")

    def chat(self, messages: list, tools: list) -> Reply:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        tool_msgs = self._tool_messages_this_turn(messages)

        if tool_msgs:
            return Reply(content=self._summarize(question, tool_msgs[-1]))
        if any(k in question for k in ("目录", "哪些文件", "结构")):
            return Reply(calls=[{"id": "c1", "name": "list_dir", "arguments": {"path": "."}}])
        if any(k in question for k in ("搜索", "查找", "哪里用到")):
            token = (re.findall(r"[A-Za-z_]\w{2,}", question) or ["TODO"])[0]
            return Reply(calls=[{"id": "c1", "name": "search_code", "arguments": {"pattern": token}}])
        match = self.PATH_RE.search(question)
        if match:
            return Reply(calls=[{"id": "c1", "name": "read_file",
                                 "arguments": {"path": match.group(0).replace("\\", "/")}}])
        return Reply(content=self._help())

    @staticmethod
    def _tool_messages_this_turn(messages: list) -> list:
        """只取最后一条 user 消息之后的 tool 结果，避免复用上一轮内容。"""
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        return [m for m in messages[last_user + 1:] if m["role"] == "tool"]

    @staticmethod
    def _help() -> str:
        return (
            "我是 **Code Agent**，一个代码审查与解释助手，遵循「输入 → 推理 → 工具调用 → 输出」循环。\n\n"
            "**可用工具**：read_file / list_dir / search_code\n\n"
            "**试试这样问**：\n"
            "- 审查 demo/sample.py 的代码质量\n"
            "- 解释 main.py 里 Agent 循环是怎么工作的\n"
            "- 项目里有哪些文件和目录\n"
            "- 搜索项目里所有 TODO\n\n"
            "> 当前为离线兜底模式（未配置 LLM_API_KEY），配置真实 Key 后可用大模型回答任意问题。"
        )

    @staticmethod
    def _summarize(question: str, tool_message: dict) -> str:
        content = tool_message.get("content", "")
        if content.startswith("[错误]"):
            return f"{content}\n\n请确认路径是否正确，或换一个文件再试。"
        if "目录：" in content:
            return (f"## 项目结构\n```\n{content}\n```\n"
                    "请指定具体文件（如「审查 demo/sample.py」），我会读取内容后给出分析。")
        if content.startswith("共 ") or content.startswith("未找到"):
            return f"## 搜索结果\n```\n{content}\n```\n如需深入某一处，请告诉我文件与行号。"
        return MockLLM._review(content)

    @staticmethod
    def _review(file_dump: str) -> str:
        lines = []
        for line in file_dump.splitlines():
            m = re.match(r"^\s*(\d+): (.*)$", line)
            if m:
                lines.append((int(m.group(1)), m.group(2)))
        header = file_dump.splitlines()[0] if file_dump.splitlines() else ""

        issues, func_start, func_len, func_name, in_doc = [], None, 0, "", False
        for lineno, text in lines:
            stripped = text.strip()
            # docstring 里也会出现 TODO / def 等关键字，必须跳过以免误报
            if '"""' in text or "'''" in text:
                if text.count('"""') < 2 and text.count("'''") < 2:
                    in_doc = not in_doc
                continue
            if in_doc:
                continue
            if stripped.startswith("def "):
                if func_start and func_len > 40:
                    issues.append(f"- 第 {func_start} 行｜函数 `{func_name}` 过长（约 {func_len} 行），建议拆分。")
                m = re.match(r"def (\w+)", stripped)
                func_start, func_len, func_name = lineno, 0, (m.group(1) if m else "")
            if func_start:
                func_len += 1
            if stripped == "except:" or re.match(r"^except\s*:", stripped):
                issues.append(f"- 第 {lineno} 行｜裸 `except:` 会吞掉所有错误，应捕获具体异常。")
            if "eval(" in stripped or "exec(" in stripped:
                issues.append(f"- 第 {lineno} 行｜高风险：`eval/exec` 可能造成代码注入，应移除。")
            if "TODO" in stripped or "FIXME" in stripped:
                issues.append(f"- 第 {lineno} 行｜遗留标记应处理或登记为任务。")
            if re.search(r"def \w+\([^)]*=\s*(\[\]|\{\})", stripped):
                issues.append(f"- 第 {lineno} 行｜可变默认参数会在多次调用间共享，应改为 None。")
        if func_start and func_len > 40:
            issues.append(f"- 第 {func_start} 行｜函数 `{func_name}` 过长（约 {func_len} 行），建议拆分。")
        if not issues:
            issues.append("- 未发现明显问题，代码风格良好；建议补充单元测试覆盖边界条件。")

        level = "高" if len(issues) >= 5 else ("中" if len(issues) >= 3 else "低")
        return (f"## 代码审查报告\n**来源**：{header}\n\n"
                f"### 结论\n共发现 {len(issues)} 处待改进点，风险等级：**{level}**。\n\n"
                "### 问题清单\n" + "\n".join(issues) + "\n\n"
                "> 离线兜底模式由规则引擎生成；配置 LLM_API_KEY 后可获得模型级深度分析。")


# ============================================================================
# 5. Agent 循环：输入 → 推理 → 工具调用 → 输出
# ============================================================================


@dataclass
class Result:
    answer: str
    rounds: int = 0
    tool_calls: int = 0
    steps: list = field(default_factory=list)


class Agent:
    """只负责编排循环，不关心模型与工具的实现细节。"""

    def __init__(self, llm, tools: Tools, max_rounds: int = 6, verbose: bool = True):
        self.llm, self.tools = llm, tools
        self.memory = Memory(system=SYSTEM_PROMPT)
        self.max_rounds, self.verbose = max_rounds, verbose

    def run(self, question: str) -> Result:
        if not question.strip():
            return Result(answer="[提示] 输入为空，请输入你的问题。")

        self.memory.add("user", question)
        result = Result(answer="")
        for round_no in range(1, self.max_rounds + 1):
            try:
                reply = self.llm.chat(self.memory.payload(), self.tools.schemas())
            except Exception as exc:  # noqa: BLE001 - 转成友好提示
                result.answer = (f"[错误] LLM 调用失败：{exc}\n"
                                 "提示：检查 LLM_API_KEY / LLM_BASE_URL / 网络。")
                result.rounds = round_no
                return result

            if reply.calls:
                self.memory.add("assistant", reply.content, tool_calls=[
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                    for c in reply.calls])
                if reply.content:
                    result.steps.append(("thought", reply.content))
                    self._log("思考", reply.content)
                for call in reply.calls:
                    result.tool_calls += 1
                    self._log("工具调用", f"{call['name']} {call['arguments']}")
                    output = self.tools.call(call["name"], call["arguments"])
                    self.memory.add("tool", output, tool_call_id=call["id"], name=call["name"])
                    result.steps.append(("tool_result", output))
                    self._log("工具返回", output.splitlines()[0][:120] if output else "")
                continue

            answer = reply.content.strip() or "[提示] 模型返回空内容，请换个说法重试。"
            self.memory.add("assistant", answer)
            result.answer, result.rounds = answer, round_no
            result.steps.append(("answer", answer))
            return result

        result.rounds = self.max_rounds
        result.answer = f"[中止] 已达最大推理轮数 {self.max_rounds}，仍未得出结论，请缩小问题范围。"
        return result

    def _log(self, prefix: str, text: str) -> None:
        if self.verbose:
            print(f"  [{prefix}] {text}")


# ============================================================================
# 6. 命令行入口
# ============================================================================

HELP = """可用命令：
  /help           显示本帮助
  /tools          列出可用工具
  /clear          清空对话记忆
  exit / quit / q 退出

直接输入自然语言即可，例如：
  审查 demo/sample.py 的代码质量
  解释 main.py 里 Agent 循环是怎么工作的
  搜索项目里所有 TODO
"""


def build_agent(cfg: Config, verbose: bool = True) -> Agent:
    llm = MockLLM() if cfg.offline() else OpenAICompatLLM(cfg)
    return Agent(llm, Tools(cfg.workspace), max_rounds=cfg.max_rounds, verbose=verbose)


def print_status(cfg: Config) -> None:
    mode = "离线兜底模式（未配置 LLM_API_KEY）" if cfg.offline() else f"真实模型 {cfg.model}"
    print(f"模式：{mode}｜工作区：{Path(cfg.workspace).resolve()}｜最大轮数：{cfg.max_rounds}")


def run_once(cfg: Config, question: str, quiet: bool) -> int:
    result = build_agent(cfg, verbose=not quiet).run(question)
    print(result.answer)
    print(f"\n[统计] 推理轮数 {result.rounds}｜工具调用 {result.tool_calls} 次")
    return 0


def run_repl(cfg: Config) -> int:
    agent = build_agent(cfg)
    print_status(cfg)
    print(HELP)
    while True:
        try:
            user = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if not user:
            continue
        if user.lower() in ("exit", "quit", "q"):
            print("bye.")
            return 0
        if user == "/help":
            print(HELP)
            continue
        if user == "/tools":
            print("可用工具：\n" + agent.tools.describe())
            continue
        if user == "/clear":
            agent.memory.clear()
            print("已清空对话记忆。")
            continue
        result = agent.run(user)
        print("\nagent > " + result.answer)
        print(f"[统计] 推理轮数 {result.rounds}｜工具调用 {result.tool_calls} 次")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Code Agent：代码审查 / 解释助手")
    parser.add_argument("--ask", "-a", help="单轮提问，给出答案后退出")
    parser.add_argument("--workspace", "-w", default=None, help="Agent 可访问的工作区根目录，默认当前目录")
    parser.add_argument("--rounds", type=int, default=None, help="最大推理轮数，默认 6")
    parser.add_argument("--quiet", action="store_true", help="不打印中间推理过程")
    args = parser.parse_args(argv)

    cfg = Config.load()
    cfg.workspace = args.workspace or str(Path(__file__).parent)
    if args.rounds:
        cfg.max_rounds = args.rounds
    if not Path(cfg.workspace).exists():
        print(f"[错误] 工作区不存在：{cfg.workspace}")
        return 2
    return run_once(cfg, args.ask, args.quiet) if args.ask else run_repl(cfg)


if __name__ == "__main__":
    sys.exit(main())
