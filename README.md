# ① Agent 循环：输入 → 推理 → 工具调用 → 输出
python main.py --ask "审查 demo/sample.py 的代码质量"

# ② 工具能力：文件读取 / 目录浏览 / 代码搜索 三个工具各演示一次
python main.py --ask "解释 main.py 里 Agent 循环是怎么工作的"
python main.py --ask "项目里有哪些文件和目录" --quiet
python main.py --ask "搜索项目里所有 TODO" --quiet

# ③ 上下文记忆：交互模式下连续追问
python main.py

you > 审查 demo/sample.py

you > 项目里还有哪些文件      ← 新一轮，不复用上一轮结果

you > exit

# ④ 边界情况处理
python main.py --ask "审查 不存在的文件.py"          # 可读错误提示，不崩溃
python main.py --ask "" -a ""                        # 空输入提示
python main.py --ask "搜索项目里所有 TODO" --rounds 1 # 触发轮数守卫

# ⑤ 单元测试
python -m unittest discover -s tests