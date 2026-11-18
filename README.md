# 评分者尺度差异下的学习成果与志愿岗位匹配

这是一个面向社区老年教育运营的本地分析原型，使用合成记录演示从业务记录到运营摘要的处理。项目把记录读取、统计口径和运营判断分开，便于核对原始事实与派生结果。

## 环境与使用

需要 Python 3.11 或更高版本，无第三方依赖。运行 `python3 -m unittest discover -s tests -v` 执行测试；运行 `python3 -m learning_skill_placement.report` 查看合成数据摘要。所有记录均为本地合成数据，不包含个人信息。
也可使用 Docker 运行：`docker build -t learning-skill-placement .`，再执行 `docker run --rm learning-skill-placement`。容器内运行测试可执行 `docker run --rm --entrypoint python3 learning-skill-placement -m unittest discover -s tests -v`。

## 现有行为

作品评分网络决定能力可比性；资格判断消费不确定性与重评；岗位匹配改变共享带教余量而不得追溯撤销旧资格。

当前摘要只反映样例记录的基础汇总，不用于形成个体资格或运营承诺。SQLite 数据库在调用时创建于临时目录，原始 CSV 保持不变。
