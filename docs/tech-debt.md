# 技术债清单

记录已知但暂时不修的问题。每条都写清「问题」「为什么暂不修」「修法」，便于以后定点处理。

---

## 1. `max_length` 与真实传输能力对齐问题（已部分处理）

- **现状**：`/token-count` 的 `max_length=10_000`。
- **背景**：原本设为 `100_000`，但写边界测试时发现 httpx2 直接抛
  `InvalidURL: URL component 'query' too long` —— 请求在客户端构造阶段就失败了。
- **结论**：**一个触发不到的约束等于没有约束**。约束值必须和链路实际能力对齐
  （浏览器约 2–8KB、Nginx 默认 8KB、CDN/WAF 常见 4–8KB）。
- **残留问题**：用 GET + query string 传文本，本身就受 URL 长度限制，
  且**文本内容会进入访问日志、浏览器历史、代理日志**。长文本场景不应走 GET。
- **修法**：新增 `POST /token-count`，用请求体接收文本。第 5 周做 LLM client 时
  会顺带完成（发消息给模型本来就必须用 POST）。

## 2. `src/app.py` 遗留问题

- **死代码**：第 22–24 行有一段用三引号包起来的旧实现。
  注意它是**字符串字面量而非注释**，解释器会真的构造该对象，且具误导性。
  历史版本可用 `git show 02ab2dc:src/app.py` 取回。
- **注释缩进**：`min_length=1,` 下方第一行注释缩进多出一截，
  不影响运行（注释被解释器忽略）但影响可读性。
- **文件末尾缺换行**：`src/app.py`、`tests/test_app.py` 均无结尾换行符，
  导致每次 diff 都出现 `\ No newline at end of file` 噪音。
- **修法**：一条 `style:` 提交即可全部解决。

## 3. 自制测试运行器，尚未使用 pytest

- **现状**：`tests/*.py` 在 `if __name__ == "__main__":` 里**手工列出每个测试函数**。
- **风险**：**漏写一行调用，该测试静默不执行，但程序照样打印 "all tests passed"**
  —— 这是"假绿灯"，是测试体系里最危险的状态。
- **背景**：Windows 侧 pip 装包受限（网络原因），暂用标准库 `assert` 替代。
- **修法**：网络恢复后 `pip install pytest`，删除 `__main__` 块与
  `sys.path` hack（pytest 会自动处理导入路径）。

## 4. 环境相关的暂缓项

- **Python 版本**：项目用 `py -3.11` 的 venv（3.11 生态兼容性最稳）。
  第 17–20 周需要的 `vllm`、`bitsandbytes` 在新版本 Python 上适配滞后。
- **Windows 控制台编码**：中文 commit message 曾因 PowerShell 编码链损坏。
  当前策略是 **commit message 一律用英文**，彻底绕开该问题。
- **待验证**：`$PROFILE` 中的 UTF-8 配置是否已生效。

---

## 已解决并值得记住的坑（不算债，作为记录）

| 问题 | 根因 | 解法 |
|---|---|---|
| `.venv` 被提交 1465 个文件 | 先 `git add .` 后建 `.gitignore` | `git rm -r --cached .venv` 后补 `.gitignore` |
| `git push` HTTPS 超时 | Git 命令行不读系统代理，443 直连被阻断 | `~/.ssh/config` 配 `HostName ssh.github.com` + `Port 443` |
| commit 邮箱为 QQ 号 | `user.email` 未设 | 改用 GitHub `noreply` 地址，兼顾隐私与贡献图 |
| `dubious ownership` | 目录所有者是 Administrators | `git config --global --add safe.directory`（不用 `*`） |
| 500 错误无信息 | 浏览器只显示通用文案 | **原因永远在服务端日志**（`ResponseValidationError` 等） |
| 404 而非 500 | 丢失 `@app.get` 装饰器 | 函数存在 ≠ 路由已注册；查 `/docs` 是否列出该接口 |



## 5. 本地环境:代理静默介入 HTTP 请求

- **现象**：本机 `httpx2.Client()`（默认 `trust_env=True`）请求
  `http://127.0.0.1:8000/health` 返回 **502**，而 `trust_env=False` 返回 200。
  同一环境下 `os.environ` 无任何代理变量、`netsh winhttp show proxy`
  显示 Direct access、注册表 `ProxyEnable=0x0`（但 `ProxyServer` 仍保留
  `127.0.0.1:7897`，且该端口有 Clash 在监听）。
- **影响**：代码行为隐式依赖运行环境的网络配置；本地走代理、云端直连，
  同一份代码在两处行为不一致。
- **已处理**：新增配置项 `LLM_TRUST_ENV`，默认 `true` 保持标准行为
  （企业内网依赖代理），本地设为 `false` 直连。
- **未查明**：代理信息的具体读取来源（环境变量、WinHTTP、注册表均已排除）。
  判定为纯本地环境问题，不影响云端部署，暂不深究。

## 6. 模型输出未经净化，前端渲染需注意 XSS

- **现象**：`POST /chat` 的 `reply` 字段中出现了模型生成的 HTML 标签
  （如 `<font color="#1e40af">`）。
- **风险**：模型输出属于**不可信输入**。若前端以 `innerHTML` 直接插入，
  会引入 XSS。当前接口只返回纯文本 JSON，风险在前端。
- **修法**：前端使用 `textContent` 而非 `innerHTML`；或对模型输出做
  HTML 净化（如 DOMPurify）。若本项目后续提供 Web 界面，需先处理此项。

## 7. 每个请求新建 LLMClient，TLS 握手开销重复

- **现象**：`app.py` 的 `/chat` 每次请求都 `LLMClient()` 并在 `finally`
  中 `close()`，连接池无法复用。
- **影响**：每次请求都付出一次 TLS 握手成本（约 100–300ms）。功能正确，
  但高并发下延迟明显。
- **修法**：改为应用级单例，用 FastAPI 的 `lifespan` 在启动时创建、
  关闭时释放，请求内通过 `request.app.state` 复用。

## 8. 重试退避参数不可注入，导致测试变慢

- **现象**：`tests/test_llm_client.py` 运行约 7.8 秒，绝大部分时间消耗在
  重试的 `time.sleep` 上；13 个测试中有 6 个触发了真实等待。
- **修法**：把 `_BACKOFF_BASE_SECONDS` 等参数改为可注入（构造参数或依赖注入），
  测试中传 0，生产用默认值。