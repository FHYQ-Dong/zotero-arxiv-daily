from openai import OpenAI
from loguru import logger
from time import sleep
import json
import shutil
import subprocess

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError("API key must be provided for OpenAI.")
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

class ClaudeCLI:
    """Runs prompts through the Claude Code CLI (`claude -p`), which authenticates with CLAUDE_CODE_OAUTH_TOKEN.
    The token is only ever used by Claude Code itself, never passed to an API client here."""
    def __init__(self, model: str = None, lang: str = "English"):
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict], json_schema: dict = None):
        """Returns the reply text, or the parsed object when json_schema is given."""
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        prompt = "\n".join(m["content"] for m in messages if m["role"] != "system")
        cmd = [shutil.which("claude") or "claude", "-p", "--output-format", "json", "--tools", "", "--no-session-persistence"]
        if system:
            cmd += ["--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        if json_schema:
            cmd += ["--json-schema", json.dumps(json_schema)]
        # The prompt goes through stdin to stay clear of command-line length limits
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=600)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI timed out after 600s.")
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"Claude CLI exited with code {proc.returncode}: {proc.stderr[-300:]}")
        if result.get("is_error") or proc.returncode != 0:
            # Usage-limit errors (429) on a subscription also land here; the paper is skipped
            raise RuntimeError(f"Claude CLI failed (status {result.get('api_error_status')}): {str(result.get('result'))[:300]}")
        return result["structured_output"] if json_schema else result["result"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English", backend: str = "openai"):
    global GLOBAL_LLM
    if backend == "claude_cli":
        GLOBAL_LLM = ClaudeCLI(model=model, lang=lang)
    else:
        GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM