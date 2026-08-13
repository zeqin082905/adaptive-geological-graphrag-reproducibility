"""
LLM 接口抽象层
统一封装 Ollama（OpenAI 兼容接口），支持同步/异步调用、结构化输出解析。
若未来切换云端模型，只需修改 LLMConfig，业务代码无需变动。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import requests
from config.settings import cfg, LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    封装 Ollama OpenAI 兼容接口。
    所有需要调用 LLM 的模块统一通过此类，不直接实例化 OpenAI 客户端。
    
    注意：由于 openai 2.31.0 与 Ollama 存在兼容性问题（Error 502），
    改用 requests 直接调用 OpenAI 兼容 API。
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self._cfg = config or cfg.llm
        # 构建完整的 API 端点 URL
        base = self._cfg.base_url.rstrip('/')
        self._api_url = f"{base}/chat/completions"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}"
        }

    # ── 基础文本生成 ──────────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        system: str = "你是一个专业的地质领域知识助手。",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """单轮文本生成，返回字符串。"""
        try:
            payload = {
                "model": self._cfg.model,
                "temperature": temperature or self._cfg.temperature,
                "max_tokens": max_tokens or self._cfg.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
            
            response = requests.post(
                self._api_url,
                json=payload,
                headers=self._headers,
                timeout=self._cfg.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM HTTP 请求失败: {e}")
            raise
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"LLM 响应解析失败: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    # ── 结构化 JSON 输出 ──────────────────────────────────────────────────────
    def generate_json(
        self,
        prompt: str,
        system: str = "你是一个专业的地质领域知识助手。请严格以 JSON 格式输出，不要包含任何额外说明。",
        temperature: Optional[float] = None,
    ) -> Any:
        """
        调用 LLM 并将返回内容解析为 JSON 对象。
        自动清理 markdown 代码块包裹。
        """
        raw = self.generate(prompt, system=system, temperature=temperature)
        return self._parse_json(raw)

    # ── 批量调用（用于实体关系抽取） ─────────────────────────────────────────
    def batch_generate(
        self,
        prompts: list[str],
        system: str = "你是一个专业的地质领域知识助手。",
        temperature: Optional[float] = None,
    ) -> list[str]:
        """
        顺序批量调用。
        TODO: 可替换为 asyncio 并发版本以提升吞吐量。
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"批量调用 [{i+1}/{len(prompts)}]")
            results.append(self.generate(prompt, system=system, temperature=temperature))
        return results

    # ── 私有工具 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> Any:
        """清理 LLM 输出中的 markdown 格式并解析 JSON。"""
        # 去除 ```json ... ``` 包裹
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试提取第一个 JSON 对象/数组
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            logger.error(f"JSON 解析失败，原始内容：\n{text}")
            raise


# 全局单例（可被各模块 import 直接使用）
llm = LLMClient()
