"""
Gateway 转发配置
通过 system_config(gateway_config) 实现全局单例配置
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.utils.database import db_manager
from src.utils.logger import setup_logger

logger = setup_logger("gateway_config")

GATEWAY_CONFIG_KEY = "gateway_config"
SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini")


@dataclass
class GatewayConfig:
    """Gateway 全局配置"""
    provider: str
    base_url: str
    timeout: int = 30
    max_retries: int = 1
    model_mapping: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self):
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {self.provider}")
        if not self.base_url:
            raise ValueError("Gateway base_url is required")
        if self.timeout <= 0:
            raise ValueError("Gateway timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Gateway max_retries must be >= 0")
        if self.model_mapping is None:
            self.model_mapping = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "model_mapping": self.model_mapping or {},
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatewayConfig":
        if not data:
            raise ValueError("Empty gateway config data")
        return cls(
            provider=data["provider"],
            base_url=data["base_url"],
            timeout=int(data.get("timeout", 30)),
            max_retries=int(data.get("max_retries", 1)),
            model_mapping=data.get("model_mapping") or {},
            enabled=bool(data.get("enabled", True)),
        )


def load_gateway_config() -> Optional[GatewayConfig]:
    """从 system_config 中加载 Gateway 配置"""
    raw = db_manager.get_config(GATEWAY_CONFIG_KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return GatewayConfig.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to parse gateway config: {e}")
        return None


def save_gateway_config(config: GatewayConfig) -> None:
    """保存 Gateway 配置到 system_config"""
    db_manager.set_config(GATEWAY_CONFIG_KEY, json.dumps(config.to_dict(), ensure_ascii=False))
    logger.info("Gateway config updated")


def delete_gateway_config() -> None:
    """删除 Gateway 配置"""
    db_manager.delete_config(GATEWAY_CONFIG_KEY)
    logger.info("Gateway config removed")
