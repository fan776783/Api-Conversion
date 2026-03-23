"""
渠道管理器
负责管理用户配置的API渠道，包括增删改查操作
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import setup_logger
from src.utils.exceptions import ChannelNotFoundError, ConfigurationError
from src.utils.database import db_manager

logger = setup_logger("channel_manager")


@dataclass
class ChannelInfo:
    """渠道信息"""
    id: str
    name: str
    provider: str  # openai, anthropic, gemini
    base_url: str
    api_key: str
    custom_key: str  # 用户自定义的key，用于调用此渠道
    timeout: int = 30
    max_retries: int = 3
    enabled: bool = True
    models_mapping: Optional[Dict[str, str]] = None
    default_target_format: Optional[str] = None
    supported_formats: Optional[List[str]] = None
    # Payload 配置：支持请求参数的默认值设置和强制覆盖
    # 格式: {"default": {"key": value}, "override": {"key": value}}
    payload_config: Optional[Dict[str, Any]] = None
    # 代理配置
    use_proxy: bool = False
    proxy_type: Optional[str] = None  # http, https, socks5
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelInfo':
        """从字典创建ChannelInfo实例"""
        return cls(**data)


class ChannelManager:
    """渠道管理器"""

    def __init__(self):
        pass
    
    def add_channel(
        self,
        name: str,
        provider: str,
        base_url: str,
        api_key: str,
        custom_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        models_mapping: Optional[Dict[str, str]] = None,
        default_target_format: Optional[str] = None,
        supported_formats: Optional[List[str]] = None,
        payload_config: Optional[Dict[str, Any]] = None,
        use_proxy: bool = False,
        proxy_type: Optional[str] = None,
        proxy_host: Optional[str] = None,
        proxy_port: Optional[int] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None
    ) -> str:
        """添加新渠道"""
        if provider not in ['openai', 'anthropic', 'gemini']:
            raise ValueError(f"Unsupported provider: {provider}")

        return db_manager.add_channel(
            name=name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            custom_key=custom_key,
            timeout=timeout,
            max_retries=max_retries,
            models_mapping=models_mapping,
            default_target_format=default_target_format,
            supported_formats=supported_formats,
            payload_config=payload_config,
            use_proxy=use_proxy,
            proxy_type=proxy_type,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            proxy_username=proxy_username,
            proxy_password=proxy_password
        )
    
    def update_channel(
        self,
        channel_id: str,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        enabled: Optional[bool] = None,
        models_mapping: Optional[Dict[str, str]] = None,
        default_target_format: Optional[str] = None,
        supported_formats: Optional[List[str]] = None,
        payload_config: Optional[Dict[str, Any]] = None,
        use_proxy: Optional[bool] = None,
        proxy_type: Optional[str] = None,
        proxy_host: Optional[str] = None,
        proxy_port: Optional[int] = None,
        proxy_username: Optional[str] = None,
        proxy_password: Optional[str] = None
    ) -> bool:
        """更新渠道信息"""
        success = db_manager.update_channel(
            channel_id=channel_id,
            name=name,
            base_url=base_url,
            api_key=api_key,
            custom_key=custom_key,
            timeout=timeout,
            max_retries=max_retries,
            enabled=enabled,
            models_mapping=models_mapping,
            default_target_format=default_target_format,
            supported_formats=supported_formats,
            payload_config=payload_config,
            use_proxy=use_proxy,
            proxy_type=proxy_type,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            proxy_username=proxy_username,
            proxy_password=proxy_password
        )

        if not success:
            raise ChannelNotFoundError(f"Channel not found: {channel_id}")

        return True
    
    def delete_channel(self, channel_id: str) -> bool:
        """删除渠道"""
        success = db_manager.delete_channel(channel_id)
        if not success:
            raise ChannelNotFoundError(f"Channel not found: {channel_id}")
        return True

    def get_channel(self, channel_id: str) -> Optional[ChannelInfo]:
        """获取渠道信息"""
        data = db_manager.get_channel(channel_id)
        return ChannelInfo.from_dict(data) if data else None

    def get_channel_by_custom_key(self, custom_key: str) -> Optional[ChannelInfo]:
        """根据自定义key获取渠道信息"""
        data = db_manager.get_channel_by_custom_key(custom_key)
        return ChannelInfo.from_dict(data) if data else None

    def get_channels_by_provider(self, provider: str) -> List[ChannelInfo]:
        """按提供商获取渠道列表"""
        channels_data = db_manager.get_channels_by_provider(provider)
        return [ChannelInfo.from_dict(data) for data in channels_data]

    def get_all_channels(self) -> List[ChannelInfo]:
        """获取所有渠道"""
        channels_data = db_manager.get_all_channels()
        return [ChannelInfo.from_dict(data) for data in channels_data]

    def get_enabled_channels(self) -> List[ChannelInfo]:
        """获取所有启用的渠道"""
        channels_data = db_manager.get_enabled_channels()
        return [ChannelInfo.from_dict(data) for data in channels_data]
    
    
    def test_channel_connection(self, channel_id: str) -> Dict[str, Any]:
        """测试渠道连接"""
        channel = self.get_channel(channel_id)
        if not channel:
            raise ChannelNotFoundError(f"Channel not found: {channel_id}")
        
        # 这里可以实现实际的连接测试
        # 暂时返回模拟结果
        return {
            "channel_id": channel_id,
            "status": "ok",
            "response_time": 100,
            "message": "Connection test successful"
        }
    
    def get_channel_statistics(self) -> Dict[str, Any]:
        """获取渠道统计信息"""
        all_channels = self.get_all_channels()
        enabled_channels = self.get_enabled_channels()

        total_channels = len(all_channels)
        enabled_count = len(enabled_channels)

        provider_counts = {}
        for channel in all_channels:
            provider_counts[channel.provider] = provider_counts.get(channel.provider, 0) + 1

        return {
            "total_channels": total_channels,
            "enabled_channels": enabled_count,
            "disabled_channels": total_channels - enabled_count,
            "provider_counts": provider_counts
        }


# 全局渠道管理器实例
channel_manager = ChannelManager()