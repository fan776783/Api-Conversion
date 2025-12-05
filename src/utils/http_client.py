"""
HTTP客户端工具，支持代理配置
"""
import httpx
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from src.channels.channel_manager import ChannelInfo
from src.utils.config import ChannelConfig
from src.utils.logger import setup_logger

logger = setup_logger("http_client")


def create_proxy_config(channel_info: ChannelInfo) -> Optional[Dict[str, str]]:
    """从渠道信息创建代理配置"""
    if not getattr(channel_info, 'use_proxy', False):
        logger.debug(f"Channel {channel_info.name}: Proxy disabled")
        return None
    
    proxy_host = getattr(channel_info, 'proxy_host', None)
    proxy_port = getattr(channel_info, 'proxy_port', None)
    
    if not proxy_host or not proxy_port:
        logger.warning(f"Channel {channel_info.name}: Proxy enabled but missing host/port")
        return None
    
    proxy_type = getattr(channel_info, 'proxy_type', 'http').lower()
    proxy_username = getattr(channel_info, 'proxy_username', None)
    proxy_password = getattr(channel_info, 'proxy_password', None)
    
    # 验证代理类型并构建URL
    if proxy_type == 'socks5':
        try:
            import socksio  # 检查是否安装了SOCKS支持
            proxy_scheme = "socks5"
        except ImportError:
            logger.error(f"Channel {channel_info.name}: SOCKS5 proxy requires 'httpx[socks]'. Please install: pip install httpx[socks]")
            return None
    elif proxy_type in ['http', 'https']:
        proxy_scheme = proxy_type
    else:
        logger.error(f"Channel {channel_info.name}: Unsupported proxy type '{proxy_type}'. Supported types: http, https, socks5")
        return None
    
    # 构建代理URL
    proxy_url = f"{proxy_scheme}://"
    if proxy_username and proxy_password:
        # 在日志中隐藏用户名密码
        logger.info(f"Channel {channel_info.name}: Using authenticated proxy {proxy_scheme}://{proxy_host}:{proxy_port}")
        proxy_url += f"{proxy_username}:{proxy_password}@"
    else:
        logger.info(f"Channel {channel_info.name}: Using proxy {proxy_scheme}://{proxy_host}:{proxy_port}")
    
    proxy_url += f"{proxy_host}:{proxy_port}"
    
    # 对于SOCKS5代理，只需要一个通用配置
    if proxy_type == 'socks5':
        proxy_config = {
            "all://": proxy_url
        }
    else:
        # HTTP/HTTPS代理需要分别配置
        proxy_config = {
            "http://": proxy_url,
            "https://": proxy_url
        }
    
    logger.debug(f"Channel {channel_info.name}: Created proxy config for {proxy_type.upper()} requests")
    return proxy_config


def create_proxy_config_from_channel_config(channel_config: ChannelConfig) -> Optional[Dict[str, str]]:
    """从渠道配置创建代理配置"""
    if not getattr(channel_config, 'use_proxy', False):
        logger.debug(f"Config for {channel_config.provider}: Proxy disabled")
        return None
    
    if not channel_config.proxy_host or not channel_config.proxy_port:
        logger.warning(f"Config for {channel_config.provider}: Proxy enabled but missing host/port")
        return None
    
    proxy_type = (channel_config.proxy_type or 'http').lower()
    
    # 验证代理类型并构建URL
    if proxy_type == 'socks5':
        try:
            import socksio  # 检查是否安装了SOCKS支持
            proxy_scheme = "socks5"
        except ImportError:
            logger.error(f"Config for {channel_config.provider}: SOCKS5 proxy requires 'httpx[socks]'. Please install: pip install httpx[socks]")
            return None
    elif proxy_type in ['http', 'https']:
        proxy_scheme = proxy_type
    else:
        logger.error(f"Config for {channel_config.provider}: Unsupported proxy type '{proxy_type}'. Supported types: http, https, socks5")
        return None
    
    # 构建代理URL
    proxy_url = f"{proxy_scheme}://"
    if channel_config.proxy_username and channel_config.proxy_password:
        logger.info(f"Config for {channel_config.provider}: Using authenticated proxy {proxy_scheme}://{channel_config.proxy_host}:{channel_config.proxy_port}")
        proxy_url += f"{channel_config.proxy_username}:{channel_config.proxy_password}@"
    else:
        logger.info(f"Config for {channel_config.provider}: Using proxy {proxy_scheme}://{channel_config.proxy_host}:{channel_config.proxy_port}")
    
    proxy_url += f"{channel_config.proxy_host}:{channel_config.proxy_port}"
    
    # 对于SOCKS5代理，只需要一个通用配置
    if proxy_type == 'socks5':
        proxy_config = {
            "all://": proxy_url
        }
    else:
        # HTTP/HTTPS代理需要分别配置
        proxy_config = {
            "http://": proxy_url,
            "https://": proxy_url
        }
    
    logger.debug(f"Config for {channel_config.provider}: Created proxy config for {proxy_type.upper()} requests")
    return proxy_config


@asynccontextmanager
async def get_http_client(channel_info: ChannelInfo, timeout: float = 30.0):
    """获取配置了代理的HTTP客户端"""
    proxy_config = create_proxy_config(channel_info)
    
    if proxy_config:
        logger.info(f"Channel {channel_info.name}: Creating HTTP client with proxy")
        logger.debug(f"Channel {channel_info.name}: Proxy config keys: {list(proxy_config.keys())}")
    else:
        logger.debug(f"Channel {channel_info.name}: Creating HTTP client without proxy")
    
    proxy = next(iter(proxy_config.values())) if proxy_config else None
    async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
        yield client


@asynccontextmanager
async def get_http_client_from_config(channel_config: ChannelConfig, timeout: float = 30.0):
    """从渠道配置获取配置了代理的HTTP客户端"""
    proxy_config = create_proxy_config_from_channel_config(channel_config)

    if proxy_config:
        logger.info(f"Config for {channel_config.provider}: Creating HTTP client with proxy")
        logger.debug(f"Config for {channel_config.provider}: Proxy config keys: {list(proxy_config.keys())}")
    else:
        logger.debug(f"Config for {channel_config.provider}: Creating HTTP client without proxy")

    proxy = next(iter(proxy_config.values())) if proxy_config else None
    async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
        yield client