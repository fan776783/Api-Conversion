"""
能力检测器
"""
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.utils.config import ChannelConfig, CapabilityTestConfig, ConfigManager
from src.utils.logger import get_logger
from src.utils.exceptions import CapabilityDetectionError, NetworkError, AuthenticationError


class CapabilityStatus(Enum):
    """能力状态枚举"""
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class CapabilityResult:
    """能力检测结果"""
    capability: str
    status: CapabilityStatus
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ChannelCapabilities:
    """渠道能力信息"""
    provider: str
    base_url: str
    models: List[str]
    capabilities: Dict[str, CapabilityResult]
    detection_time: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "models": self.models,
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "detection_time": self.detection_time
        }


class BaseCapabilityDetector(ABC):
    """基础能力检测器"""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.logger = get_logger(f"detector.{config.provider}")
        self.config_manager = ConfigManager()
        self.console = Console()
        self.target_model = None  # 指定要检测的模型
        self.debug_mode = False  # 调试模式
        self.progress_callback = None  # 进度回调函数
    
    @abstractmethod
    async def detect_models(self) -> List[str]:
        """检测支持的模型"""
        pass
    
    @abstractmethod
    async def test_capability(self, capability_config: CapabilityTestConfig) -> CapabilityResult:
        """测试单个能力"""
        pass
    
    async def detect_all_capabilities(self) -> ChannelCapabilities:
        """检测所有能力"""
        from datetime import datetime
        
        self.logger.info(f"Starting capability detection for {self.config.provider}")
        
        # 检测模型（如果已设置目标模型则跳过）
        if hasattr(self, 'target_model') and self.target_model:
            models = [self.target_model]
            self.logger.info(f"Using target model: {self.target_model}")
        else:
            try:
                models = await self.detect_models()
                self.logger.info(f"Detected {len(models)} models")
            except Exception as e:
                self.logger.error(f"Failed to detect models: {e}")
                models = []
        
        # 检测能力
        capabilities = {}
        capability_configs = self.config_manager.get_all_capabilities()
        total_capabilities = len(capability_configs)
        
        # 初始化进度
        if self.progress_callback:
            await self.progress_callback({
                "status": "running",
                "progress": 0,
                "current_capability": None,
                "completed_capabilities": [],
                "total_capabilities": total_capabilities
            })
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("检测能力中...", total=total_capabilities)
            completed_count = 0
            
            for name, config in capability_configs.items():
                progress.update(task, description=f"检测 {config.description}")
                
                # 更新当前检测的能力
                if self.progress_callback:
                    await self.progress_callback({
                        "status": "running",
                        "progress": int((completed_count / total_capabilities) * 100),
                        "current_capability": config.description,
                        "completed_capabilities": list(capabilities.keys()),
                        "total_capabilities": total_capabilities
                    })
                
                try:
                    result = await self.test_capability(config)
                    capabilities[name] = result
                    self.logger.info(f"Capability {name}: {result.status.value}")
                except Exception as e:
                    self.logger.error(f"Failed to test capability {name}: {e}")
                    capabilities[name] = CapabilityResult(
                        capability=name,
                        status=CapabilityStatus.ERROR,
                        error=str(e)
                    )
                
                completed_count += 1
                progress.advance(task)
                
                # 更新完成进度
                if self.progress_callback:
                    await self.progress_callback({
                        "status": "running",
                        "progress": int((completed_count / total_capabilities) * 100),
                        "current_capability": None,
                        "completed_capabilities": list(capabilities.keys()),
                        "total_capabilities": total_capabilities
                    })
        
        return ChannelCapabilities(
            provider=self.config.provider,
            base_url=self.config.base_url,
            models=models,
            capabilities=capabilities,
            detection_time=datetime.now().isoformat()
        )
    
    async def detect_selected_capabilities(self, selected_capabilities: List[str]) -> ChannelCapabilities:
        """检测选定的能力"""
        from datetime import datetime
        
        self.logger.info(f"Starting selected capability detection for {self.config.provider}")
        
        # 检测模型（如果已设置目标模型则跳过）
        if hasattr(self, 'target_model') and self.target_model:
            models = [self.target_model]
            self.logger.info(f"Using target model: {self.target_model}")
        else:
            try:
                models = await self.detect_models()
                self.logger.info(f"Detected {len(models)} models")
            except Exception as e:
                self.logger.error(f"Failed to detect models: {e}")
                models = []
        
        # 获取所有能力配置
        all_capability_configs = self.config_manager.get_all_capabilities()
        
        # 过滤选定的能力
        capability_configs = {}
        for cap_name in selected_capabilities:
            if cap_name in all_capability_configs:
                capability_configs[cap_name] = all_capability_configs[cap_name]
            else:
                self.logger.warning(f"Unknown capability: {cap_name}")
        
        # 检测能力
        capabilities = {}
        total_capabilities = len(capability_configs)
        
        # 初始化进度
        if self.progress_callback:
            await self.progress_callback({
                "status": "running",
                "progress": 0,
                "current_capability": None,
                "completed_capabilities": [],
                "total_capabilities": total_capabilities
            })
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("检测能力中...", total=total_capabilities)
            completed_count = 0
            
            for name, config in capability_configs.items():
                progress.update(task, description=f"检测 {config.description}")
                
                # 更新当前检测的能力
                if self.progress_callback:
                    await self.progress_callback({
                        "status": "running",
                        "progress": int((completed_count / total_capabilities) * 100),
                        "current_capability": config.description,
                        "completed_capabilities": list(capabilities.keys()),
                        "total_capabilities": total_capabilities
                    })
                
                try:
                    result = await self.test_capability(config)
                    capabilities[name] = result
                    self.logger.info(f"Capability {name}: {result.status.value}")
                except Exception as e:
                    self.logger.error(f"Failed to test capability {name}: {e}")
                    capabilities[name] = CapabilityResult(
                        capability=name,
                        status=CapabilityStatus.ERROR,
                        error=str(e)
                    )
                
                completed_count += 1
                progress.advance(task)
                
                # 更新完成进度
                if self.progress_callback:
                    await self.progress_callback({
                        "status": "running",
                        "progress": int((completed_count / total_capabilities) * 100),
                        "current_capability": None,
                        "completed_capabilities": list(capabilities.keys()),
                        "total_capabilities": total_capabilities
                    })
        
        return ChannelCapabilities(
            provider=self.config.provider,
            base_url=self.config.base_url,
            models=models,
            capabilities=capabilities,
            detection_time=datetime.now().isoformat()
        )
    
    async def _make_request(
        self, 
        method: str, 
        url: str, 
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        show_details: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """发送HTTP请求"""
        default_headers = {
            "Content-Type": "application/json"
        }
        
        if headers:
            default_headers.update(headers)
        
        # 展示请求详情（使用实例的debug_mode或方法参数）
        if show_details or getattr(self, 'debug_mode', False):
            self._show_request_details(method, url, data, default_headers)
        
        # 配置代理设置
        proxy = None
        if getattr(self.config, 'use_proxy', False) and self.config.proxy_host and self.config.proxy_port:
            proxy = f"{self.config.proxy_type}://"
            if self.config.proxy_username and self.config.proxy_password:
                proxy += f"{self.config.proxy_username}:{self.config.proxy_password}@"
            proxy += f"{self.config.proxy_host}:{self.config.proxy_port}"

        try:
            async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=default_headers)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data, headers=default_headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # 尝试解析JSON响应
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = {"text": response.text}
                
                # 展示响应详情（使用实例的debug_mode或方法参数）
                if show_details or getattr(self, 'debug_mode', False):
                    self._show_response_details(response.status_code, response_data, response.headers)
                
                return response.status_code, response_data
                
        except httpx.TimeoutException:
            raise NetworkError(f"Request timeout for {url}")
        except httpx.ConnectError:
            raise NetworkError(f"Failed to connect to {url}")
        except Exception as e:
            raise NetworkError(f"Network error: {e}")
    
    def _check_authentication_error(self, status_code: int, response_data: Dict[str, Any]) -> None:
        """检查认证错误"""
        if status_code == 401:
            raise AuthenticationError("Invalid API key or authentication failed")
        elif status_code == 403:
            raise AuthenticationError("Access forbidden - check API key permissions")
    
    def _extract_error_message(self, response_data: Dict[str, Any]) -> str:
        """提取错误信息"""
        # 常见的错误字段
        error_fields = ["error", "message", "detail", "details"]
        
        for field in error_fields:
            if field in response_data:
                error_info = response_data[field]
                if isinstance(error_info, dict):
                    return error_info.get("message", str(error_info))
                return str(error_info)
        
        return "Unknown error"
    
    def _show_request_details(self, method: str, url: str, data: Optional[Dict[str, Any]], headers: Dict[str, str]):
        """记录请求详情（仅在调试模式下）"""
        if not getattr(self, 'debug_mode', False):
            return

        import json
        # 只记录到日志文件，不在控制台显示
        self.logger.debug(f"HTTP Request: {method.upper()} {url}")
        if data:
            from src.utils.security import safe_log_request
            self.logger.debug(f"Request Body: {safe_log_request(data)}")
    
    def _show_response_details(self, status_code: int, data: Dict[str, Any], headers):
        """记录响应详情（仅在调试模式下）"""
        if not getattr(self, 'debug_mode', False):
            return

        import json
        # 只记录到日志文件，不在控制台显示
        self.logger.debug(f"HTTP Response: Status {status_code}")
        if data:
            from src.utils.security import safe_log_response
            self.logger.debug(f"Response Body: {safe_log_response(data)}")


class CapabilityDetectorFactory:
    """能力检测器工厂"""
    
    _detectors = {}
    
    @classmethod
    def register(cls, provider: str, detector_class):
        """注册检测器"""
        cls._detectors[provider] = detector_class
    
    @classmethod
    def create(cls, config: ChannelConfig) -> BaseCapabilityDetector:
        """创建检测器"""
        if config.provider not in cls._detectors:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        return cls._detectors[config.provider](config)
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """获取支持的提供商"""
        return list(cls._detectors.keys())