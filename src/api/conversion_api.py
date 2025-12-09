"""
格式转换API
提供API格式转换的核心路由和处理逻辑
"""
import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx
from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from channels.channel_manager import channel_manager, ChannelInfo
from formats.converter_factory import ConverterFactory, convert_request, convert_response
from src.utils.logger import (
    setup_logger, log_structured_error,
    ERROR_TYPE_NETWORK, ERROR_TYPE_AUTH, ERROR_TYPE_RATE_LIMIT, ERROR_TYPE_UPSTREAM_API
)
from src.utils.exceptions import ChannelNotFoundError, ConversionError, APIError, TimeoutError
from src.utils.http_client import get_http_client
from src.core.gateway_config import (
    GatewayConfig, load_gateway_config, save_gateway_config, delete_gateway_config
)

logger = setup_logger("conversion_api")

router = APIRouter()

# 导入已存在的认证函数，避免重复定义
def get_session_user(request: Request):
    """获取会话用户，验证是否已登录"""
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="未登录")
    return True


class ChannelCreateRequest(BaseModel):
    """创建渠道请求"""
    name: str = Field(..., description="渠道名称")
    provider: str = Field(..., description="提供商 (openai/anthropic/gemini)")
    base_url: str = Field(..., description="API基础URL")
    api_key: str = Field(..., description="API密钥")
    custom_key: str = Field(..., description="自定义key，用户调用时使用")
    timeout: int = Field(30, description="超时时间")
    max_retries: int = Field(3, description="最大重试次数")
    # 新增：模型映射（请求模型名 -> 映射模型名）
    models_mapping: Optional[Dict[str, str]] = None
    use_proxy: Optional[bool] = None
    proxy_type: Optional[str] = None
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None


class ChannelUpdateRequest(BaseModel):
    """更新渠道请求"""
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    custom_key: Optional[str] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None
    enabled: Optional[bool] = None
    models_mapping: Optional[Dict[str, str]] = None
    use_proxy: Optional[bool] = None
    proxy_type: Optional[str] = None
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None


async def detect_request_format(request_data: Dict[str, Any], path: str) -> str:
    """检测请求格式"""
    # 基于URL路径检测
    if "/openai/" in path or path.endswith("/chat/completions"):
        return "openai"
    elif "/anthropic/" in path or path.endswith("/messages") or "/v1/messages" in path:
        # Anthropic: /anthropic/ 前缀，或以 /messages 结尾，或包含 /v1/messages（如 /v1/messages/count_tokens）
        return "anthropic"
    elif "/gemini/" in path or "generateContent" in path:
        return "gemini"
    
    # 基于请求数据结构检测
    if "messages" in request_data and "model" in request_data:
        if "system" in request_data:
            return "anthropic"
        else:
            return "openai"
    elif "contents" in request_data:
        return "gemini"
    
    # 默认返回openai格式
    return "openai"



async def forward_request(
    channel: ChannelInfo,
    converted_data: Dict[str, Any],
    headers: Dict[str, str],
    method: str = "POST"
) -> Dict[str, Any]:
    """转发请求到目标API"""
    # 应用模型映射（如果配置）
    try:
        if isinstance(converted_data, dict) and "model" in converted_data and channel.models_mapping:
            original_model = converted_data.get("model")
            mapped = channel.models_mapping.get(original_model)
            if mapped:
                logger.info(f"Applying model mapping for channel {channel.name}: {original_model} -> {mapped}")
                converted_data = {**converted_data, "model": mapped}
            else:
                logger.debug(
                    f"Model mapping not found for '{original_model}'. Available keys: {list(channel.models_mapping.keys())}"
                )
    except Exception as e:
        logger.warning(f"Failed to apply model mapping: {e}")
    # 构建请求URL
    if channel.provider == "openai":
        url = f"{channel.base_url.rstrip('/')}/chat/completions"
        headers["Authorization"] = f"Bearer {channel.api_key}"
    elif channel.provider == "anthropic":
        url = f"{channel.base_url.rstrip('/')}/v1/messages"
        headers["x-api-key"] = channel.api_key
        headers["anthropic-version"] = "2023-06-01"
    elif channel.provider == "gemini":
        # Gemini需要从converted_data中提取模型名称（已在上面应用了映射）
        model = converted_data.get("model")
        if not model:
            raise ValueError("Model name is required for Gemini requests")
        url = f"{channel.base_url.rstrip('/')}/models/{model}:generateContent?key={channel.api_key}"
    else:
        raise ValueError(f"Unsupported provider: {channel.provider}")
    
    # 设置请求头
    headers["Content-Type"] = "application/json"
    
    try:
        async with httpx.AsyncClient(timeout=channel.timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                json=converted_data,
                headers=headers
            )

            if response.status_code == 200:
                return response.json()

            # 根据状态码分类错误类型
            status = response.status_code
            if status in (401, 403):
                error_type = ERROR_TYPE_AUTH
            elif status == 429:
                error_type = ERROR_TYPE_RATE_LIMIT
            else:
                error_type = ERROR_TYPE_UPSTREAM_API

            error_detail = f"API request failed with status {status}: {response.text}"
            log_structured_error(
                logger,
                error_type=error_type,
                request_method=method,
                request_url=url,
                request_headers=headers,
                request_body=converted_data,
                response_status=status,
                response_headers=dict(response.headers),
                response_body=response.text,
                extra={
                    "channel_name": channel.name,
                    "channel_provider": channel.provider,
                    "stage": "forward_request",
                },
            )
            raise APIError(error_detail)

    except httpx.TimeoutException as e:
        log_structured_error(
            logger,
            error_type=ERROR_TYPE_NETWORK,
            exc=e,
            request_method=method,
            request_url=url,
            request_headers=headers,
            request_body=converted_data,
            extra={
                "channel_name": channel.name,
                "channel_provider": channel.provider,
                "timeout": channel.timeout,
                "stage": "forward_request",
            },
        )
        raise TimeoutError(f"Request timeout after {channel.timeout} seconds")
    except APIError:
        raise
    except Exception as e:
        log_structured_error(
            logger,
            error_type=ERROR_TYPE_UPSTREAM_API,
            exc=e,
            request_method=method,
            request_url=url,
            request_headers=headers,
            request_body=converted_data,
            extra={
                "channel_name": channel.name,
                "channel_provider": channel.provider,
                "stage": "forward_request",
            },
        )
        raise APIError(f"Request failed: {e}")


# 渠道管理API
@router.post("/channels")
async def create_channel(request: ChannelCreateRequest, _: bool = Depends(get_session_user)):
    """创建新渠道"""
    
    try:
        channel_id = channel_manager.add_channel(
            name=request.name,
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
            custom_key=request.custom_key,
            timeout=request.timeout,
            max_retries=request.max_retries,
            models_mapping=request.models_mapping,
            use_proxy=request.use_proxy,
            proxy_type=request.proxy_type,
            proxy_host=request.proxy_host,
            proxy_port=request.proxy_port,
            proxy_username=request.proxy_username,
            proxy_password=request.proxy_password
        )
        
        return {
            "success": True,
            "channel_id": channel_id,
            "message": "Channel created successfully"
        }
    except Exception as e:
        logger.error(f"Failed to create channel: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/channels")
async def list_channels(_: bool = Depends(get_session_user)):
    """获取所有渠道"""
    try:
        channels = channel_manager.get_all_channels()
        return {
            "success": True,
            "channels": [
                {
                    "id": channel.id,
                    "name": channel.name,
                    "provider": channel.provider,
                    "base_url": channel.base_url,
                    "api_key": "***" if channel.api_key else None,
                    "custom_key": channel.custom_key,
                    "timeout": getattr(channel, 'timeout', 30),
                    "max_retries": getattr(channel, 'max_retries', 3),
                    "enabled": channel.enabled,
                    "models_mapping": getattr(channel, 'models_mapping', None),
                    # 代理配置
                    "proxy_host": getattr(channel, 'proxy_host', None),
                    "proxy_port": getattr(channel, 'proxy_port', None),
                    "proxy_type": getattr(channel, 'proxy_type', None),
                    "proxy_username": getattr(channel, 'proxy_username', None),
                    "proxy_password": "***" if getattr(channel, 'proxy_password', None) else None,
                    # 时间戳
                    "created_at": channel.created_at,
                    "updated_at": channel.updated_at
                }
                for channel in channels
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list channels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/channels/{channel_id}")
async def get_channel(channel_id: str, _: bool = Depends(get_session_user)):
    """获取特定渠道信息"""
    try:
        channel = channel_manager.get_channel(channel_id)
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")
        
        return {
            "success": True,
            "channel": {
                "id": channel.id,
                "name": channel.name,
                "provider": channel.provider,
                "base_url": channel.base_url,
                "api_key": "***" if channel.api_key else None,
                "custom_key": channel.custom_key,
                "timeout": getattr(channel, 'timeout', 30),
                "max_retries": getattr(channel, 'max_retries', 3),
                "enabled": channel.enabled,
                "models_mapping": getattr(channel, 'models_mapping', None),
                # 代理配置
                "proxy_host": getattr(channel, 'proxy_host', None),
                "proxy_port": getattr(channel, 'proxy_port', None),
                "proxy_type": getattr(channel, 'proxy_type', None),
                "proxy_username": getattr(channel, 'proxy_username', None),
                "proxy_password": getattr(channel, 'proxy_password', None),
                # 时间戳
                "created_at": channel.created_at,
                "updated_at": channel.updated_at
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/channels/{channel_id}")
async def update_channel(channel_id: str, request: ChannelUpdateRequest, _: bool = Depends(get_session_user)):
    """更新渠道信息"""
    try:
        success = channel_manager.update_channel(
            channel_id=channel_id,
            name=request.name,
            base_url=request.base_url,
            api_key=request.api_key,
            custom_key=request.custom_key,
            timeout=request.timeout,
            max_retries=request.max_retries,
            enabled=request.enabled,
            models_mapping=request.models_mapping,
            use_proxy=request.use_proxy,
            proxy_type=request.proxy_type,
            proxy_host=request.proxy_host,
            proxy_port=request.proxy_port,
            proxy_username=request.proxy_username,
            proxy_password=request.proxy_password
        )
        
        if success:
            return {
                "success": True,
                "message": "Channel updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Channel not found")
            
    except ChannelNotFoundError:
        raise HTTPException(status_code=404, detail="Channel not found")
    except Exception as e:
        logger.error(f"Failed to update channel: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/channels/{channel_id}")
async def delete_channel(channel_id: str, _: bool = Depends(get_session_user)):
    """删除渠道"""
    try:
        success = channel_manager.delete_channel(channel_id)
        if success:
            return {
                "success": True,
                "message": "Channel deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Channel not found")
            
    except ChannelNotFoundError:
        raise HTTPException(status_code=404, detail="Channel not found")
    except Exception as e:
        logger.error(f"Failed to delete channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/channels/{channel_id}/test")
async def test_channel(channel_id: str, _: bool = Depends(get_session_user)):
    """测试渠道连接"""
    try:
        result = channel_manager.test_channel_connection(channel_id)
        return {
            "success": True,
            "test_result": result
        }
    except ChannelNotFoundError:
        raise HTTPException(status_code=404, detail="Channel not found")
    except Exception as e:
        logger.error(f"Failed to test channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 格式转换API
@router.post("/openai/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI格式API端点"""
    return await handle_conversion_request(request, "openai")


@router.post("/anthropic/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic格式API端点"""
    return await handle_conversion_request(request, "anthropic")


@router.post("/gemini/v1beta/generateContent")
async def gemini_generate_content(request: Request):
    """Gemini格式API端点"""
    return await handle_conversion_request(request, "gemini")


async def handle_conversion_request(request: Request, target_format: str):
    """处理转换请求的通用逻辑"""
    try:
        # 获取请求数据
        request_data = await request.json()
        headers = dict(request.headers)
        
        # 检测源格式
        source_format = await detect_request_format(request_data, str(request.url.path))
        
        logger.info(f"Request URL path: {request.url.path}")
        logger.info(f"Detected source_format: {source_format}, target_format: {target_format}")
        logger.info(f"Converting request from {source_format} to {target_format}")
        
        # 如果源格式和目标格式相同，直接转发
        if source_format == target_format:
            # 根据目标格式找到合适的渠道
            channels = channel_manager.get_channels_by_provider(target_format)
            if not channels:
                raise HTTPException(
                    status_code=503,
                    detail=f"No available {target_format} channels configured"
                )
            
            # 选择第一个可用渠道
            channel = channels[0]
            
            # 直接转发请求
            response_data = await forward_request(channel, request_data, headers)
            return response_data
        
        # 格式转换
        conversion_result = convert_request(source_format, target_format, request_data, headers)
        
        if not conversion_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Request conversion failed: {conversion_result.error}"
            )
        
        # 找到目标格式的渠道
        channels = channel_manager.get_channels_by_provider(target_format)
        if not channels:
            raise HTTPException(
                status_code=503,
                detail=f"No available {target_format} channels configured"
            )
        
        # 选择第一个可用渠道
        channel = channels[0]
        
        # 转发请求
        response_data = await forward_request(channel, conversion_result.data, headers)
        
        # 转换响应格式
        response_conversion_result = convert_response(source_format, target_format, response_data)
        
        if not response_conversion_result.success:
            logger.warning(f"Response conversion failed: {response_conversion_result.error}")
            # 如果响应转换失败，返回原始响应
            return response_data
        
        return response_conversion_result.data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversion request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/conversion/formats")
async def get_supported_formats():
    """获取支持的格式列表"""
    return {
        "success": True,
        "formats": ConverterFactory.get_supported_formats()
    }


@router.get("/conversion/statistics")
async def get_conversion_statistics():
    """获取转换统计信息"""
    try:
        channel_stats = channel_manager.get_channel_statistics()
        return {
            "success": True,
            "statistics": {
                "channels": channel_stats,
                "supported_formats": ConverterFactory.get_supported_formats()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ProxyTestRequest(BaseModel):
    """代理测试请求"""
    proxy_type: str = Field(..., description="代理类型 (http/https/socks5)", pattern="^(http|https|socks5)$")
    proxy_host: str = Field(..., description="代理地址")
    proxy_port: int = Field(..., description="代理端口", ge=1, le=65535)
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    # 允许自定义测试地址，提高灵活性
    test_urls: Optional[List[str]] = None
    timeout: float = Field(10.0, description="超时时间(秒)", ge=1, le=60)


# 默认测试地址，提供多个备选以提高可靠性
DEFAULT_TEST_URLS = [
    "http://httpbin.org/ip",
    "https://httpbin.org/ip",
    "https://ifconfig.me/all.json", 
    "https://ip.sb/info"
]


def build_proxy_url(request: ProxyTestRequest) -> str:
    """构造代理URL，避免业务对象耦合"""
    auth = ""
    if request.proxy_username and request.proxy_password:
        auth = f"{request.proxy_username}:{request.proxy_password}@"
    return f"{request.proxy_type}://{auth}{request.proxy_host}:{request.proxy_port}"


async def probe_one_url(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    """测试单个URL的连通性"""
    result = {
        "url": url,
        "success": False,
        "response_time": None,
        "external_ip": None,
        "error": None
    }
    
    start_time = time.perf_counter()  # 使用单调时钟
    try:
        response = await client.get(url)
        result["response_time"] = round((time.perf_counter() - start_time) * 1000, 2)  # ms
        
        if response.status_code == 200:
            try:
                data = response.json()
                result["success"] = True
                # 兼容不同API的响应格式
                result["external_ip"] = (
                    data.get("origin") or  # httpbin.org
                    data.get("ip") or      # ip.sb
                    data.get("ip_addr") or # ifconfig.me
                    "Unknown"
                )
            except Exception:
                # 如果不是JSON，至少连接成功了
                result["success"] = True
                result["external_ip"] = "Connected"
        else:
            result["error"] = f"HTTP {response.status_code}"
            
    except httpx.TimeoutException:
        result["error"] = "连接超时"
    except httpx.ProxyError as e:
        result["error"] = f"代理错误: {str(e)}"
    except httpx.TransportError as e:
        result["error"] = f"传输错误: {str(e)}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
    
    return result


@router.post("/test_proxy")
async def test_proxy_connection(request: ProxyTestRequest, _: bool = Depends(get_session_user)):
    """测试代理连通性 - 优化版本：并发测试，减少耦合，细化错误处理"""
    
    # 预检查SOCKS5支持
    if request.proxy_type == "socks5":
        try:
            import socksio
        except ImportError:
            return {
                "success": False,
                "message": "SOCKS5支持未安装",
                "error": "请运行: pip install httpx[socks]"
            }
    
    try:
        # 构造代理URL，避免创建完整的业务对象
        proxy_url = build_proxy_url(request)
        test_urls = request.test_urls or DEFAULT_TEST_URLS
        
        # 一次性创建HTTP客户端，减少连接开销
        async with httpx.AsyncClient(
            proxy=proxy_url,
            timeout=request.timeout,
            follow_redirects=False,
            verify=True,
        ) as client:
            # 并发测试所有URL，提高效率
            tasks = [probe_one_url(client, url) for url in test_urls]
            test_results = await asyncio.gather(*tasks)
        
        # 统计结果
        success_count = sum(1 for result in test_results if result["success"])
        overall_success = success_count == len(test_results)
        
        return {
            "success": overall_success,
            "message": (
                f"代理测试完成: {success_count}/{len(test_results)} 个测试通过" 
                if overall_success 
                else f"代理测试部分失败: 只有 {success_count}/{len(test_results)} 个测试通过"
            ),
            "proxy": {
                "scheme": request.proxy_type,
                "endpoint": f"{request.proxy_host}:{request.proxy_port}",
                "auth": bool(request.proxy_username and request.proxy_password),
            },
            "test_results": test_results,
            "summary": {
                "total_tests": len(test_results),
                "successful_tests": success_count,
                "failed_tests": len(test_results) - success_count,
                "success_rate": round(success_count / len(test_results) * 100, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Proxy test failed: {e}")
        return {
            "success": False,
            "message": "代理测试过程中发生错误",
            "error": str(e)
        }


# Gateway 配置管理 API
class GatewayConfigRequest(BaseModel):
    """Gateway 配置请求"""
    provider: str = Field(..., description="目标提供商 (openai/anthropic/gemini)")
    base_url: str = Field(..., description="目标 API 基础 URL")
    timeout: int = Field(30, description="超时时间（秒）", ge=1, le=600)
    max_retries: int = Field(1, description="最大重试次数", ge=0, le=10)
    model_mapping: Optional[Dict[str, str]] = Field(default=None, description="模型映射")
    enabled: bool = Field(True, description="是否启用 Gateway")


class GatewayConfigResponse(BaseModel):
    """Gateway 配置响应"""
    success: bool
    config: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


@router.get("/gateway/config", response_model=GatewayConfigResponse)
async def get_gateway_config(_: bool = Depends(get_session_user)):
    """获取 Gateway 全局配置"""
    config = load_gateway_config()
    if not config:
        return GatewayConfigResponse(
            success=True,
            config=None,
            message="Gateway 未配置"
        )
    return GatewayConfigResponse(
        success=True,
        config=config.to_dict(),
        message="OK"
    )


@router.put("/gateway/config", response_model=GatewayConfigResponse)
async def upsert_gateway_config(
    request: GatewayConfigRequest,
    _: bool = Depends(get_session_user),
):
    """创建或更新 Gateway 全局配置"""
    try:
        config = GatewayConfig(
            provider=request.provider,
            base_url=request.base_url,
            timeout=request.timeout,
            max_retries=request.max_retries,
            model_mapping=request.model_mapping or {},
            enabled=request.enabled,
        )
        save_gateway_config(config)
        return GatewayConfigResponse(
            success=True,
            config=config.to_dict(),
            message="Gateway 配置已更新",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save gateway config: {e}")
        raise HTTPException(status_code=500, detail="Failed to save gateway config")


@router.delete("/gateway/config", response_model=GatewayConfigResponse)
async def clear_gateway_config(_: bool = Depends(get_session_user)):
    """删除 Gateway 配置"""
    try:
        delete_gateway_config()
        return GatewayConfigResponse(
            success=True,
            config=None,
            message="Gateway 配置已删除",
        )
    except Exception as e:
        logger.error(f"Failed to delete gateway config: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete gateway config")
