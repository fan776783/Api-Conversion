"""
统一API端点
支持通过自定义key调用不同的AI服务，自动进行格式转换
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode, parse_qsl
import httpx
from fastapi import APIRouter, HTTPException, Request, Response, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from channels.channel_manager import channel_manager, ChannelInfo
from formats.converter_factory import ConverterFactory, convert_request, convert_response, convert_streaming_chunk
from formats.base_converter import ConversionResult
from utils.security import mask_api_key, safe_log_request, safe_log_response
from src.utils.logger import (
    setup_logger, log_structured_error, log_request_entry,
    ERROR_TYPE_NETWORK, ERROR_TYPE_AUTH, ERROR_TYPE_RATE_LIMIT,
    ERROR_TYPE_CONVERSION, ERROR_TYPE_UPSTREAM_API
)
from src.utils.exceptions import ChannelNotFoundError, ConversionError, APIError, TimeoutError
from src.utils.http_client import get_http_client
from src.utils.env_config import env_config
from src.core.gateway_config import load_gateway_config
from api.conversion_api import detect_request_format

logger = setup_logger("unified_api")

router = APIRouter()


def _extract_max_tokens(source_format: str, request_data: Dict[str, Any]) -> Optional[int]:
    """从请求中提取 max_tokens 值"""
    if not isinstance(request_data, dict):
        return None
    # OpenAI / Anthropic: max_tokens
    if "max_tokens" in request_data:
        try:
            return int(request_data["max_tokens"])
        except (TypeError, ValueError):
            return None
    # Gemini: generationConfig.maxOutputTokens
    if source_format == "gemini":
        config = request_data.get("generationConfig")
        if isinstance(config, dict) and "maxOutputTokens" in config:
            try:
                return int(config["maxOutputTokens"])
            except (TypeError, ValueError):
                return None
    return None


def _parse_upstream_error(provider: str, status_code: int, body: str) -> str:
    """解析上游错误，返回友好的错误信息"""
    try:
        data = json.loads(body) if body else {}
    except Exception:
        data = {}

    # 尝试从常见结构中提取错误信息
    error_msg = None
    if isinstance(data, dict):
        error_obj = data.get("error")
        if isinstance(error_obj, dict):
            error_msg = error_obj.get("message") or error_obj.get("description")
            error_type = error_obj.get("type")
            code = error_obj.get("code")
            if error_msg and (error_type or code):
                meta = []
                if error_type:
                    meta.append(f"type={error_type}")
                if code:
                    meta.append(f"code={code}")
                error_msg = f"{error_msg} ({', '.join(meta)})"
        elif isinstance(error_obj, str):
            error_msg = error_obj
        if not error_msg:
            error_msg = data.get("message") or data.get("detail")

    prefix = f"{provider} API error ({status_code})"
    if error_msg:
        return f"{prefix}: {error_msg}"

    # 截断过长的原始响应
    if body and len(body) > 300:
        body = body[:300] + "..."
    return f"{prefix}: {body}" if body else prefix


async def fetch_models_from_channel_for_format(channel: ChannelInfo, target_format: str) -> List[Dict[str, Any]]:
    """从目标渠道获取模型列表并转换为指定格式"""
    try:
        logger.info(f"Fetching models from {channel.provider} channel for {target_format} format")
        
        # 先获取原始模型数据
        raw_models = await fetch_raw_models_from_channel(channel)
        
        # 根据目标格式转换
        if target_format == "openai":
            return convert_models_to_openai_format(raw_models, channel.provider)
        elif target_format == "anthropic":
            return convert_models_to_anthropic_format(raw_models, channel.provider)
        elif target_format == "gemini":
            return convert_models_to_gemini_format(raw_models, channel.provider)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported target format: {target_format}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch models for {target_format} format: {e}")
        logger.exception("Full traceback:")
        return []




async def fetch_raw_models_from_channel(channel: ChannelInfo) -> List[Dict[str, Any]]:
    """从目标渠道获取原始模型数据"""
    try:
        logger.info(f"Fetching raw models from {channel.provider} channel: {channel.name}")
        logger.debug(f"Channel details - Base URL: {channel.base_url}, API Key: {mask_api_key(channel.api_key)}")
        
        if channel.provider == "openai":
            raw_models = await fetch_openai_raw_models(channel)
        elif channel.provider == "anthropic":
            raw_models = await fetch_anthropic_raw_models(channel) 
        elif channel.provider == "gemini":
            raw_models = await fetch_gemini_raw_models(channel)
        else:
            logger.error(f"Unknown provider: {channel.provider}")
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {channel.provider}")
        
        logger.info(f"Successfully fetched {len(raw_models)} raw models from {channel.provider} channel")
        return raw_models
            
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"Failed to fetch raw models from {channel.provider}: {e}")
        logger.exception("Full traceback:")  # 记录完整堆栈跟踪
        # 返回空列表而不是默认模型
        logger.warning(f"Returning empty model list due to API failure")
        return []


async def fetch_openai_raw_models(channel: ChannelInfo) -> List[Dict[str, Any]]:
    """获取OpenAI原始模型数据"""
    logger.info(f"Calling OpenAI models API: {channel.base_url}")
    
    url = f"{channel.base_url.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {channel.api_key}",
        "Content-Type": "application/json"
    }
    
    async with get_http_client(channel, timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        
        if response.status_code != 200:
            error_msg = f"OpenAI API returned {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        data = response.json()
        models = data.get("data", [])
        
        if not models:
            logger.warning("OpenAI API returned empty model list")
        
        logger.info(f"Retrieved {len(models)} models from OpenAI API")
        return models


async def fetch_anthropic_raw_models(channel: ChannelInfo) -> List[Dict[str, Any]]:
    """获取Anthropic原始模型数据"""
    logger.info(f"Calling Anthropic models API: {channel.base_url}")
    
    url = f"{channel.base_url.rstrip('/')}/v1/models"
    headers = {
        "x-api-key": channel.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    async with get_http_client(channel, timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        
        if response.status_code != 200:
            error_msg = f"Anthropic API returned {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        data = response.json()
        models = data.get("data", [])
        
        if not models:
            logger.warning("Anthropic API returned empty model list")
        
        logger.info(f"Retrieved {len(models)} models from Anthropic API")
        return models


async def fetch_gemini_raw_models(channel: ChannelInfo) -> List[Dict[str, Any]]:
    """获取Gemini原始模型数据"""
    logger.info(f"Calling Gemini models API: {channel.base_url}")
    
    url = f"{channel.base_url.rstrip('/')}/models"
    params = {"key": channel.api_key}
    
    async with get_http_client(channel, timeout=30.0) as client:
        response = await client.get(url, params=params)
        
        if response.status_code != 200:
            error_msg = f"Gemini API returned {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        data = response.json()
        models = data.get("models", [])
        
        if not models:
            logger.warning("Gemini API returned empty model list")
        
        logger.info(f"Retrieved {len(models)} models from Gemini API")
        return models


def convert_models_to_openai_format(raw_models: List[Dict[str, Any]], source_provider: str) -> List[Dict[str, Any]]:
    """将原始模型数据转换为OpenAI格式"""
    models = []
    current_time = int(time.time())
    
    for model in raw_models:
        if source_provider == "openai":
            # OpenAI格式直接返回
            models.append(model)
        elif source_provider == "anthropic":
            # Anthropic格式转换
            model_id = model.get("id", "")
            created_at = model.get("created_at", "")
            
            # 转换创建时间为timestamp
            created_timestamp = current_time
            if created_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_timestamp = int(dt.timestamp())
                except (ValueError, AttributeError):
                    pass
            
            models.append({
                "id": model_id,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "anthropic"
            })
        elif source_provider == "gemini":
            # Gemini格式转换
            model_name = model.get("name", "")
            # 移除 "models/" 前缀
            if model_name.startswith("models/"):
                model_name = model_name[7:]
            
            # 只包含生成模型，过滤掉嵌入模型等
            supported_methods = model.get("supportedGenerationMethods", [])
            if "generateContent" in supported_methods:
                models.append({
                    "id": model_name,
                    "object": "model",
                    "created": current_time,
                    "owned_by": "google"
                })
    
    return models


def convert_models_to_anthropic_format(raw_models: List[Dict[str, Any]], source_provider: str) -> List[Dict[str, Any]]:
    """将原始模型数据转换为Anthropic格式"""
    models = []
    
    for model in raw_models:
        if source_provider == "anthropic":
            # Anthropic格式直接返回
            models.append(model)
        elif source_provider == "openai":
            # OpenAI格式转换
            models.append({
                "type": "model",
                "id": model.get("id", ""),
                "display_name": model.get("id", ""),
                "created_at": model.get("created") and datetime.fromtimestamp(model["created"]).isoformat() + "Z",
            })
        elif source_provider == "gemini":
            # Gemini格式转换  
            model_name = model.get("name", "")
            if model_name.startswith("models/"):
                model_name = model_name[7:]
            
            supported_methods = model.get("supportedGenerationMethods", [])
            if "generateContent" in supported_methods:
                models.append({
                    "type": "model",
                    "id": model_name,
                    "display_name": model.get("displayName", model_name),
                    "created_at": datetime.now().isoformat() + "Z",
                })
    
    return models


def convert_models_to_gemini_format(raw_models: List[Dict[str, Any]], source_provider: str) -> List[Dict[str, Any]]:
    """将原始模型数据转换为Gemini格式（极简版，只包含name字段）"""
    models = []
    
    for model in raw_models:
        if source_provider == "gemini":
            # Gemini格式，只保留name
            models.append({
                "name": model.get("name", f"models/{model.get('id', '')}")
            })
        elif source_provider == "openai":
            # OpenAI格式转换
            models.append({
                "name": f"models/{model.get('id', '')}"
            })
        elif source_provider == "anthropic":
            # Anthropic格式转换
            models.append({
                "name": f"models/{model.get('id', '')}"
            })
    
    return models



def extract_openai_api_key(authorization: Optional[str] = Header(None)) -> str:
    """从OpenAI格式的Authorization header中提取API key"""
    logger.debug(f"OpenAI auth - Received authorization header: {mask_api_key(authorization) if authorization else 'None'}")
    
    if not authorization:
        logger.error("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    if not authorization.startswith("Bearer "):
        logger.error(f"Invalid Authorization header format: {mask_api_key(authorization)}")
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    
    api_key = authorization[7:]  # 移除 "Bearer " 前缀
    logger.debug(f"Extracted OpenAI API key: {mask_api_key(api_key)}")
    return api_key


def extract_anthropic_api_key(x_api_key: Optional[str] = Header(None, alias="x-api-key"), authorization: Optional[str] = Header(None, alias="authorization")) -> str:
    """从Anthropic格式的x-api-key header或Authorization header中提取API key"""
    logger.debug(f"Anthropic auth - Received x-api-key header: {mask_api_key(x_api_key) if x_api_key else 'None'}")
    logger.debug(f"Anthropic auth - Received authorization header: {mask_api_key(authorization) if authorization else 'None'}")
    
    # 首先尝试从x-api-key获取token
    if x_api_key:
        logger.info(f"Extracted Anthropic API key from x-api-key: {mask_api_key(x_api_key)}")
        return x_api_key
    
    # 如果x-api-key不存在，尝试从Authorization header获取
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]  # 移除 "Bearer " 前缀
        logger.info(f"Extracted Anthropic API key from Authorization header: {mask_api_key(api_key)}")
        return api_key
    
    # 如果两种方式都无法获取token，则报错
    logger.error("Missing authentication: neither x-api-key nor valid Authorization header found")
    error_detail = "Missing authentication. For Anthropic API format, please provide your API key either in the x-api-key header or as 'Bearer TOKEN' in the Authorization header."
    raise HTTPException(status_code=401, detail=error_detail)


def extract_gemini_api_key(request: Request) -> str:
    """从Gemini格式的URL参数或header中提取API key"""
    logger.info(f"Gemini auth - Request URL: {request.url}")
    logger.info(f"Gemini auth - Query params: {dict(request.query_params)}")
    logger.info(f"Gemini auth - Headers: {dict(request.headers)}")
    
    # Gemini API支持多种认证方式，按优先级检查：
    # 1. URL参数 ?key=your_api_key
    api_key = request.query_params.get("key")
    if api_key:
        logger.debug(f"Gemini auth - Extracted API key from URL parameter: {mask_api_key(api_key)}")
        return api_key
    
    # 2. Google官方SDK使用的 x-goog-api-key header
    x_goog_api_key = request.headers.get("x-goog-api-key")
    if x_goog_api_key:
        logger.debug(f"Gemini auth - Extracted API key from x-goog-api-key header: {mask_api_key(x_goog_api_key)}")
        return x_goog_api_key
    
    # 3. 标准的Authorization Bearer header
    authorization = request.headers.get("authorization")
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
        logger.debug(f"Gemini auth - Extracted API key from Authorization header: {mask_api_key(api_key)}")
        return api_key
    
    logger.error("Missing API key in URL parameter, x-goog-api-key header, and Authorization header")
    raise HTTPException(status_code=401, detail="Missing API key")


async def forward_request_to_channel(
    channel: ChannelInfo,
    request_data: Dict[str, Any],
    source_format: str,
    headers: Optional[Dict[str, str]] = None
):
    """转发请求到目标渠道（统一处理流式和非流式）"""
    # 1. 检查是否为同格式透传 - 但对于Anthropic需要特殊处理图片排序
    if source_format == channel.provider:
        # 对于Anthropic格式，即使是透传也需要应用图片优先的最佳实践
        if channel.provider == "anthropic":
            # 检查是否包含图片内容
            has_images = False
            messages = request_data.get("messages", [])
            for message in messages:
                if isinstance(message.get("content"), list):
                    for content in message["content"]:
                        if content.get("type") == "image":
                            has_images = True
                            break
                if has_images:
                    break
            
            if has_images:
                logger.info(f"Anthropic-to-Anthropic with images detected, applying image ordering best practice")
                # 强制进行转换以应用图片排序最佳实践
                conversion_result = convert_request(source_format, channel.provider, request_data, headers)
            else:
                logger.info(f"Anthropic-to-Anthropic without images, using passthrough")
                conversion_result = ConversionResult(success=True, data=request_data)
        else:
            logger.info(f"Same format detected, skipping request conversion: {source_format} -> {channel.provider}")
            # For Gemini passthrough, we need to remove the internal stream field 
            # because Gemini API doesn't accept it in the request body
            if channel.provider == "gemini" and request_data.get("stream"):
                passthrough_data = request_data.copy()
                passthrough_data.pop("stream")
                conversion_result = ConversionResult(success=True, data=passthrough_data)
            else:
                conversion_result = ConversionResult(success=True, data=request_data)
    else:
        # 转换请求格式
        conversion_result = convert_request(source_format, channel.provider, request_data, headers)
        
        if not conversion_result.success:
            raise ConversionError(f"Request conversion failed: {conversion_result.error}")
    
    # 在构建URL与发送前，按渠道配置应用模型映射（仅影响下游请求，不改变原始request_data，确保客户端看到原始模型名）
    mapped_model = None
    try:
        if channel.models_mapping and isinstance(request_data, dict):
            original_model = request_data.get("model")
            if original_model:
                mapped_model = channel.models_mapping.get(original_model)
                if mapped_model:
                    logger.info(f"Applying model mapping for channel {channel.name}: {original_model} -> {mapped_model}")
                    # 确保发送到下游的请求体中也使用映射后的模型
                    if isinstance(conversion_result.data, dict):
                        conversion_result.data = {**conversion_result.data, "model": mapped_model}
                else:
                    logger.debug(
                        f"Model mapping not found for '{original_model}'. Available keys: {list(channel.models_mapping.keys())}"
                    )
    except Exception as e:
        logger.warning(f"Failed to apply model mapping: {e}")

    # 2. 统一构建目标API的URL和headers（支持所有功能组合）
    url = None  # 防御性初始化
    target_headers = {"Content-Type": "application/json"}
    # Always use original request_data for stream detection, since conversion_result.data
    # may have stream field removed (especially for Gemini passthrough)
    is_streaming = request_data.get("stream", False)
    
    if channel.provider == "openai":
        url = f"{channel.base_url.rstrip('/')}/chat/completions"
        target_headers["Authorization"] = f"Bearer {channel.api_key}"
    elif channel.provider == "anthropic":
        url = f"{channel.base_url.rstrip('/')}/v1/messages"
        target_headers["x-api-key"] = channel.api_key
        target_headers["anthropic-version"] = "2023-06-01"
    elif channel.provider == "gemini":
        # 对Gemini而言，模型也会体现在URL中，这里优先使用映射后的模型
        model = mapped_model or request_data.get("model")
        if not model:
            raise ValueError("Model is required for Gemini API requests")
        
        # Gemini根据流式参数选择不同端点
        if is_streaming:
            url = f"{channel.base_url.rstrip('/')}/models/{model}:streamGenerateContent?alt=sse&key={channel.api_key}"
            target_headers["Accept"] = "text/event-stream"
        else:
            url = f"{channel.base_url.rstrip('/')}/models/{model}:generateContent?key={channel.api_key}"
    else:
        raise ValueError(f"Unsupported provider: {channel.provider}")
    
    # 3. 统一请求处理
    try:
        logger.debug(f"Sending {'streaming' if is_streaming else 'non-streaming'} request to {channel.provider}: {url}")
        logger.debug(f"Request data: {safe_log_request(conversion_result.data)}")
        
        # 检查渠道是否配置了代理
        if getattr(channel, 'use_proxy', False):
            proxy_host = getattr(channel, 'proxy_host', None)
            proxy_port = getattr(channel, 'proxy_port', None)
            logger.info(f"PROXY CHECK: Channel {channel.name} has proxy enabled - {proxy_host}:{proxy_port}")
        else:
            logger.info(f"PROXY CHECK: Channel {channel.name} has no proxy configured")
        
        if is_streaming:
            # 流式请求处理 - 创建独立的生成器函数
            async def stream_generator():
                try:
                    async with get_http_client(channel, timeout=channel.timeout) as client:
                        async with client.stream(
                            "POST",
                            url=url,
                            json=conversion_result.data,
                            headers=target_headers
                        ) as response:
                            async for chunk in handle_streaming_response(response, channel, request_data, source_format):
                                yield chunk
                except httpx.TimeoutException as e:
                    log_structured_error(
                        logger,
                        error_type=ERROR_TYPE_NETWORK,
                        exc=e,
                        request_method="POST",
                        request_url=url,
                        request_headers=headers,
                        request_body=conversion_result.data,
                        extra={
                            "channel_name": channel.name,
                            "channel_provider": channel.provider,
                            "is_streaming": True,
                            "stage": "stream_generator",
                            "timeout": channel.timeout,
                        },
                    )
                    raise TimeoutError(f"Streaming request timeout after {channel.timeout} seconds")
                except HTTPException:
                    # 透传 HTTPException，保持原有状态码
                    raise
                except Exception as e:
                    error_msg = f"Streaming request failed: {str(e) if e else 'Unknown error'}"
                    log_structured_error(
                        logger,
                        error_type=ERROR_TYPE_UPSTREAM_API,
                        exc=e,
                        request_method="POST",
                        request_url=url,
                        request_headers=headers,
                        request_body=conversion_result.data,
                        extra={
                            "channel_name": channel.name,
                            "channel_provider": channel.provider,
                            "is_streaming": True,
                            "stage": "stream_generator",
                        },
                    )
                    raise APIError(error_msg)
            
            return stream_generator()
        else:
            # 统一处理非流式请求：发送转换后的请求到目标渠道
            logger.debug(f"Sending non-streaming request to {channel.provider}: {url}")
            async with get_http_client(channel, timeout=channel.timeout) as client:
                response = await client.post(
                    url=url,
                    json=conversion_result.data,
                    headers=target_headers
                )
                result = handle_non_streaming_response(response, channel, request_data, source_format)
                return result
                    
    except httpx.TimeoutException as e:
        log_structured_error(
            logger,
            error_type=ERROR_TYPE_NETWORK,
            exc=e,
            request_method="POST",
            request_url=url,
            request_headers=headers,
            request_body=conversion_result.data,
            extra={
                "channel_name": channel.name,
                "channel_provider": channel.provider,
                "is_streaming": False,
                "stage": "forward_request_to_channel",
                "timeout": channel.timeout,
            },
        )
        raise TimeoutError(f"Non-streaming request timeout after {channel.timeout} seconds")
    except HTTPException:
        # 透传 HTTPException，保持原有状态码
        raise
    except Exception as e:
        log_structured_error(
            logger,
            error_type=ERROR_TYPE_UPSTREAM_API,
            exc=e,
            request_method="POST",
            request_url=url,
            request_headers=headers,
            request_body=conversion_result.data,
            extra={
                "channel_name": channel.name,
                "channel_provider": channel.provider,
                "is_streaming": False,
                "stage": "forward_request_to_channel",
            },
        )
        raise APIError(f"Non-streaming request failed: {e}")


async def handle_streaming_response(response, channel, request_data, source_format):
    """处理流式响应"""
    logger.info(f"STREAMING RESPONSE: channel.provider='{channel.provider}', source_format='{source_format}', status={response.status_code}")
    logger.debug(f"Received streaming response from {channel.provider}: status={response.status_code}")
    
    if response.status_code == 200:
        # 流式处理响应
        logger.debug("Starting to process streaming response")
        chunk_count = 0
        
        # 根据客户端期望的格式选择合适的结束标记
        if source_format == "openai":
            end_marker = "data: [DONE]\n\n"
        elif source_format == "gemini":
            # Gemini不需要特殊的结束标记，最后一个chunk包含finishReason即可
            end_marker = ""
        elif source_format == "anthropic":
            # Anthropic使用event: message_stop
            end_marker = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        else:
            end_marker = "data: [DONE]\n\n"

        # For same-format passthrough, we need to preserve the complete SSE structure
        if channel.provider == source_format:
            logger.info(f"PASSTHROUGH MODE ACTIVATED: {channel.provider} -> {source_format}")
            logger.info(f"PASSTHROUGH: Response status = {response.status_code}")
            logger.info(f"PASSTHROUGH: Response headers = {dict(response.headers)}")
            
            # Direct passthrough using aiter_bytes to preserve exact formatting
            # 使用字节流传输以保持原始格式不变
            try:
                async for chunk in response.aiter_bytes():
                    if chunk:  # 只传输非空chunk
                        yield chunk.decode('utf-8')
                            
                logger.info(f"PASSTHROUGH COMPLETED")
            except Exception as e:
                logger.error(f"PASSTHROUGH ERROR: {e}")
                raise
            return

        async for line in response.aiter_lines():
            # 记录所有接收到的行用于调试
            logger.debug(f"Received SSE line: '{line}'")
            
            # 只处理以 "data: " 开头的行，其余 SSE 行（如 event: keep-alive）直接忽略
            if not line.startswith("data: "):
                # 记录被忽略的行，特别关注思考模型可能的特殊格式
                if line.strip():  # 只记录非空行
                    logger.debug(f"Ignored non-data SSE line: '{line}'")
                continue

            data_content = line[6:]  # 移除 "data: " 前缀
            chunk_count += 1
            logger.debug(f"RAW CHUNK {chunk_count}: '{data_content}'")  # 详细记录原始数据

            # 处理结束哨兵或空数据 - 必须在JSON解析之前检查
            if data_content.strip() in ("[DONE]", ""):
                logger.info(f"Stream ended with marker: '{data_content.strip()}'")
                logger.info(f"Sending end_marker to client: '{end_marker}'")
                if end_marker:  # 只有非空的end_marker才发送
                    yield end_marker
                break
            
            try:
                # 解析JSON数据
                chunk_data = json.loads(data_content)
                logger.debug(f"Parsed chunk data: {chunk_data}")
                
                # 通用的chunk处理逻辑：检查是否有内容和结束标记
                # 这里不应该假设特定的格式结构，让转换器来处理格式差异
                has_content = False
                is_finish_chunk = False
                
                # 简单检测是否包含内容或结束标记，具体格式由转换器处理
                if chunk_data:
                    # 检测不同格式的结束标记
                    if isinstance(chunk_data, dict):
                        # OpenAI格式检测
                        if "choices" in chunk_data and chunk_data["choices"]:
                            choice = chunk_data["choices"][0]
                            if choice.get("finish_reason"):
                                is_finish_chunk = True
                            if choice.get("delta", {}).get("content"):
                                has_content = True
                            elif choice.get("delta", {}).get("tool_calls"):
                                # 检查tool_calls是否有效，避免undefined错误
                                tool_calls = choice.get("delta", {}).get("tool_calls", [])
                                if tool_calls and any(tc and tc.get("function") for tc in tool_calls):
                                    has_content = True  # 工具调用也算作有内容
                        
                        # Gemini格式检测 - 修复：允许同时有内容和结束标记
                        elif "candidates" in chunk_data and chunk_data["candidates"]:
                            candidate = chunk_data["candidates"][0]
                            # 检测是否有内容（文本或工具调用）
                            if candidate.get("content", {}).get("parts"):
                                parts = candidate["content"]["parts"]
                                # 检查是否有文本内容或工具调用
                                if any("text" in part or "functionCall" in part for part in parts):
                                    has_content = True
                            # 检测是否是结束chunk
                            if candidate.get("finishReason"):
                                is_finish_chunk = True
                        
                        # Anthropic格式检测 - 精确匹配需要处理的事件类型
                        elif chunk_data.get("type") == "content_block_delta":
                            # 文本或工具参数增量，需要处理
                            has_content = True
                        elif chunk_data.get("type") == "content_block_start":
                            # content_block_start标志内容块开始，包含文本或工具调用
                            content_block = chunk_data.get("content_block", {})
                            block_type = content_block.get("type")
                            if block_type in ["tool_use", "text"]:
                                has_content = True
                        elif chunk_data.get("type") == "content_block_stop":
                            # content_block_stop标志工具调用完成，需要处理
                            has_content = True
                        elif chunk_data.get("type") == "message_delta":
                            # message_delta包含stop_reason等结束信息
                            delta = chunk_data.get("delta", {})
                            if "stop_reason" in delta:
                                has_content = True
                        elif chunk_data.get("type") == "message_stop":
                            # message_stop标志流结束
                            is_finish_chunk = True
                        # 明确排除不需要处理的事件类型
                        elif chunk_data.get("type") in ["message_start"]:
                            # message_start只是流开始标记，不包含实际内容
                            has_content = False
                        # 其他未知类型默认不处理
                
                logger.debug(f"Chunk {chunk_count} analysis: has_content={has_content}, is_finish_chunk={is_finish_chunk}")
                
                # 如果有内容，转换并发送内容chunk（不管是否也是结束chunk）
                if has_content:
                    original_model = request_data.get("model")
                    # Fix parameter order: source_format=provider, target_format=client_format
                    logger.debug(f"Calling convert_streaming_chunk: source={channel.provider}, target={source_format}")
                    try:
                        response_conversion = convert_streaming_chunk(channel.provider, source_format, chunk_data, original_model)
                        logger.debug(f"Content chunk conversion result: success={response_conversion.success if response_conversion else 'None'}")
                    except Exception as e:
                        logger.error(f"Error in convert_streaming_chunk for content chunk: {e}")
                        logger.error(f"Parameters: provider={channel.provider}, source={source_format}, chunk={chunk_data}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        # 发送原始数据作为后备，然后继续处理下一个chunk
                        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                        continue
                    
                    if response_conversion and response_conversion.success:
                        converted_data = response_conversion.data
                        
                        if isinstance(converted_data, str):
                            # 如果是SSE格式字符串（Anthropic），直接输出
                            if converted_data.strip():  # 只有非空字符串才输出
                                logger.debug(f"Sending SSE chunk {chunk_count}: {converted_data[:100]}...")
                                yield converted_data
                        elif isinstance(converted_data, list):
                            # 多个事件，逐个发送保持事件边界
                            for ev in converted_data:
                                if ev.strip():
                                    logger.debug(f"Sending SSE chunk {chunk_count}: {ev[:100]}...")
                                    yield ev
                        else:
                            # 如果是JSON对象（OpenAI/Gemini），包装成data字段
                            logger.debug(f"Sending JSON chunk {chunk_count} to client: {json.dumps(converted_data, ensure_ascii=False)}")
                            yield f"data: {json.dumps(converted_data, ensure_ascii=False)}\n\n"
                    else:
                        # 如果转换失败，返回原始数据
                        logger.warning(f"Conversion failed: {response_conversion.error}")
                        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                # 检查是否是结束chunk（各种格式的结束标记）
                # 注意：如果chunk既有内容又是结束，避免重复处理（内容处理时已经处理了结束逻辑）
                if is_finish_chunk:
                    if has_content:
                        logger.debug(f"Stream ending with content+finish chunk - already processed by content handler")
                    else:
                        logger.debug(f"Stream ending with finish-only chunk: {chunk_data}")
                
                if is_finish_chunk and not has_content:
                    # 转换并发送结束chunk（可能包含最后的内容和结束事件）
                    original_model = request_data.get("model")
                    # Fix parameter order for finish event conversion as well
                    logger.debug(f"Calling convert_streaming_chunk for finish: source={channel.provider}, target={source_format}")
                    try:
                        response_conversion = convert_streaming_chunk(channel.provider, source_format, chunk_data, original_model)
                        logger.debug(f"Finish chunk conversion result: success={response_conversion.success if response_conversion else 'None'}")
                    except Exception as e:
                        logger.error(f"Error in convert_streaming_chunk for finish chunk: {e}")
                        logger.error(f"Parameters: provider={channel.provider}, source={source_format}, chunk={chunk_data}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        # 发送原始数据作为后备
                        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                        if end_marker:  # 只有非空的end_marker才发送
                            yield end_marker
                        break
                    
                    if response_conversion and response_conversion.success:
                        converted_data = response_conversion.data
                        if isinstance(converted_data, list):
                            # 如果是事件列表（Anthropic），逐个发送每个完整事件
                            for event in converted_data:
                                if event.strip():
                                    logger.debug(f"Sending finish event: {event[:100]}...")
                                    yield event
                        elif isinstance(converted_data, str):
                            if converted_data.strip():
                                logger.debug(f"Sending finish chunk: {converted_data[:100]}...")
                                yield converted_data
                        else:
                            logger.debug(f"Sending finish chunk to client: {json.dumps(converted_data, ensure_ascii=False)}")
                            yield f"data: {json.dumps(converted_data, ensure_ascii=False)}\n\n"
                    
                    # 发送结束标记
                    if end_marker:  # 只有非空的end_marker才发送
                        yield end_marker
                    break
                    
            except json.JSONDecodeError as e:
                # 使用结构化日志记录流式 JSON 解析错误
                log_structured_error(
                    logger,
                    error_type=ERROR_TYPE_UPSTREAM_API,
                    exc=e,
                    request_body=request_data,
                    response_status=response.status_code,
                    response_body=data_content,
                    extra={
                        "channel_name": channel.name,
                        "channel_provider": channel.provider,
                        "source_format": source_format,
                        "chunk_count": chunk_count,
                        "raw_chunk_length": len(data_content),
                        "stage": "streaming_json_decode",
                    },
                )
                
                # 特殊处理：如果数据内容看起来像[DONE]但被其他字符包围
                if "[DONE]" in data_content:
                    logger.warning(f"Found [DONE] in malformed chunk: '{data_content}', sending end marker")
                    yield end_marker
                    break
                
                # 对于其他非法JSON，尝试透传（保持连接）
                logger.warning(f"Attempting to pass through malformed chunk as-is")
                yield f"data: {data_content}\n\n"
                continue
        
        logger.debug(f"Streaming completed. Total chunks processed: {chunk_count}")
        
        # 如果没有处理任何chunks，发送错误响应
        if chunk_count == 0:
            logger.warning("No chunks received from streaming response")
            # 发送一个错误响应
            error_chunk = {
                "id": "chatcmpl-error",
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": request_data.get("model", "unknown"),
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Error: No response received from AI service."},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            if end_marker:  # 只有非空的end_marker才发送
                yield end_marker
    else:
        body_bytes = await response.aread()
        try:
            body_text = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            body_text = str(body_bytes)

        status = response.status_code
        # 解析上游错误，生成友好的错误信息
        friendly_error = _parse_upstream_error(channel.provider, status, body_text)

        # 根据状态码分类错误类型
        if status in (401, 403):
            error_type = ERROR_TYPE_AUTH
        elif status == 429:
            error_type = ERROR_TYPE_RATE_LIMIT
        else:
            error_type = ERROR_TYPE_UPSTREAM_API

        log_structured_error(
            logger,
            error_type=error_type,
            request_body=request_data,
            response_status=status,
            response_headers=dict(response.headers),
            response_body=body_text,
            extra={
                "channel_name": channel.name,
                "channel_provider": channel.provider,
                "source_format": source_format,
                "is_streaming": True,
                "stage": "streaming_response_status",
                "friendly_error": friendly_error,
            },
        )

        # 对于 400 错误，抛出 HTTPException 以便返回清晰的错误信息
        if status == 400:
            raise HTTPException(status_code=400, detail=friendly_error)

        raise APIError(friendly_error)


def handle_non_streaming_response(response, channel, request_data, source_format):
    """处理非流式响应"""
    logger.info(f"Received response from {channel.provider}: status={response.status_code}")
    
    # 处理非流式响应
    if response.status_code == 200:
        response_data = response.json()
        logger.debug(f"Received response from {channel.provider}: {safe_log_response(response_data)}")
        
        # 检查是否为同格式透传
        if channel.provider == source_format:
            logger.debug(f"Same format passthrough for non-streaming response: {channel.provider} -> {source_format}")
            # 同格式直接返回原始数据
            return response_data
        else:
            # 转换响应格式
            converter = ConverterFactory().get_converter(source_format)
            
            # 设置原始模型名称
            original_model = request_data.get("model")
            if hasattr(converter, 'set_original_model') and original_model:
                converter.set_original_model(original_model)
            
            conversion_result = converter.convert_response(
                response_data, 
                channel.provider, 
                source_format
            )
            
            if not conversion_result.success:
                raise ConversionError(f"Response conversion failed: {conversion_result.error}")
            
            logger.debug(f"Converted response: {safe_log_response(conversion_result.data)}")
            return conversion_result.data
    
    # 处理 429 限流错误，返回带重试建议的响应
    elif response.status_code == 429:
        error_data = response.json() if response.content else {}
        retry_after = "20"  # OpenAI 默认建议 20 秒

        # 尝试从错误消息中提取具体等待时间
        if "error" in error_data and "message" in error_data["error"]:
            import re
            match = re.search(r'try again in (\d+)s', error_data["error"]["message"])
            if match:
                retry_after = match.group(1)

        log_structured_error(
            logger,
            error_type=ERROR_TYPE_RATE_LIMIT,
            request_body=request_data,
            response_status=response.status_code,
            response_headers=dict(response.headers),
            response_body=error_data or response.text,
            extra={
                "channel_name": channel.name,
                "channel_provider": channel.provider,
                "source_format": source_format,
                "retry_after": retry_after,
                "stage": "non_streaming_response",
            },
        )

        # 抛出 HTTPException 由上层统一处理
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": retry_after})

    else:
        error_text = response.text
        status = response.status_code

        # 解析上游错误，生成友好的错误信息
        friendly_error = _parse_upstream_error(channel.provider, status, error_text)

        # 根据状态码分类错误类型
        if status in (401, 403):
            error_type = ERROR_TYPE_AUTH
        else:
            error_type = ERROR_TYPE_UPSTREAM_API

        log_structured_error(
            logger,
            error_type=error_type,
            request_body=request_data,
            response_status=status,
            response_headers=dict(response.headers),
            response_body=error_text,
            extra={
                "channel_name": channel.name,
                "channel_provider": channel.provider,
                "source_format": source_format,
                "stage": "non_streaming_response",
                "friendly_error": friendly_error,
            },
        )

        # 对于 400 错误，直接返回清晰的错误信息
        if status == 400:
            raise HTTPException(status_code=400, detail=friendly_error)

        raise APIError(friendly_error)


# --------------------------------------------------------------------


# 统一的OpenAI兼容端点
@router.post("/v1/chat/completions")
async def unified_openai_format_endpoint(
    request: Request,
    api_key: str = Depends(extract_openai_api_key)
):
    """OpenAI格式统一端点（使用标准OpenAI认证）"""
    return await handle_unified_request(request, api_key, source_format="openai")




# 统一的模型列表端点：根据认证方式自动识别格式
@router.get("/v1/models")
async def list_models_unified(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    """统一的模型列表端点，根据认证方式自动识别OpenAI或Anthropic格式"""
    try:
        # 根据认证方式确定格式和API key
        if authorization and authorization.startswith("Bearer "):
            # OpenAI格式认证
            api_key = authorization[7:]
            target_format = "openai"
            logger.info(f"OpenAI format models request with API key: {mask_api_key(api_key)}")
        elif x_api_key:
            # Anthropic格式认证
            api_key = x_api_key
            target_format = "anthropic"
            logger.info(f"Anthropic format models request with API key: {mask_api_key(api_key)}")
        else:
            raise HTTPException(status_code=401, detail="Missing authorization header")
        
        channel = channel_manager.get_channel_by_custom_key(api_key)
        if not channel:
            logger.error(f"No channel found for API key: {mask_api_key(api_key)}")
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        logger.info(f"Found channel: {channel.name} (provider: {channel.provider})")
        
        # 从目标渠道获取真实的模型列表，转换为指定格式
        models = await fetch_models_from_channel_for_format(channel, target_format)
        
        logger.info(f"Returning {len(models)} {target_format} format models")
        
        # 根据格式返回不同的响应结构
        if target_format == "openai":
            return {
                "object": "list",
                "data": models
            }
        else:  # anthropic
            return {
                "data": models,
                "has_more": False,
                "first_id": models[0]["id"] if models else None,
                "last_id": models[-1]["id"] if models else None
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified models list failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Gemini格式：列出可用模型  
@router.get("/v1beta/models")
async def list_gemini_models(api_key: str = Depends(extract_gemini_api_key)):
    """Gemini格式：列出可用模型"""
    try:
        logger.info(f"Gemini format models request with API key: {mask_api_key(api_key)}")
        
        channel = channel_manager.get_channel_by_custom_key(api_key)
        if not channel:
            logger.error(f"No channel found for API key: {mask_api_key(api_key)}")
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        logger.info(f"Found channel: {channel.name} (provider: {channel.provider})")
        
        # 从目标渠道获取真实的模型列表，转换为Gemini格式
        models = await fetch_models_from_channel_for_format(channel, "gemini")
        
        logger.info(f"Returning {len(models)} Gemini format models")
        
        return {
            "models": models
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini format list models failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# 统一端点：基于路径自动识别格式，基于key识别目标渠道
@router.post("/v1/messages")
async def unified_anthropic_format_endpoint(
    request: Request,
    api_key: str = Depends(extract_anthropic_api_key)
):
    """Anthropic格式统一端点（使用标准Anthropic认证）"""
    return await handle_unified_request(request, api_key, source_format="anthropic")


@router.post("/v1/messages/count_tokens")
async def unified_anthropic_count_tokens_endpoint(
    request: Request,
    api_key: str = Depends(extract_anthropic_api_key)
):
    """Anthropic格式Token计数端点

    接收与 /v1/messages 相同的请求格式，返回 input_tokens 数量。
    支持转发到不同渠道：
    - Anthropic 渠道：直接转发到 Anthropic API
    - OpenAI 渠道：使用 tiktoken 估算
    - Gemini 渠道：使用 Gemini countTokens API
    """
    return await handle_anthropic_count_tokens(request, api_key)


@router.post("/v1beta/models/{model_id}:generateContent")
@router.post("/v1beta/models/{model_id}:streamGenerateContent") 
async def unified_gemini_format_endpoint(
    request: Request,
    model_id: str,
    api_key: str = Depends(extract_gemini_api_key)
):
    """Gemini格式统一端点（使用标准Gemini认证）"""
    # 检测是否为流式请求 (Gemini API特有的流式检测方式)
    is_streaming = False
    original_url = str(request.url)
    
    # Gemini API流式请求标准: 必须同时满足两个条件
    # 1. URL路径包含 :streamGenerateContent  
    # 2. URL参数包含 alt=sse
    if ":streamGenerateContent" in original_url and "alt=sse" in original_url:
        is_streaming = True
        logger.debug("Detected Gemini streaming request: :streamGenerateContent + alt=sse")
    
    # 清理模型ID，移除可能的后缀
    clean_model_id = model_id
    if ':generateContent' in model_id:
        clean_model_id = model_id.replace(':generateContent', '')
        logger.debug(f"Cleaned model ID: {model_id} -> {clean_model_id}")
    elif ':streamGenerateContent' in model_id:
        clean_model_id = model_id.replace(':streamGenerateContent', '')
        logger.debug(f"Cleaned model ID: {model_id} -> {clean_model_id}")
    
    # 将清理后的模型ID和流式标识添加到请求数据中
    request_data = await request.json()
    request_data["model"] = clean_model_id
    
    # Gemini流式检测：通过URL路径控制，但需要在请求数据中标记以便后续处理
    if is_streaming:
        # 为了让handle_unified_request知道这是流式请求，我们需要添加stream标志
        # 虽然实际发送到Gemini API时不包含这个字段，但内部处理需要
        request_data["stream"] = True
        logger.debug("Detected Gemini streaming request - added internal stream flag for processing")
    
    # 重新构建请求对象
    class ModifiedRequest:
        def __init__(self, original_request, modified_data):
            self.headers = original_request.headers
            self._json_data = modified_data
        
        async def json(self):
            return self._json_data
    
    modified_request = ModifiedRequest(request, request_data)
    return await handle_unified_request(modified_request, api_key, source_format="gemini")


@router.post("/v1beta/models/{model_id}:countTokens")
async def unified_gemini_count_tokens_endpoint(
    request: Request,
    model_id: str,
    api_key: str = Depends(extract_gemini_api_key)
):
    """Gemini格式countTokens端点（用于计算token数量）"""
    logger.info(f"Gemini countTokens request for model: {model_id}")
    
    try:
        # 清理模型ID，移除可能的countTokens后缀
        clean_model_id = model_id
        if ':countTokens' in model_id:
            clean_model_id = model_id.replace(':countTokens', '')
            logger.info(f"Cleaned model ID: {model_id} -> {clean_model_id}")
        
        # 获取请求数据
        request_data = await request.json()
        
        # 对于countTokens，只需要contents字段
        # 应用模型映射（如果配置）
        logger.debug(f"Looking for channel with custom_key: {mask_api_key(api_key)}")
        channel = channel_manager.get_channel_by_custom_key(api_key)
        if not channel:
            logger.error(f"No available channel found for API key: {mask_api_key(api_key)}")
            # 列出所有可用的渠道用于调试
            all_channels = channel_manager.get_all_channels()
            logger.info(f"Available channels: {[(ch.custom_key, ch.provider) for ch in all_channels]}")
            raise HTTPException(status_code=503, detail="No available channels")

        effective_model_id = clean_model_id
        if channel.models_mapping:
            effective_model_id = channel.models_mapping.get(clean_model_id, clean_model_id)
            if effective_model_id != clean_model_id:
                logger.info(f"Applying model mapping for countTokens: {clean_model_id} -> {effective_model_id}")

        count_request_data = {
            "model": effective_model_id,
            "contents": request_data.get("contents", [])
        }
        
        logger.debug(f"Count tokens request data: {safe_log_request(count_request_data)}")
        
        logger.info(f"Found channel: {channel.name} (provider: {channel.provider}, custom_key: {channel.custom_key})")
        
        # 根据渠道provider类型处理countTokens请求
        if channel.provider == "gemini":
            # Gemini渠道：直接转发countTokens请求
            return await handle_gemini_count_tokens(channel, effective_model_id, count_request_data)
        elif channel.provider == "openai":
            # OpenAI渠道：转换为OpenAI格式并使用tiktoken计算
            return await handle_openai_count_tokens_for_gemini(channel, effective_model_id, count_request_data)
        elif channel.provider == "anthropic":
            # Anthropic渠道：转换为Anthropic格式并估算token数量
            return await handle_anthropic_count_tokens_for_gemini(channel, effective_model_id, count_request_data)
        else:
            logger.error(f"Channel provider {channel.provider} does not support countTokens")
            raise HTTPException(status_code=400, detail=f"Channel provider {channel.provider} does not support countTokens")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini countTokens request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


async def handle_unified_request(request, api_key: str, source_format: str):
    """统一请求处理逻辑"""
    try:
        # 1. 根据key识别目标渠道
        channel = channel_manager.get_channel_by_custom_key(api_key)
        if not channel:
            logger.error(f"No channel found for api_key: {mask_api_key(api_key)}")
            raise HTTPException(status_code=401, detail="Invalid API key")

        # 2. 获取请求数据
        request_data = await request.json()

        # 3. 验证必须字段
        if not request_data.get("model"):
            raise HTTPException(status_code=400, detail="Model name is required")

        # Anthropic格式需要max_tokens字段
        if source_format == "anthropic" and not request_data.get("max_tokens"):
            raise HTTPException(status_code=400, detail="max_tokens is required for Anthropic format")

        # 验证 max_tokens 上限
        max_tokens_limit = env_config.max_tokens_limit
        if max_tokens_limit > 0:
            requested_max_tokens = _extract_max_tokens(source_format, request_data)
            if requested_max_tokens is not None and requested_max_tokens > max_tokens_limit:
                raise HTTPException(
                    status_code=400,
                    detail=f"max_tokens ({requested_max_tokens}) exceeds limit ({max_tokens_limit})"
                )

        is_streaming = request_data.get("stream", False)

        # 4. 记录请求入口日志
        log_request_entry(
            logger,
            request_method=request.method,
            request_url=str(request.url),
            request_headers=dict(request.headers),
            source_format=source_format,
            model=request_data.get("model"),
            is_streaming=is_streaming,
            channel_name=channel.name,
            channel_provider=channel.provider,
        )

        # 5. 根据流式参数选择处理方式
        
        if is_streaming:
            # 流式请求
            logger.debug("Processing streaming request")
            stream_generator = await forward_request_to_channel(
                channel=channel,
                request_data=request_data,
                source_format=source_format,
                headers=dict(request.headers)
            )
            
            return StreamingResponse(
                stream_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "X-Accel-Buffering": "no"  # 禁用Nginx缓冲，确保实时流式传输
                }
            )
        else:
            # 非流式请求
            logger.debug("Processing non-streaming request")
            response_data = await forward_request_to_channel(
                channel=channel,
                request_data=request_data,
                source_format=source_format,
                headers=dict(request.headers)
            )
            
            logger.debug(f"Final response data type: {type(response_data)}")
            logger.debug(f"Final response data: {safe_log_response(response_data)}")
            
            # 使用JSONResponse确保正确的Content-Type和编码
            return JSONResponse(
                content=response_data,
                status_code=200,
                headers={
                    "Content-Type": "application/json; charset=utf-8"
                }
            )
        
    except HTTPException:
        raise
    except ConversionError as e:
        # 转换错误
        log_structured_error(
            logger,
            error_type=ERROR_TYPE_CONVERSION,
            exc=e,
            request_method="POST",
            request_url=str(request.url) if hasattr(request, 'url') else None,
            request_headers=dict(request.headers) if hasattr(request, 'headers') else None,
            request_body=request_data if 'request_data' in dir() else None,
            extra={
                "source_format": source_format,
                "api_key": mask_api_key(api_key),
                "stage": "handle_unified_request",
            },
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_structured_error(
            logger,
            error_type=ERROR_TYPE_UPSTREAM_API,
            exc=e,
            request_method="POST",
            request_url=str(request.url) if hasattr(request, 'url') else None,
            request_headers=dict(request.headers) if hasattr(request, 'headers') else None,
            request_body=request_data if 'request_data' in dir() else None,
            extra={
                "source_format": source_format,
                "api_key": mask_api_key(api_key),
                "stage": "handle_unified_request",
            },
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


async def handle_gemini_count_tokens(channel: ChannelInfo, model_id: str, request_data: dict):
    """处理Gemini渠道的countTokens请求"""
    logger.info(f"Handling Gemini countTokens for model: {model_id}")
    
    # 构建countTokens的URL和请求
    count_tokens_url = f"{channel.base_url.rstrip('/')}/models/{model_id}:countTokens"
    logger.info(f"Sending request to: {count_tokens_url}")
    logger.info(f"Channel base_url: {channel.base_url}")
    logger.debug(f"Using API key: {mask_api_key(channel.api_key)}")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # 添加API key到URL参数
    if "?" in count_tokens_url:
        count_tokens_url += f"&key={channel.api_key}"
    else:
        count_tokens_url += f"?key={channel.api_key}"
        
    logger.info(f"Final URL with API key: {count_tokens_url}")
    
    # 发送请求到目标渠道
    async with get_http_client(channel, timeout=30.0) as client:
        response = await client.post(
            count_tokens_url,
            json=request_data,
            headers=headers
        )
        
        if response.status_code != 200:
            logger.error(f"Gemini count tokens request failed: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Count tokens request failed: {response.text}"
            )
        
        result = response.json()
        logger.info(f"Gemini count tokens response: {result}")
        
        return JSONResponse(
            content=result,
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )


async def handle_openai_count_tokens_for_gemini(channel: ChannelInfo, model_id: str, request_data: dict):
    """处理OpenAI渠道的countTokens请求，转换为Gemini格式响应"""
    logger.info(f"Handling OpenAI countTokens for Gemini format request, model: {model_id}")

    try:
        # 从Gemini格式的contents和systemInstruction提取文本用于token计数
        text_to_count = ""
        multimodal_token_estimate = 0

        # P1: 处理 systemInstruction
        system_instruction = request_data.get("systemInstruction") or request_data.get("system_instruction")
        if system_instruction:
            if isinstance(system_instruction, str):
                text_to_count += system_instruction + "\n"
            elif isinstance(system_instruction, dict):
                sys_parts = system_instruction.get("parts", [])
                for part in sys_parts:
                    if isinstance(part, dict) and "text" in part:
                        text_to_count += part["text"] + "\n"

        contents = request_data.get("contents", [])
        for content in contents:
            if isinstance(content, dict):
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, dict):
                        if "text" in part:
                            text_to_count += part["text"] + "\n"
                        elif "inlineData" in part:
                            # 多模态内容估算：图片约 85 tokens, 视频/音频按时长估算
                            mime_type = part["inlineData"].get("mimeType", "")
                            if mime_type.startswith("image/"):
                                multimodal_token_estimate += 85
                            elif mime_type.startswith("video/") or mime_type.startswith("audio/"):
                                # 假设每秒约 32 tokens, 默认估算 10 秒
                                multimodal_token_estimate += 320

        logger.info(f"Extracted text for token counting: {text_to_count[:200]}...")
        
        # 使用tiktoken计算token数量
        import tiktoken
        
        # 根据模型选择正确的编码
        if "gpt-4" in model_id.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model_id.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # 默认使用cl100k_base编码（适用于大多数现代模型）
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # 计算token数量（文本 + 多模态估算）
        token_count = len(encoding.encode(text_to_count)) + multimodal_token_estimate
        logger.info(f"Calculated token count: {token_count} (multimodal estimate: {multimodal_token_estimate})")

        # 构建Gemini格式的响应
        gemini_response = {
            "totalTokens": token_count
        }
        
        return JSONResponse(
            content=gemini_response,
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
        
    except ImportError:
        # 如果tiktoken不可用，回退到简单的字符数估算
        logger.warning("tiktoken not available, using character-based estimation")

        text_to_count = ""
        multimodal_token_estimate = 0

        # 处理 systemInstruction
        system_instruction = request_data.get("systemInstruction") or request_data.get("system_instruction")
        if system_instruction:
            if isinstance(system_instruction, str):
                text_to_count += system_instruction + "\n"
            elif isinstance(system_instruction, dict):
                for part in system_instruction.get("parts", []):
                    if isinstance(part, dict) and "text" in part:
                        text_to_count += part["text"] + "\n"

        contents = request_data.get("contents", [])
        for content in contents:
            if isinstance(content, dict):
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, dict):
                        if "text" in part:
                            text_to_count += part["text"] + "\n"
                        elif "inlineData" in part:
                            mime_type = part["inlineData"].get("mimeType", "")
                            if mime_type.startswith("image/"):
                                multimodal_token_estimate += 85
                            elif mime_type.startswith("video/") or mime_type.startswith("audio/"):
                                multimodal_token_estimate += 320

        # 简单估算：平均4个字符=1个token
        estimated_tokens = len(text_to_count) // 4 + multimodal_token_estimate
        logger.info(f"Estimated token count (character-based): {estimated_tokens}")

        gemini_response = {
            "totalTokens": estimated_tokens
        }

        return JSONResponse(
            content=gemini_response,
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    
    except Exception as e:
        logger.error(f"OpenAI countTokens conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Token counting failed: {e}")


async def handle_anthropic_count_tokens_for_gemini(channel: ChannelInfo, model_id: str, request_data: dict):
    """处理Anthropic渠道的countTokens请求，转换为Gemini格式响应"""
    logger.info(f"Handling Anthropic countTokens for Gemini format request, model: {model_id}")

    try:
        # 从Gemini格式的contents和systemInstruction提取文本用于token计数
        text_to_count = ""
        multimodal_token_estimate = 0

        # P1: 处理 systemInstruction
        system_instruction = request_data.get("systemInstruction") or request_data.get("system_instruction")
        if system_instruction:
            if isinstance(system_instruction, str):
                text_to_count += system_instruction + "\n"
            elif isinstance(system_instruction, dict):
                for part in system_instruction.get("parts", []):
                    if isinstance(part, dict) and "text" in part:
                        text_to_count += part["text"] + "\n"

        contents = request_data.get("contents", [])
        for content in contents:
            if isinstance(content, dict):
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, dict):
                        if "text" in part:
                            text_to_count += part["text"] + "\n"
                        elif "inlineData" in part:
                            mime_type = part["inlineData"].get("mimeType", "")
                            if mime_type.startswith("image/"):
                                multimodal_token_estimate += 85
                            elif mime_type.startswith("video/") or mime_type.startswith("audio/"):
                                multimodal_token_estimate += 320

        logger.info(f"Extracted text for token counting (Anthropic): {text_to_count[:200]}...")

        # Anthropic API没有专门的token计数端点，我们使用估算方法
        # Anthropic的token计算大致是：1 token ≈ 3.5个字符（英文）
        char_count = len(text_to_count)
        estimated_tokens = max(1, int(char_count / 3.5)) + multimodal_token_estimate

        logger.info(f"Estimated token count for Anthropic (char-based): {estimated_tokens}")

        # 构建Gemini格式的响应
        gemini_response = {
            "totalTokens": estimated_tokens
        }

        return JSONResponse(
            content=gemini_response,
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    except Exception as e:
        logger.error(f"Anthropic countTokens conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Token counting failed: {e}")


async def handle_anthropic_count_tokens(request: Request, api_key: str):
    """处理 Anthropic /v1/messages/count_tokens 请求

    根据渠道类型选择不同的 token 计数策略：
    - Anthropic 渠道：直接转发到 Anthropic API
    - OpenAI 渠道：使用 tiktoken 估算
    - Gemini 渠道：转换格式后调用 Gemini countTokens API
    """
    try:
        # 1. 获取渠道
        channel = channel_manager.get_channel_by_custom_key(api_key)
        if not channel:
            logger.error(f"No channel found for api_key: {mask_api_key(api_key)}")
            raise HTTPException(status_code=401, detail="Invalid API key")

        # 2. 获取请求数据
        request_data = await request.json()
        model = request_data.get("model", "")

        # 3. 应用模型映射
        effective_model = model
        if channel.models_mapping:
            effective_model = channel.models_mapping.get(model, model)
            if effective_model != model:
                logger.info(f"Model mapping applied: {model} -> {effective_model}")

        logger.info(f"Anthropic count_tokens request: model={model}, channel={channel.name}, provider={channel.provider}")

        # 4. 根据渠道类型处理
        if channel.provider == "anthropic":
            return await _forward_anthropic_count_tokens(channel, request_data, effective_model)
        elif channel.provider == "openai":
            return await _estimate_tokens_with_tiktoken(request_data, effective_model)
        elif channel.provider == "gemini":
            return await _count_tokens_via_gemini(channel, request_data, effective_model)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Channel provider '{channel.provider}' does not support count_tokens"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anthropic count_tokens request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Token counting failed: {e}")


async def _forward_anthropic_count_tokens(channel: ChannelInfo, request_data: dict, model: str):
    """直接转发到 Anthropic count_tokens API"""
    url = f"{channel.base_url.rstrip('/')}/v1/messages/count_tokens"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": channel.api_key,
        "anthropic-version": "2023-06-01",
    }

    # 更新请求中的 model
    forward_data = {**request_data, "model": model}

    logger.info(f"Forwarding count_tokens to Anthropic: {url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=forward_data, headers=headers)

        if response.status_code != 200:
            logger.error(f"Anthropic count_tokens failed: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        logger.info(f"Anthropic count_tokens result: {result}")

        return JSONResponse(
            content=result,
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )


async def _estimate_tokens_with_tiktoken(request_data: dict, model: str):
    """使用 tiktoken 估算 token 数量（OpenAI 渠道）"""
    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken not installed, falling back to character-based estimation")
        return await _estimate_tokens_by_chars(request_data)

    # 提取所有文本内容
    text_content = _extract_text_from_anthropic_request(request_data)

    # 选择合适的编码器
    try:
        if "gpt-4" in model or "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model(model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    token_count = len(encoding.encode(text_content))
    logger.info(f"tiktoken estimated tokens: {token_count}")

    return JSONResponse(
        content={"input_tokens": token_count},
        status_code=200,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )


async def _count_tokens_via_gemini(channel: ChannelInfo, request_data: dict, model: str):
    """通过 Gemini API 计算 token 数量"""
    # 转换 Anthropic 格式到 Gemini 格式
    gemini_contents = _convert_anthropic_messages_to_gemini_contents(request_data)

    url = f"{channel.base_url.rstrip('/')}/models/{model}:countTokens"
    if "?" in url:
        url += f"&key={channel.api_key}"
    else:
        url += f"?key={channel.api_key}"

    gemini_request = {"contents": gemini_contents}

    logger.info(f"Forwarding count_tokens to Gemini: {url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            json=gemini_request,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            logger.error(f"Gemini countTokens failed: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()
        total_tokens = result.get("totalTokens", 0)
        logger.info(f"Gemini countTokens result: {total_tokens}")

        # 转换为 Anthropic 响应格式
        return JSONResponse(
            content={"input_tokens": total_tokens},
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )


async def _estimate_tokens_by_chars(request_data: dict):
    """基于字符数估算 token 数量（备用方案）"""
    text_content = _extract_text_from_anthropic_request(request_data)

    # Anthropic/OpenAI 大约 1 token ≈ 4 字符（英文），中文约 1.5 字符
    char_count = len(text_content)
    estimated_tokens = max(1, int(char_count / 3.5))

    logger.info(f"Character-based token estimation: {char_count} chars -> {estimated_tokens} tokens")

    return JSONResponse(
        content={"input_tokens": estimated_tokens},
        status_code=200,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )


def _extract_text_from_anthropic_request(request_data: dict) -> str:
    """从 Anthropic 请求中提取所有文本内容"""
    text_parts = []

    # 提取 system prompt
    system = request_data.get("system")
    if isinstance(system, str):
        text_parts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

    # 提取 messages
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "tool_use":
                        text_parts.append(block.get("name", ""))
                        text_parts.append(json.dumps(block.get("input", {})))
                    elif block_type == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            text_parts.append(result_content)

    # 提取 tools 定义
    for tool in request_data.get("tools", []):
        text_parts.append(tool.get("name", ""))
        text_parts.append(tool.get("description", ""))
        text_parts.append(json.dumps(tool.get("input_schema", {})))

    return "\n".join(text_parts)


def _convert_anthropic_messages_to_gemini_contents(request_data: dict) -> list:
    """将 Anthropic messages 转换为 Gemini contents 格式"""
    contents = []

    # 添加 system prompt 作为第一条 user 消息
    system = request_data.get("system")
    if system:
        system_text = system if isinstance(system, str) else system[0].get("text", "") if system else ""
        if system_text:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System]: {system_text}"}]
            })

    # 转换 messages
    for msg in request_data.get("messages", []):
        role = msg.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"

        parts = []
        content = msg.get("content")
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        parts.append({"text": block.get("text", "")})
                    elif block_type == "tool_use":
                        parts.append({"text": f"[Tool Call: {block.get('name', '')}]"})
                    elif block_type == "tool_result":
                        result = block.get("content", "")
                        parts.append({"text": f"[Tool Result]: {result}" if isinstance(result, str) else str(result)})

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    return contents


# 健康检查端点
@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


# Gateway Anthropic count_tokens 端点（必须在通用 gateway_proxy 之前定义）
@router.post("/gateway/v1/messages/count_tokens")
async def gateway_anthropic_count_tokens(request: Request):
    """Gateway 模式下的 Anthropic /v1/messages/count_tokens 端点

    根据 GatewayConfig.provider 选择不同的 token 计数策略：
    - provider == 'anthropic'：转发到 Anthropic /v1/messages/count_tokens
    - provider == 'openai'：使用 tiktoken 在本地估算
    - provider == 'gemini'：转换为 Gemini contents 后调用 /models/{model}:countTokens
    """
    config = load_gateway_config()
    if not config or not config.enabled:
        raise HTTPException(status_code=503, detail="Gateway is not configured or disabled")

    try:
        request_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request body")

    if not isinstance(request_data, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    api_key = extract_anthropic_api_key(
        x_api_key=request.headers.get("x-api-key"),
        authorization=request.headers.get("authorization"),
    )

    original_model = request_data.get("model", "")
    if not original_model:
        raise HTTPException(status_code=400, detail="model is required for count_tokens")
    effective_model = original_model
    if config.model_mapping:
        effective_model = config.model_mapping.get(original_model, original_model)
        if effective_model != original_model:
            logger.info(f"Gateway count_tokens model mapping: {original_model} -> {effective_model}")

    logger.info(
        f"Gateway count_tokens: provider={config.provider}, "
        f"model={original_model}, effective_model={effective_model}"
    )

    try:
        if config.provider == "anthropic":
            temp_channel = ChannelInfo(
                id="gateway", name="Gateway", provider="anthropic",
                base_url=config.base_url, api_key=api_key, custom_key="__gateway__",
                timeout=config.timeout, max_retries=config.max_retries, enabled=True,
                models_mapping=config.model_mapping, created_at="", updated_at=""
            )
            return await _forward_anthropic_count_tokens(temp_channel, request_data, effective_model)

        elif config.provider == "openai":
            return await _estimate_tokens_with_tiktoken(request_data, effective_model)

        elif config.provider == "gemini":
            temp_channel = ChannelInfo(
                id="gateway", name="Gateway", provider="gemini",
                base_url=config.base_url, api_key=api_key, custom_key="__gateway__",
                timeout=config.timeout, max_retries=config.max_retries, enabled=True,
                models_mapping=config.model_mapping, created_at="", updated_at=""
            )
            return await _count_tokens_via_gemini(temp_channel, request_data, effective_model)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Gateway provider '{config.provider}' does not support count_tokens"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gateway count_tokens failed: {e}")
        raise HTTPException(status_code=500, detail=f"Token counting failed: {e}")


# Gateway 转发端点
@router.api_route("/gateway/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def gateway_proxy(full_path: str, request: Request):
    """
    Gateway 转发端点：
    - 接收 Anthropic/Gemini/OpenAI 格式请求
    - 转换为配置的目标 provider 格式
    - 使用用户自带的 API Key 转发到目标 base_url
    - 支持流式和非流式请求
    """
    # 1. 加载 Gateway 配置
    config = load_gateway_config()
    if not config or not config.enabled:
        raise HTTPException(status_code=503, detail="Gateway is not configured or disabled")

    # 2. 解析请求体（仅对 POST / PUT 解析 JSON）
    request_data = {}
    has_body = request.method in ("POST", "PUT")
    if has_body:
        try:
            request_data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON request body")

    # 3. 检测源格式（基于路径 + 请求体结构）
    detect_path = "/" + full_path if full_path else ""
    try:
        source_format = await detect_request_format(request_data, path=detect_path)
    except Exception as e:
        logger.warning(f"Failed to detect request format: {e}, defaulting to openai")
        source_format = "openai"

    # 3.1 Gemini 格式：从 URL 路径中提取 model（Gemini 请求的 model 在 URL 而非请求体中）
    if source_format == "gemini" and isinstance(request_data, dict) and not request_data.get("model"):
        # 匹配 /v1beta/models/{model_id}:generateContent 或 :streamGenerateContent
        path_for_match = "/" + full_path.lstrip("/") if full_path else ""
        marker = "/models/"
        idx = path_for_match.find(marker)
        if idx != -1:
            model_and_action = path_for_match[idx + len(marker):]
            model_id = model_and_action.split(":", 1)[0]
            if model_id:
                request_data["model"] = model_id
                logger.debug(f"Extracted Gemini model from URL path: {model_id}")

    # 4. 提取用户 API Key（根据源格式）
    if source_format == "openai":
        api_key = extract_openai_api_key(authorization=request.headers.get("authorization"))
    elif source_format == "anthropic":
        api_key = extract_anthropic_api_key(
            x_api_key=request.headers.get("x-api-key"),
            authorization=request.headers.get("authorization"),
        )
    elif source_format == "gemini":
        api_key = extract_gemini_api_key(request)
    else:
        api_key = extract_openai_api_key(authorization=request.headers.get("authorization"))

    logger.info(
        f"Gateway request: source={source_format}, target={config.provider}, "
        f"path=/gateway/{full_path}, api_key={mask_api_key(api_key)}"
    )

    # 5. 基础校验
    if request.method in ("POST", "PUT"):
        if not isinstance(request_data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        model = request_data.get("model")
        if not model and source_format != "gemini":
            raise HTTPException(status_code=400, detail="Model name is required")

        if source_format == "anthropic" and not request_data.get("max_tokens"):
            raise HTTPException(status_code=400, detail="max_tokens is required for Anthropic format")

        # max_tokens 上限校验
        max_tokens_limit = env_config.max_tokens_limit
        if max_tokens_limit > 0:
            requested_max_tokens = _extract_max_tokens(source_format, request_data)
            if requested_max_tokens and requested_max_tokens > max_tokens_limit:
                raise HTTPException(
                    status_code=400,
                    detail=f"max_tokens ({requested_max_tokens}) exceeds limit ({max_tokens_limit})"
                )

    # 计算是否为流式请求
    if has_body:
        is_streaming = bool(request_data.get("stream", False))
    else:
        is_streaming = False
    # Gemini 额外根据 URL 中的 :streamGenerateContent + alt=sse 检测
    if not is_streaming and source_format == "gemini":
        original_url = str(request.url)
        if ":streamGenerateContent" in original_url and "alt=sse" in original_url:
            is_streaming = True
            # 必须同步设置 request_data，确保转换到 OpenAI/Anthropic 时包含 stream 字段
            if isinstance(request_data, dict):
                request_data["stream"] = True
    original_model = request_data.get("model")

    # 6. 记录请求日志
    log_request_entry(
        logger,
        request_method=request.method,
        request_url=str(request.url),
        request_headers=dict(request.headers),
        source_format=source_format,
        model=original_model,
        is_streaming=is_streaming,
        channel_name="Gateway",
        channel_provider=config.provider,
    )

    # 7. 请求格式转换（仅对有 JSON 请求体的方法）
    if has_body:
        if source_format == config.provider:
            converted_data = request_data.copy()
            # Gemini 不接受请求体中的 stream 字段
            if config.provider == "gemini":
                converted_data.pop("stream", None)
        else:
            conversion_result = convert_request(
                source_format,
                config.provider,
                request_data,
                headers=dict(request.headers),
            )
            if not conversion_result.success:
                raise HTTPException(
                    status_code=400,
                    detail=f"Request conversion failed: {conversion_result.error}",
                )
            converted_data = conversion_result.data or {}
    else:
        # 无请求体的方法（如 GET / DELETE）不做格式转换
        converted_data = {}

    # 8. 应用模型映射（仅对有请求体的场景）
    if has_body and config.model_mapping and isinstance(converted_data, dict):
        mapped = config.model_mapping.get(original_model) or config.model_mapping.get(converted_data.get("model"))
        if mapped:
            logger.info(f"Gateway model mapping: {original_model} -> {mapped}")
            converted_data["model"] = mapped

    # DEBUG: 打印转换后的完整请求 (使用 print 确保显示)
    print(f"🔴🔴🔴 [DEBUG] converted_data keys: {list(converted_data.keys()) if isinstance(converted_data, dict) else 'not dict'}")
    if isinstance(converted_data, dict) and "tools" in converted_data:
        print(f"🔴🔴🔴 [DEBUG] tools: {json.dumps(converted_data['tools'], ensure_ascii=False, default=str)[:5000]}")

    # 9. 构造目标 URL（基于目标 provider，而非源格式路径）
    base = config.base_url.rstrip("/")
    raw_query = str(request.url.query) if request.url.query else ""
    query = ""
    if raw_query:
        query_pairs = parse_qsl(raw_query, keep_blank_values=True)
        # Gemini → 非 Gemini 跨格式转换时，移除 Gemini 专用参数
        if source_format == "gemini" and config.provider != "gemini":
            query_pairs = [(k, v) for k, v in query_pairs if k not in ("alt", "key")]
        if query_pairs:
            query = urlencode(query_pairs, doseq=True)

    if config.provider == "gemini" and request.method == "POST":
        # Gemini 目标：构建 Gemini 格式 URL
        model = converted_data.get("model", original_model)
        if model:
            if is_streaming:
                url = f"{base}/models/{model}:streamGenerateContent?alt=sse"
            else:
                url = f"{base}/models/{model}:generateContent"
        else:
            path = "/" + full_path.lstrip("/") if full_path else ""
            url = f"{base}{path}"
    elif request.method in ("POST", "PUT") and source_format != config.provider:
        # 跨格式转换：使用目标 provider 的规范端点
        if config.provider == "openai":
            url = f"{base}/v1/chat/completions"
        elif config.provider == "anthropic":
            url = f"{base}/v1/messages"
        else:
            path = "/" + full_path.lstrip("/") if full_path else ""
            url = f"{base}{path}"
    else:
        # 同源 provider 或非 POST/PUT，保留原始子路径
        path = "/" + full_path.lstrip("/") if full_path else ""
        url = f"{base}{path}"

    if query:
        url = f"{url}{'&' if '?' in url else '?'}{query}"

    # 10. 构造目标请求头（使用用户 API Key）
    target_headers = {"Content-Type": "application/json"}
    if config.provider == "openai":
        target_headers["Authorization"] = f"Bearer {api_key}"
    elif config.provider == "anthropic":
        target_headers["x-api-key"] = api_key
        target_headers["anthropic-version"] = "2023-06-01"
    elif config.provider == "gemini":
        target_headers["x-goog-api-key"] = api_key
        if is_streaming:
            target_headers["Accept"] = "text/event-stream"

    logger.debug(f"Gateway forwarding to: {url}")

    # 11. 转发请求
    try:
        if is_streaming:
            # 流式请求
            async def stream_generator():
                async with httpx.AsyncClient(timeout=config.timeout) as client:
                    async with client.stream(
                        request.method,
                        url=url,
                        json=converted_data if request.method in ("POST", "PUT") else None,
                        headers=target_headers,
                    ) as response:
                        if response.status_code != 200:
                            body = await response.aread()
                            body_text = body.decode("utf-8", errors="replace")
                            status = response.status_code
                            error_msg = _parse_upstream_error(config.provider, status, body_text)

                            # 根据状态码分类错误类型
                            if status in (401, 403):
                                error_type = ERROR_TYPE_AUTH
                            elif status == 429:
                                error_type = ERROR_TYPE_RATE_LIMIT
                            else:
                                error_type = ERROR_TYPE_UPSTREAM_API

                            log_structured_error(
                                logger,
                                error_type=error_type,
                                request_method=request.method,
                                request_url=url,
                                request_headers=target_headers,
                                request_body=converted_data if request.method in ("POST", "PUT") else None,
                                response_status=status,
                                response_headers=dict(response.headers),
                                response_body=body_text,
                                extra={"gateway": True, "provider": config.provider, "streaming": True},
                            )
                            raise HTTPException(status_code=status, detail=error_msg)

                        # 构造临时 channel 用于响应转换
                        temp_channel = ChannelInfo(
                            id="gateway",
                            name="Gateway",
                            provider=config.provider,
                            base_url=config.base_url,
                            api_key=api_key,
                            custom_key="__gateway__",
                            timeout=config.timeout,
                            max_retries=config.max_retries,
                            enabled=True,
                            models_mapping=config.model_mapping,
                            created_at="",
                            updated_at="",
                        )
                        async for chunk in handle_streaming_response(response, temp_channel, request_data, source_format):
                            yield chunk

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            # 非流式请求
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.request(
                    method=request.method,
                    url=url,
                    json=converted_data if request.method in ("POST", "PUT") else None,
                    headers=target_headers,
                )

            if response.status_code != 200:
                status = response.status_code
                body_text = response.text
                error_msg = _parse_upstream_error(config.provider, status, body_text)

                # 根据状态码分类错误类型
                if status in (401, 403):
                    error_type = ERROR_TYPE_AUTH
                elif status == 429:
                    error_type = ERROR_TYPE_RATE_LIMIT
                else:
                    error_type = ERROR_TYPE_UPSTREAM_API

                log_structured_error(
                    logger,
                    error_type=error_type,
                    request_method=request.method,
                    request_url=url,
                    request_headers=target_headers,
                    request_body=converted_data if request.method in ("POST", "PUT") else None,
                    response_status=status,
                    response_headers=dict(response.headers),
                    response_body=body_text,
                    extra={"gateway": True, "provider": config.provider, "streaming": False},
                )
                raise HTTPException(status_code=status, detail=error_msg)

            response_data = response.json()

            # 响应格式转换（仅对 POST / PUT 做格式转换）
            if has_body and source_format != config.provider:
                response_result = convert_response(config.provider, source_format, response_data)
                if response_result.success:
                    response_data = response_result.data

            return JSONResponse(
                content=response_data,
                status_code=200,
                headers={"Content-Type": "application/json; charset=utf-8"}
            )

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Gateway timeout after {config.timeout} seconds")
    except Exception as e:
        logger.error(f"Gateway error: {e}")
        raise HTTPException(status_code=502, detail=f"Gateway error: {str(e)}")
