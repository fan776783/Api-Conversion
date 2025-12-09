"""
Web API for capability detection
"""
import asyncio
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

from core.capability_detector import CapabilityDetectorFactory
from core.openai_detector import OpenAICapabilityDetector
from core.anthropic_detector import AnthropicCapabilityDetector
from core.gemini_detector import GeminiCapabilityDetector
from src.utils.config import ConfigManager, ChannelConfig
from src.utils.logger import setup_logger
from src.utils.auth import auth_manager
from src.utils.security import mask_api_key
from api.conversion_api import router as conversion_router
from api.unified_api import router as unified_router

app = FastAPI(title="AI API统一转换代理系统", version="1.0.0")
logger = setup_logger("web_api")

# 添加会话中间件
import os
import secrets
from datetime import datetime, timedelta

def get_session_secret_key():
    """获取会话密钥，如果未设置则生成并持久化随机密钥"""
    session_key = os.getenv("SESSION_SECRET_KEY")
    if not session_key:
        # 尝试从数据库获取已保存的密钥
        from src.utils.database import db_manager
        stored_key = db_manager.get_config("session_secret_key")
        if stored_key:
            session_key = stored_key
            logger.info("Using stored session secret key from database")
        else:
            # 生成新的随机密钥并存储
            session_key = secrets.token_hex(32)
            db_manager.set_config("session_secret_key", session_key)
            logger.warning("SESSION_SECRET_KEY not set, generated and stored new random key.")
            logger.info("Sessions will now persist across server restarts.")
    else:
        logger.info("Using SESSION_SECRET_KEY from environment variables")
    return session_key

secret_key = get_session_secret_key()
app.add_middleware(SessionMiddleware, secret_key=secret_key)

# 包含API路由
app.include_router(conversion_router, prefix="/api")  # 管理API
app.include_router(unified_router)  # 统一转换API（直接挂载到根路径）

# 注册检测器
CapabilityDetectorFactory.register("openai", OpenAICapabilityDetector)
CapabilityDetectorFactory.register("anthropic", AnthropicCapabilityDetector)
CapabilityDetectorFactory.register("gemini", GeminiCapabilityDetector)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局状态存储
detection_results: Dict[str, Any] = {}
detection_progress: Dict[str, Dict[str, Any]] = {}
task_timestamps: Dict[str, float] = {}
active_tasks: Dict[str, bool] = {}

# 简单的并发控制
MAX_CONCURRENT_TASKS = 10
TASK_EXPIRY_HOURS = 2
CLEANUP_INTERVAL_MINUTES = 30

import uuid
import time
import asyncio
from fastapi import BackgroundTasks

class DetectionRequest(BaseModel):
    """检测请求模型"""
    provider: str
    base_url: str
    api_key: str
    timeout: int = 30
    capabilities: Optional[List[str]] = None
    target_model: str

class DetectionResponse(BaseModel):
    """检测响应模型"""
    task_id: str
    message: str

class CapabilityResultModel(BaseModel):
    """能力检测结果模型"""
    capability: str
    status: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response_time: Optional[float] = None


class ChannelCapabilitiesModel(BaseModel):
    """渠道能力模型"""
    provider: str
    base_url: str
    models: List[str]
    capabilities: Dict[str, CapabilityResultModel]
    detection_time: str


class LoginRequest(BaseModel):
    """登录请求"""
    password: str


class ChangePasswordRequest(BaseModel):
    """修改密码请求"""
    old_password: str
    new_password: str


class LoginResponse(BaseModel):
    """登录响应"""
    success: bool
    session_token: Optional[str] = None
    message: str


def get_session_user(request: Request):
    """获取会话用户，验证是否已登录"""
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="未登录")
    return True


def get_optional_session_user(request: Request):
    """可选的会话用户获取，用于页面渲染"""
    return request.session.get("authenticated", False)


def channel_info_to_config(channel_info) -> ChannelConfig:
    """将ChannelInfo转换为ChannelConfig"""
    return ChannelConfig(
        provider=channel_info.provider,
        base_url=channel_info.base_url,
        api_key=channel_info.api_key,
        timeout=channel_info.timeout,
        max_retries=channel_info.max_retries,
        use_proxy=getattr(channel_info, 'use_proxy', False),
        proxy_type=getattr(channel_info, 'proxy_type', None),
        proxy_host=getattr(channel_info, 'proxy_host', None),
        proxy_port=getattr(channel_info, 'proxy_port', None),
        proxy_username=getattr(channel_info, 'proxy_username', None),
        proxy_password=getattr(channel_info, 'proxy_password', None)
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """返回登录页面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI API统一转换代理系统 - 登录</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div class="login-container">
            <div class="login-form">
                <h1>AI API FORMAT CONVERSION</h1>
                <h2>管理员登录</h2>
                <form id="loginForm">
                    <div class="form-group">
                        <label for="password">管理员密码:</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn-primary">登录</button>
                </form>
                <div id="error-message" class="error-message" style="display: none;"></div>
            </div>
        </div>

        <script>
            document.getElementById('loginForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const password = document.getElementById('password').value;
                const errorDiv = document.getElementById('error-message');

                try {
                    const response = await fetch('/api/login', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ password })
                    });

                    const result = await response.json();

                    if (result.success) {
                        // Cookie已自动设置，直接跳转
                        window.location.href = '/dashboard';
                    } else {
                        errorDiv.textContent = result.message;
                        errorDiv.style.display = 'block';
                    }
                } catch (error) {
                    errorDiv.textContent = '登录失败，请重试';
                    errorDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/")
async def index():
    """重定向到登录页面"""
    return RedirectResponse(url="/login", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """管理仪表板 - 需要认证"""
    # 检查会话认证状态
    if not request.session.get("authenticated"):
        # 认证失败，重定向到登录页
        return RedirectResponse(url="/login", status_code=302)
    
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI API统一转换代理系统</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div class="container">
            <div class="header-bar">
                <h1>AI API FORMAT CONVERSION</h1>
                <div class="header-actions">
                    <button onclick="showChangePasswordModal()" class="btn-tertiary">修改密码</button>
                    <button onclick="logout()" class="btn-secondary">注销</button>
                </div>
            </div>

            <!-- 标签页导航 -->
            <div class="tab-navigation">
                <button class="tab-button active" data-tab="converter">API转换器</button>
                <button class="tab-button" data-tab="detector">能力检测器</button>
            </div>

            <!-- API转换器标签页 -->
            <div id="converter-tab" class="tab-content active">
                <!-- 渠道管理区域 -->
                <div class="channel-management">
                <h2>渠道管理</h2>
                <div class="channel-form">
                    <h3>添加新渠道</h3>
                    <form id="channelForm">
                        <div class="form-row">
                            <div class="form-group">
                                <label for="name">渠道名称:</label>
                                <input type="text" id="name" name="name" required>
                            </div>
                            <div class="form-group">
                                <label for="channel_provider">提供商:</label>
                                <select id="channel_provider" name="provider" required>
                                    <option value="">请选择</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label for="base_url">Base URL:</label>
                                <input type="url" id="base_url" name="base_url" required>
                            </div>
                            <div class="form-group">
                                <label for="api_key">API密钥:</label>
                                <div class="password-field-container">
                                    <input type="password" id="api_key" name="api_key" required>
                                    <button type="button" class="password-toggle-btn" onclick="togglePasswordVisibility('api_key')" title="显示">
                                        <span class="icon" id="api_key_icon">○○</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label for="custom_key">自定义Key:</label>
                                <input type="text" id="custom_key" name="custom_key" placeholder="用户调用时使用的key，例如：my-key-123" required>
                                <small class="form-hint">用户调用API时使用此key进行身份验证</small>
                            </div>
                            <div class="proxy-toggle-container" id="proxyToggleContainer">
                                <input type="checkbox" id="use_proxy" name="use_proxy" onchange="toggleProxyFields()">
                                <div class="proxy-switch" id="proxySwitch" onclick="toggleProxySwitch()"></div>
                                <div class="proxy-toggle-content">
                                    <div class="proxy-toggle-label" onclick="toggleProxySwitch()">
                                        <span>启用代理服务器</span>
                                    </div>
                                    <div class="proxy-toggle-description">通过代理服务器转发API请求</div>
                                </div>
                            </div>
                        </div>
                        
                        <div id="proxy-fields" class="form-section" style="display: none;">
                            <h4>代理配置</h4>
                            <div class="form-row">
                                <div class="form-group">
                                    <label for="proxy_type">代理类型:</label>
                                    <select id="proxy_type" name="proxy_type">
                                        <option value="http">HTTP</option>
                                        <option value="https">HTTPS</option>
                                        <option value="socks5">SOCKS5</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="proxy_host">代理地址:</label>
                                    <input type="text" id="proxy_host" name="proxy_host" placeholder="例如：127.0.0.1 或 proxy.example.com">
                                </div>
                            </div>
                            <div class="form-row">
                                <div class="form-group">
                                    <label for="proxy_port">代理端口:</label>
                                    <input type="number" id="proxy_port" name="proxy_port" placeholder="例如：8080" min="1" max="65535">
                                </div>
                                <div class="form-group">
                                    <label for="proxy_username">代理用户名 (可选):</label>
                                    <input type="text" id="proxy_username" name="proxy_username" placeholder="如果需要认证则填写">
                                </div>
                            </div>
                            <div class="form-row">
                                <div class="form-group">
                                    <label for="proxy_password">代理密码 (可选):</label>
                                    <div class="password-field-container">
                                        <input type="password" id="proxy_password" name="proxy_password" placeholder="如果需要认证则填写">
                                        <button type="button" class="password-toggle-btn" onclick="togglePasswordVisibility('proxy_password')" title="显示">
                                            <span class="icon" id="proxy_password_icon">○○</span>
                                        </button>
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label>&nbsp;</label>
                                    <button type="button" id="testProxyBtn" class="btn-secondary" onclick="testProxyConnection()" style="margin-top: 5px;">
                                        <span id="testProxyBtnText">测试代理连接</span>
                                        <span id="testProxySpinner" class="spinner" style="display: none;">⟳</span>
                                    </button>
                                </div>
                            </div>
                            <div class="form-row">
                                <div class="form-group" style="grid-column: 1 / -1;">
                                    <div id="proxyTestResult" class="test-result" style="display: none;">
                                        <div id="proxyTestContent"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label for="timeout">超时时间(秒):</label>
                                <input type="number" id="timeout" name="timeout" value="30" min="10" max="300">
                            </div>
                            <div class="form-group">
                                <label for="max_retries">重试次数:</label>
                                <input type="number" id="max_retries" name="max_retries" value="3" min="1" max="10">
                            </div>
                        </div>
                        <div class="form-section" id="model-mapping-section">
                            <h4>模型映射</h4>
                            <p class="form-hint">当客户端请求某个模型时，可映射为渠道实际调用的模型。例如：将 <code>A1</code> 映射为 <code>B1</code>。可配置多条。</p>
                            <div id="model-mapping-rows">
                                <!-- 动态添加映射行 -->
                            </div>
                            <div class="form-actions" style="margin-top: 8px;">
                                <button type="button" class="btn-secondary" onclick="apiConverter.addMappingRow()">新增映射</button>
                            </div>
                        </div>
                        <button type="submit" class="btn-primary">添加渠道</button>
                    </form>
                </div>
                
                <div class="channel-list">
                    <h3>已配置渠道</h3>
                    <div id="channelsList">
                        <p>正在加载渠道列表...</p>
                    </div>
                </div>
            </div>

            <!-- Gateway 配置区域 -->
            <div class="channel-management">
                <h2>Gateway 转发配置</h2>
                <div class="channel-form">
                    <div class="gateway-form-header">
                        <h3>全局配置</h3>
                        <span id="gatewayStatus" class="status-badge status-unconfigured">未配置</span>
                    </div>
                    <p class="form-hint" style="margin-bottom: 20px;">Gateway 允许用户使用自己的 API Key，通过 <code>/gateway/*</code> 端点访问，系统会自动进行格式转换后转发到目标服务。</p>
                    <form id="gatewayForm">
                        <div class="form-row">
                            <div class="form-group">
                                <label for="gateway_provider">目标提供商:</label>
                                <select id="gateway_provider" name="provider" required>
                                    <option value="">请选择</option>
                                    <option value="openai">OpenAI</option>
                                    <option value="anthropic">Anthropic</option>
                                    <option value="gemini">Google Gemini</option>
                                </select>
                                <small class="form-hint">请求将被转换为此格式</small>
                            </div>
                            <div class="form-group">
                                <label for="gateway_base_url">目标 Base URL:</label>
                                <input type="url" id="gateway_base_url" name="base_url" placeholder="例如：https://api.openai.com/v1" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label for="gateway_timeout">超时时间(秒):</label>
                                <input type="number" id="gateway_timeout" name="timeout" value="30" min="1" max="600">
                            </div>
                            <div class="form-group">
                                <label for="gateway_max_retries">重试次数:</label>
                                <input type="number" id="gateway_max_retries" name="max_retries" value="1" min="0" max="10">
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-group" style="grid-column: 1 / -1;">
                                <label class="checkbox-label">
                                    <input type="checkbox" id="gateway_enabled" name="enabled" checked>
                                    <span>启用 Gateway</span>
                                </label>
                            </div>
                        </div>
                        <div class="form-section" id="gateway-model-mapping-section">
                            <h4>模型映射</h4>
                            <p class="form-hint">用户请求中的模型名将被替换为目标模型名。例如：<code>claude-3-5-sonnet</code> → <code>gpt-4o</code></p>
                            <div id="gateway-model-mapping-rows">
                                <!-- 动态添加映射行 -->
                            </div>
                            <div class="form-actions" style="margin-top: 8px;">
                                <button type="button" class="btn-secondary" onclick="addGatewayMappingRow()">新增映射</button>
                            </div>
                        </div>
                        <div class="form-actions" style="margin-top: 20px;">
                            <button type="submit" class="btn-primary">保存配置</button>
                            <button type="button" class="btn-danger" onclick="deleteGatewayConfig()" id="deleteGatewayBtn" style="display: none;">删除配置</button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- API使用说明区域 -->
            <div class="api-usage">
                <h2>API使用说明</h2>
                <div class="usage-description">
                    <p><strong>核心功能：</strong>AI API格式统一转换代理系统，支持OpenAI、Anthropic、Gemini三种格式的相互转换</p>
                    <p><strong>智能路由：</strong>根据请求路径自动识别API格式，根据自定义key识别目标渠道，自动进行格式转换</p>
                </div>
                <div class="usage-notes">
                    <h4>工作原理</h4>
                    <ol>
                        <li><strong>格式识别：</strong>根据请求路径自动识别源API格式（OpenAI/Anthropic/Gemini）</li>
                        <li><strong>渠道路由：</strong>根据自定义key查找对应的目标渠道配置</li>
                        <li><strong>格式转换：</strong>自动将请求格式转换为目标渠道的API格式</li>
                        <li><strong>请求转发：</strong>调用真实的AI服务API</li>
                        <li><strong>响应转换：</strong>将响应格式转换回源格式并返回给客户端</li>
                    </ol>
                    <h4>支持的转换</h4>
                    <ul>
                        <li>OpenAI ↔ Anthropic ↔ Gemini（任意格式间相互转换）</li>
                        <li>流式和非流式请求</li>
                        <li>函数调用、视觉理解、结构化输出等高级功能</li>
                        <li>自动模型映射和参数适配</li>
                    </ul>
                </div>
            </div>
            </div>

            <!-- 能力检测器标签页 -->
            <div id="detector-tab" class="tab-content">
                <div class="detection-form">
                    <h2>能力检测</h2>
                    <form id="detectionForm">
                        <div class="form-group">
                            <label for="provider">AI提供商:</label>
                            <select id="provider" name="provider" required>
                                <option value="">请选择</option>
                                <option value="openai">OpenAI</option>
                                <option value="anthropic">Anthropic</option>
                                <option value="gemini">Google Gemini</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="detection_base_url">API基础URL:</label>
                            <input type="url" id="detection_base_url" name="base_url" required>
                        </div>

                        <div class="form-group">
                            <label for="detection_api_key">API密钥:</label>
                            <input type="password" id="detection_api_key" name="api_key" required>
                        </div>

                        <div class="form-group">
                            <label for="detection_timeout">超时时间(秒):</label>
                            <input type="number" id="detection_timeout" name="timeout" value="30" min="10" max="300">
                        </div>
                    
                    <div class="form-group">
                        <label for="target_model">指定模型 (必填):</label>
                        <div class="model-selection-container">
                            <select id="target_model_select" name="target_model_select" onchange="updateTargetModel()">
                                <option value="">请选择模型</option>
                            </select>
                            <button type="button" id="fetch_models_btn" class="btn-secondary" onclick="fetchModels()">获取模型列表</button>
                        </div>
                        <input type="text" id="target_model" name="target_model" placeholder="或手动输入模型名称，例如：gpt-4o-mini" style="margin-top: 10px;" required>
                        <small class="form-hint">必须指定要检测的模型</small>
                    </div>
                    
                    <div class="form-group">
                        <label>要检测的能力:</label>
                        <div class="capabilities-grid">
                            <label>
                                <input type="checkbox" name="capabilities" value="basic_chat">
                                <span class="capability-text">基础聊天</span>
                                <span class="capability-icon"></span>
                            </label>
                            <label>
                                <input type="checkbox" name="capabilities" value="streaming">
                                <span class="capability-text">流式输出</span>
                                <span class="capability-icon"></span>
                            </label>
                            <label>
                                <input type="checkbox" name="capabilities" value="system_message">
                                <span class="capability-text">系统消息</span>
                                <span class="capability-icon"></span>
                            </label>
                            <label>
                                <input type="checkbox" name="capabilities" value="function_calling">
                                <span class="capability-text">函数调用</span>
                                <span class="capability-icon"></span>
                            </label>
                            <label>
                                <input type="checkbox" name="capabilities" value="structured_output">
                                <span class="capability-text">结构化输出</span>
                                <span class="capability-icon"></span>
                            </label>
                            <label>
                                <input type="checkbox" name="capabilities" value="vision">
                                <span class="capability-text">视觉理解</span>
                                <span class="capability-icon"></span>
                            </label>
                        </div>
                        <div class="form-actions">
                            <button type="button" onclick="selectAllCapabilities()">全选</button>
                            <button type="button" onclick="clearAllCapabilities()">清空</button>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn-primary">开始检测</button>
                </form>
            </div>
            
            <div id="progress" class="progress-section" style="display: none;">
                <h2>检测进度</h2>
                <div id="progress-bar" class="progress-bar">
                    <div id="progress-fill" class="progress-fill"></div>
                </div>
                <div id="progress-text">准备中...</div>
            </div>
            
                <div id="results" class="results-section" style="display: none;">
                    <h2>检测结果</h2>
                    <div id="results-content"></div>
                </div>
            </div>

            <!-- 能力检测器标签页 -->
            <div id="detector-tab" class="tab-content">
                <h2>能力检测器</h2>
                <p>能力检测功能正在开发中...</p>
            </div>

            <!-- 修改密码模态框 -->
            <div id="changePasswordModal" class="modal" style="display: none;">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>修改管理员密码</h3>
                        <button type="button" class="modal-close" onclick="hideChangePasswordModal()">&times;</button>
                    </div>
                    <form id="changePasswordForm" class="password-form">
                        <div class="form-group">
                            <label for="oldPassword">当前密码:</label>
                            <input type="password" id="oldPassword" name="oldPassword" required>
                        </div>
                        <div class="form-group">
                            <label for="newPassword">新密码:</label>
                            <input type="password" id="newPassword" name="newPassword" required minlength="6">
                            <small class="form-hint">密码长度不少于6位</small>
                        </div>
                        <div class="form-group">
                            <label for="confirmPassword">确认新密码:</label>
                            <input type="password" id="confirmPassword" name="confirmPassword" required minlength="6">
                        </div>
                        <div class="modal-actions">
                            <button type="button" class="btn-secondary" onclick="hideChangePasswordModal()">取消</button>
                            <button type="submit" class="btn-primary">修改密码</button>
                        </div>
                    </form>
                    <div id="password-message" class="message" style="display: none;"></div>
                </div>
            </div>
        </div>

        <script>
            // 切换代理字段显示
            function toggleProxyFields() {
                const useProxy = document.getElementById('use_proxy');
                const proxyFields = document.getElementById('proxy-fields');
                
                if (useProxy && proxyFields) {
                    if (useProxy.checked) {
                        proxyFields.style.display = 'block';
                        // 设置必填字段
                        document.getElementById('proxy_host').required = true;
                        document.getElementById('proxy_port').required = true;
                    } else {
                        proxyFields.style.display = 'none';
                        // 取消必填字段
                        document.getElementById('proxy_host').required = false;
                        document.getElementById('proxy_port').required = false;
                        // 清空字段
                        document.getElementById('proxy_host').value = '';
                        document.getElementById('proxy_port').value = '';
                        document.getElementById('proxy_username').value = '';
                        document.getElementById('proxy_password').value = '';
                    }
                }
            }

            // 注销功能
            async function logout() {
                try {
                    const response = await fetch('/api/logout', {
                        method: 'POST'
                    });
                    if (response.ok) {
                        window.location.href = '/login';
                    }
                } catch (error) {
                    console.error('注销失败:', error);
                    window.location.href = '/login'; // 即使失败也跳转
                }
            }

            // 模态框显示/隐藏功能
            function showChangePasswordModal() {
                const modal = document.getElementById('changePasswordModal');
                modal.style.display = 'flex';
                // 清空表单和消息
                document.getElementById('changePasswordForm').reset();
                const messageDiv = document.getElementById('password-message');
                messageDiv.style.display = 'none';
                // 聚焦到第一个输入框
                document.getElementById('oldPassword').focus();
            }

            function hideChangePasswordModal() {
                const modal = document.getElementById('changePasswordModal');
                modal.style.display = 'none';
            }

            // 点击模态框背景关闭
            window.onclick = function(event) {
                const modal = document.getElementById('changePasswordModal');
                if (event.target === modal) {
                    hideChangePasswordModal();
                }
            }

            // 修改密码功能
            async function changePassword(event) {
                event.preventDefault();
                
                const oldPassword = document.getElementById('oldPassword').value;
                const newPassword = document.getElementById('newPassword').value;
                const confirmPassword = document.getElementById('confirmPassword').value;
                const messageDiv = document.getElementById('password-message');
                
                // 前端验证
                if (newPassword !== confirmPassword) {
                    showPasswordMessage('新密码和确认密码不一致', 'error');
                    return;
                }
                
                if (newPassword.length < 6) {
                    showPasswordMessage('新密码长度不能少于6位', 'error');
                    return;
                }
                
                if (oldPassword === newPassword) {
                    showPasswordMessage('新密码不能与旧密码相同', 'error');
                    return;
                }
                
                try {
                    const response = await fetch('/api/change-password', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            old_password: oldPassword,
                            new_password: newPassword
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        showPasswordMessage(result.message, 'success');
                        // 清空表单
                        document.getElementById('changePasswordForm').reset();
                        
                        // 如果需要重新认证，2秒后关闭模态框并跳转到登录页
                        if (result.require_reauth) {
                            setTimeout(() => {
                                hideChangePasswordModal();
                                setTimeout(() => {
                                    window.location.href = '/login';
                                }, 500);
                            }, 2000);
                        }
                    } else {
                        showPasswordMessage(result.message, 'error');
                    }
                } catch (error) {
                    console.error('修改密码失败:', error);
                    showPasswordMessage('修改密码失败，请稍后重试', 'error');
                }
            }
            
            function showPasswordMessage(message, type) {
                const messageDiv = document.getElementById('password-message');
                messageDiv.textContent = message;
                messageDiv.className = `message ${type}`;
                messageDiv.style.display = 'block';
                
                // 3秒后自动隐藏消息（除非是成功消息）
                if (type !== 'success') {
                    setTimeout(() => {
                        messageDiv.style.display = 'none';
                    }, 3000);
                }
            }

            // 页面加载时初始化
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Dashboard loaded, user authenticated by server');

                console.log('Token exists, loading script.js...');
                // 认证成功，加载主要功能脚本
                const script = document.createElement('script');
                script.src = '/static/script.js';
                script.onload = function() {
                    console.log('script.js loaded successfully');
                    console.log('apiConverter available:', typeof apiConverter !== 'undefined');
                };
                script.onerror = function() {
                    console.error('Failed to load script.js');
                };
                document.head.appendChild(script);
                
                // 绑定修改密码表单事件
                const changePasswordForm = document.getElementById('changePasswordForm');
                if (changePasswordForm) {
                    changePasswordForm.addEventListener('submit', changePassword);
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/api/login")
async def login(login_request: LoginRequest, request: Request):
    """管理员登录"""
    # 验证密码
    if auth_manager.verify_admin_password(login_request.password):
        # 设置会话
        request.session["authenticated"] = True
        request.session["login_time"] = datetime.now().isoformat()
        
        return JSONResponse(
            content={
                "success": True, 
                "message": "登录成功"
            }
        )
    else:
        return JSONResponse(
            content={
                "success": False,
                "message": "密码错误"
            },
            status_code=401
        )


@app.post("/api/logout")
async def logout(request: Request):
    """管理员注销"""
    # 清除会话
    request.session.clear()
    
    return JSONResponse(
        content={
            "success": True, 
            "message": "注销成功"
        }
    )


@app.post("/api/change-password")
async def change_password(change_request: ChangePasswordRequest, request: Request):
    """修改管理员密码 - 需要认证"""
    # 检查会话认证状态
    if not request.session.get("authenticated"):
        return JSONResponse(
            content={
                "success": False,
                "message": "未授权访问，请先登录"
            },
            status_code=401
        )
    
    # 验证旧密码
    if not auth_manager.verify_admin_password(change_request.old_password):
        return JSONResponse(
            content={
                "success": False,
                "message": "旧密码错误"
            },
            status_code=400
        )
    
    # 验证新密码格式
    if len(change_request.new_password) < 6:
        return JSONResponse(
            content={
                "success": False,
                "message": "新密码长度不能少于6位"
            },
            status_code=400
        )
    
    if change_request.old_password == change_request.new_password:
        return JSONResponse(
            content={
                "success": False,
                "message": "新密码不能与旧密码相同"
            },
            status_code=400
        )
    
    try:
        # 设置新密码（会自动清除所有会话）
        auth_manager.set_admin_password(change_request.new_password)
        
        return JSONResponse(
            content={
                "success": True,
                "message": "密码修改成功，所有会话已失效，请重新登录",
                "require_reauth": True
            }
        )
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        return JSONResponse(
            content={
                "success": False,
                "message": "密码修改失败，请稍后重试"
            },
            status_code=500
        )




@app.get("/api/providers")
async def get_providers():
    """获取支持的提供商"""
    return {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "OpenAI (GPT-4o, GPT-3.5-turbo 等)",
                "default_url": "https://api.openai.com/v1"
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "description": "Anthropic (Claude-3, Claude-3.5 等)",
                "default_url": "https://api.anthropic.com"
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "description": "Google Gemini (Gemini-1.5, Gemini-2.0 等)",
                "default_url": "https://generativelanguage.googleapis.com/v1beta"
            }
        ]
    }


# 注意：获取渠道列表API已统一到 conversion_api.py 中
# 原本此处的 @app.get("/api/channels") 端点与 conversion_api.py 重复
# 为避免路由冲突，已移除。前端请求会自动使用 conversion_api.py 中的端点


# 注意：创建渠道API已统一到 conversion_api.py 中
# 原本此处的 @app.post("/api/channels") 端点与 conversion_api.py 重复
# 为避免路由冲突，已移除。前端请求会自动使用 conversion_api.py 中的端点


# 注意：渠道更新API已统一到 conversion_api.py 中
# 原本此处的 @app.put("/api/channels/{channel_id}") 端点与 conversion_api.py 重复
# 为避免路由冲突，已移除。前端请求会自动使用 conversion_api.py 中的端点


# 注意：删除渠道API已统一到 conversion_api.py 中
# 原本此处的 @app.delete("/api/channels/{channel_id}") 端点与 conversion_api.py 重复
# 为避免路由冲突，已移除。前端请求会自动使用 conversion_api.py 中的端点


@app.get("/api/capabilities")
async def get_capabilities():
    """获取所有可检测的能力"""
    config_manager = ConfigManager()
    capabilities = config_manager.get_all_capabilities()
    
    return {
        "capabilities": [
            {
                "id": name,
                "name": name,
                "description": config.description
            }
            for name, config in capabilities.items()
        ]
    }






@app.post("/api/detect", response_model=DetectionResponse)
async def start_detection(request: DetectionRequest, background_tasks: BackgroundTasks):
    """开始能力检测"""
    
    # 简单的并发控制
    active_count = len([task for task, is_active in active_tasks.items() if is_active])
    if active_count >= MAX_CONCURRENT_TASKS:
        raise HTTPException(
            status_code=429, 
            detail=f"服务器繁忙，请稍后重试 ({active_count}/{MAX_CONCURRENT_TASKS})"
        )
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 创建配置
    try:
        config = ChannelConfig(
            provider=request.provider,
            base_url=request.base_url,
            api_key=request.api_key,
            timeout=request.timeout
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # 初始化进度和时间戳
    current_time = time.time()
    task_timestamps[task_id] = current_time
    active_tasks[task_id] = True
    
    detection_progress[task_id] = {
        "status": "starting",
        "progress": 0,
        "current_capability": None,
        "completed_capabilities": [],
        "total_capabilities": 0
    }
    
    # 后台执行检测
    logger.info(f"启动检测任务: {request.provider} - {request.target_model}")
    background_tasks.add_task(run_detection, task_id, config, request.capabilities, request.target_model)
    
    return DetectionResponse(
        task_id=task_id,
        message="检测任务已开始"
    )


@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    """获取检测进度"""
    if task_id not in detection_progress:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return detection_progress[task_id]


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """获取检测结果"""
    if task_id not in detection_results:
        raise HTTPException(status_code=404, detail="结果不存在")
    
    return detection_results[task_id]


@app.post("/api/fetch_models")
async def fetch_models(request: dict):
    """获取模型列表"""
    try:
        provider = request.get("provider")
        base_url = request.get("base_url")
        api_key = request.get("api_key")
        
        if not all([provider, base_url, api_key]):
            raise HTTPException(status_code=400, detail="缺少必要参数")
        
        # 创建配置
        config = ChannelConfig(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            timeout=30
        )
        
        # 创建检测器
        detector = CapabilityDetectorFactory.create(config)
        
        # 获取模型列表
        logger.info(f"正在获取 {provider} 的模型列表...")
        models = await detector.detect_models()
        logger.info(f"成功获取 {len(models)} 个模型")
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"模型获取失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_detection(task_id: str, config: ChannelConfig, selected_capabilities: Optional[List[str]], target_model: str):
    """运行能力检测"""
    try:
        # 更新进度
        detection_progress[task_id]["status"] = "running"
        
        # 创建检测器
        detector = CapabilityDetectorFactory.create(config)
        
        # 设置目标模型（现在是必填的）
        detector.target_model = target_model
        
        # 设置进度回调
        async def progress_callback(progress_data):
            """更新检测进度"""
            if task_id in detection_progress:
                detection_progress[task_id].update(progress_data)
        
        detector.progress_callback = progress_callback
        
        # 执行检测
        if selected_capabilities:
            results = await detector.detect_selected_capabilities(selected_capabilities)
        else:
            results = await detector.detect_all_capabilities()
        
        # 保存结果
        detection_results[task_id] = results.to_dict()
        
        # 更新进度
        if task_id in detection_progress:
            detection_progress[task_id]["status"] = "completed"
            detection_progress[task_id]["progress"] = 100
        
        logger.info(f"检测任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"检测任务失败: {e}")
        if task_id in detection_progress:
            detection_progress[task_id]["status"] = "error"
            detection_progress[task_id]["error"] = str(e)
    finally:
        # 标记任务为非活跃状态
        active_tasks[task_id] = False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
