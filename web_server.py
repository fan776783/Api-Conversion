#!/usr/bin/env python3
"""
Web服务器启动脚本
"""
import sys
import os
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 注册检测器
from core.openai_detector import OpenAICapabilityDetector
from core.anthropic_detector import AnthropicCapabilityDetector
from core.gemini_detector import GeminiCapabilityDetector
from core.capability_detector import CapabilityDetectorFactory

CapabilityDetectorFactory.register("openai", OpenAICapabilityDetector)
CapabilityDetectorFactory.register("anthropic", AnthropicCapabilityDetector)
CapabilityDetectorFactory.register("gemini", GeminiCapabilityDetector)

def main():
    # 导入环境配置
    from src.utils.env_config import env_config

    parser = argparse.ArgumentParser(description="AI API统一转换代理系统Web服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=env_config.web_port, help=f"服务器端口 (默认: {env_config.web_port})")
    parser.add_argument("--reload", action="store_true", help="开启自动重载 (开发模式)")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")

    args = parser.parse_args()

    # 检查调试模式：命令行参数或环境变量
    debug_mode = args.debug or os.environ.get("DEBUG_CONVERTER", "").lower() in ("1", "true", "yes")
    if debug_mode:
        from src.utils.logger import enable_debug
        enable_debug(True)
        print("🔧 调试模式已启用 (DEBUG_CONVERTER)")

    # 验证配置
    config_errors = env_config.validate_config()
    if config_errors:
        print("❌ 配置验证失败:")
        for error in config_errors:
            print(f"   - {error}")
        sys.exit(1)
    
    # 验证数据库连接
    print("🔧 验证数据库连接...")
    try:
        from src.utils.database import db_manager
        # 触发数据库初始化
        db_manager._ensure_initialized()
        print(f"✅ 数据库连接成功 ({env_config.database_type})")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        print("请检查数据库配置后重试")
        sys.exit(1)
    
    print("🚀 启动AI API统一转换代理系统...")
    print(f"📖 访问 http://localhost:{args.port} 查看Web界面")
    print(f"📚 API文档: http://localhost:{args.port}/docs")
    print(f"🌐 服务器地址: {args.host}:{args.port}")
    
    if args.reload:
        print("⚠️  开发模式：自动重载已启用")
    
    import uvicorn
    
    # 设置日志级别
    log_level = "debug" if args.debug else "info"
    
    # 启动服务器
    uvicorn.run(
        "api.web_api:app",  # 使用import string格式
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level
    )

if __name__ == "__main__":
    main()