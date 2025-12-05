"""
环境变量配置管理器
统一管理所有环境变量配置
"""
import os
from typing import Optional, Union, List
from pathlib import Path

class EnvConfig:
    """环境变量配置管理器"""
    
    def __init__(self):
        self._load_env_file()
    
    def _load_env_file(self):
        """加载.env文件"""
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 移除引号
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        os.environ[key] = value
    
    def get_str(self, key: str, default: str = "") -> str:
        """获取字符串配置"""
        return os.getenv(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """获取整数配置"""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点数配置"""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔配置"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_list(self, key: str, default: List[str] = None, separator: str = ",") -> List[str]:
        """获取列表配置"""
        if default is None:
            default = []
        value = os.getenv(key, "")
        if not value:
            return default
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    # ================================
    # 管理员认证配置
    # ================================
    
    @property
    def admin_password(self) -> str:
        """管理员密码"""
        return self.get_str("ADMIN_PASSWORD", "admin123")
    

    
    # ================================
    # Web服务器配置
    # ================================

    @property
    def web_port(self) -> int:
        """Web服务器端口"""
        return self.get_int("WEB_PORT", 3000)
    
    # ================================
    # AI服务商配置
    # ================================
    
    @property
    def anthropic_max_tokens(self) -> int:
        """Anthropic最大token数"""
        return self.get_int("ANTHROPIC_MAX_TOKENS", 4096)

    @property
    def max_tokens_limit(self) -> int:
        """max_tokens 上限（0 表示不限制）"""
        return self.get_int("MAX_TOKENS_LIMIT", 0)

    # ================================
    # 数据库配置
    # ================================

    @property
    def database_type(self) -> str:
        """数据库类型 (sqlite|mysql)"""
        return self.get_str("DATABASE_TYPE", "sqlite").lower()

    @property
    def database_path(self) -> str:
        """数据库文件路径（SQLite使用）"""
        return self.get_str("DATABASE_PATH", "data/channels.db")

    @property
    def mysql_host(self) -> str:
        """MySQL主机地址"""
        return self.get_str("MYSQL_HOST", "localhost")

    @property
    def mysql_port(self) -> int:
        """MySQL端口"""
        return self.get_int("MYSQL_PORT", 3306)

    @property
    def mysql_user(self) -> str:
        """MySQL用户名"""
        return self.get_str("MYSQL_USER", "root")

    @property
    def mysql_password(self) -> str:
        """MySQL密码"""
        return self.get_str("MYSQL_PASSWORD", "")

    @property
    def mysql_database(self) -> str:
        """MySQL数据库名"""
        return self.get_str("MYSQL_DATABASE", "default_db")

    @property
    def mysql_socket(self) -> Optional[str]:
        """MySQL Socket路径（可选）"""
        socket_path = self.get_str("MYSQL_SOCKET", "")
        return socket_path if socket_path else None

    # ================================
    # 日志配置
    # ================================

    @property
    def log_level(self) -> str:
        """日志级别"""
        return self.get_str("LOG_LEVEL", "WARNING")

    @property
    def debug_mode(self) -> bool:
        """是否启用调试模式"""
        return self.get_bool("DEBUG_MODE", False)

    @property
    def log_file(self) -> str:
        """日志文件路径"""
        return self.get_str("LOG_FILE", "logs/app.log")

    @property
    def log_max_days(self) -> int:
        """日志文件保留天数"""
        return self.get_int("LOG_MAX_DAYS", 1)

    
    def validate_config(self) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []

        # 验证必需配置
        if not self.admin_password:
            errors.append("ADMIN_PASSWORD cannot be empty")

        # 验证数据库配置
        if self.database_type not in ["sqlite", "mysql"]:
            errors.append(f"DATABASE_TYPE must be 'sqlite' or 'mysql', got '{self.database_type}'")
        elif self.database_type == "sqlite":
            # 验证SQLite数据库路径
            db_dir = Path(self.database_path).parent
            if not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create database directory {db_dir}: {e}")
        elif self.database_type == "mysql":
            # 验证MySQL配置
            if not self.mysql_host:
                errors.append("MYSQL_HOST cannot be empty when using MySQL")
            if not self.mysql_user:
                errors.append("MYSQL_USER cannot be empty when using MySQL")
            if not self.mysql_database:
                errors.append("MYSQL_DATABASE cannot be empty when using MySQL")
            if not (1 <= self.mysql_port <= 65535):
                errors.append(f"MYSQL_PORT must be between 1 and 65535, got {self.mysql_port}")

        # 验证端口范围
        if not (1 <= self.web_port <= 65535):
            errors.append(f"WEB_PORT must be between 1 and 65535, got {self.web_port}")

        # 验证Anthropic最大token数
        if self.anthropic_max_tokens <= 0:
            errors.append(f"ANTHROPIC_MAX_TOKENS must be positive, got {self.anthropic_max_tokens}")

        # 验证日志配置
        if self.log_max_days <= 0:
            errors.append(f"LOG_MAX_DAYS must be positive, got {self.log_max_days}")

        # 验证日志文件路径
        if self.log_file:
            log_dir = Path(self.log_file).parent
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create log directory {log_dir}: {e}")

        return errors


# 全局配置实例
env_config = EnvConfig()
