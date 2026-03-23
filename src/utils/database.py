"""
数据库管理器
支持SQLite和MySQL数据库，用于存储渠道信息和系统配置
"""
import sqlite3
import pymysql
import os
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import contextmanager

from src.utils.logger import setup_logger
from src.utils.env_config import env_config
from src.utils.encryption import encryption_manager

logger = setup_logger("database")


class DatabaseManager:
    """数据库管理器，支持SQLite和MySQL"""
    
    def __init__(self, db_path: str = None):
        self.db_type = env_config.database_type
        self.db_path = db_path or env_config.database_path
        self._initialized = False
        
        if self.db_type == "sqlite":
            self._ensure_data_dir()
        
        # 立即验证数据库连接，不再使用懒加载
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """确保数据库已初始化"""
        if not self._initialized:
            try:
                self._init_database()
                self._initialized = True
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                # 直接抛出异常，不再自动回退到SQLite
                raise RuntimeError(f"Failed to initialize {self.db_type} database: {e}")
    
    def _ensure_data_dir(self):
        """确保SQLite数据目录存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _get_raw_connection(self):
        """获取原始数据库连接（不检查初始化状态）"""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        elif self.db_type == "mysql":
            connect_params = {
                'host': env_config.mysql_host,
                'port': env_config.mysql_port,
                'user': env_config.mysql_user,
                'password': env_config.mysql_password,
                'database': env_config.mysql_database,
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,
                'autocommit': False,
                'connect_timeout': 5,   # 减少连接超时时间
                'read_timeout': 10,    # 减少读取超时时间  
                'write_timeout': 10,   # 减少写入超时时间
                'ssl_disabled': False  # Enable SSL for cloud databases
            }
            
            if env_config.mysql_socket:
                connect_params['unix_socket'] = env_config.mysql_socket
                connect_params.pop('host', None)
                connect_params.pop('port', None)
            
            return pymysql.connect(**connect_params)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _execute_query(self, conn, query: str, params: tuple = None):
        """执行查询，自动处理SQLite和MySQL的差异"""
        cursor = conn.cursor()
        
        if self.db_type == "mysql":
            # 将SQLite的?占位符转换为MySQL的%s占位符
            query = query.replace('?', '%s')
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        return cursor
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        self._ensure_initialized()
        
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
        elif self.db_type == "mysql":
            # MySQL连接参数
            connect_params = {
                'host': env_config.mysql_host,
                'port': env_config.mysql_port,
                'user': env_config.mysql_user,
                'password': env_config.mysql_password,
                'database': env_config.mysql_database,
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,  # 使结果可以通过列名访问
                'autocommit': False,
                'connect_timeout': 5,   # 5秒连接超时
                'read_timeout': 10,     # 10秒读取超时
                'write_timeout': 10,    # 10秒写入超时
                'ssl_disabled': False   # Enable SSL for cloud databases
            }
            
            # 如果配置了socket路径，则使用socket连接
            if env_config.mysql_socket:
                connect_params['unix_socket'] = env_config.mysql_socket
                # 使用socket时不需要host和port
                connect_params.pop('host', None)
                connect_params.pop('port', None)
            
            conn = pymysql.connect(**connect_params)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        try:
            yield conn
        except Exception as e:
            if hasattr(conn, 'rollback'):
                conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_raw_connection()
        try:
            cursor = conn.cursor()
            
            if self.db_type == "sqlite":
                # SQLite表结构
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS channels (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        base_url TEXT NOT NULL,
                        api_key TEXT NOT NULL,
                        custom_key TEXT UNIQUE NOT NULL,
                        timeout INTEGER DEFAULT 30,
                        max_retries INTEGER DEFAULT 3,
                        enabled BOOLEAN DEFAULT 1,
                        models_mapping TEXT,
                        default_target_format TEXT,
                        supported_formats TEXT,
                        use_proxy BOOLEAN DEFAULT 0,
                        proxy_type TEXT,
                        proxy_host TEXT,
                        proxy_port INTEGER,
                        proxy_username TEXT,
                        proxy_password TEXT,
                        payload_config TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_config (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
            elif self.db_type == "mysql":
                # MySQL表结构
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS channels (
                        id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        provider VARCHAR(100) NOT NULL,
                        base_url TEXT NOT NULL,
                        api_key TEXT NOT NULL,
                        custom_key VARCHAR(255) UNIQUE NOT NULL,
                        timeout INT DEFAULT 30,
                        max_retries INT DEFAULT 3,
                        enabled TINYINT(1) DEFAULT 1,
                        models_mapping TEXT,
                        default_target_format VARCHAR(100),
                        supported_formats TEXT,
                        use_proxy TINYINT(1) DEFAULT 0,
                        proxy_type VARCHAR(20),
                        proxy_host VARCHAR(255),
                        proxy_port INT,
                        proxy_username VARCHAR(255),
                        proxy_password TEXT,
                        payload_config TEXT,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_config (
                        `key` VARCHAR(255) PRIMARY KEY,
                        `value` TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                ''')
            
            conn.commit()
            
            # 进行数据库迁移 - 添加代理字段（如果不存在）
            self._migrate_proxy_fields(cursor, conn)
            
            # 进行数据库迁移 - 添加 payload_config 字段（如果不存在）
            self._migrate_payload_config_field(cursor, conn)

            # 进行数据库迁移 - 添加 target format 字段（如果不存在）
            self._migrate_target_format_fields(cursor, conn)
            
            logger.info(f"Database ({self.db_type}) initialized successfully")
        finally:
            conn.close()

    def _migrate_proxy_fields(self, cursor, conn):
        """迁移数据库，添加代理字段（如果不存在）"""
        try:
            if self.db_type == "sqlite":
                # 检查是否已经存在代理字段
                cursor.execute("PRAGMA table_info(channels)")
                columns = [row[1] for row in cursor.fetchall()]
                
                proxy_fields = ['use_proxy', 'proxy_type', 'proxy_host', 'proxy_port', 'proxy_username', 'proxy_password']
                for field in proxy_fields:
                    if field not in columns:
                        if field == 'use_proxy':
                            cursor.execute(f"ALTER TABLE channels ADD COLUMN {field} BOOLEAN DEFAULT 0")
                        elif field in ['proxy_port']:
                            cursor.execute(f"ALTER TABLE channels ADD COLUMN {field} INTEGER")
                        else:
                            cursor.execute(f"ALTER TABLE channels ADD COLUMN {field} TEXT")
                        logger.info(f"Added column {field} to channels table")
                        
            elif self.db_type == "mysql":
                # 检查是否已经存在代理字段
                cursor.execute("SHOW COLUMNS FROM channels")
                columns = [row['Field'] for row in cursor.fetchall()]
                
                proxy_fields = {
                    'use_proxy': 'TINYINT(1) DEFAULT 0',
                    'proxy_type': 'VARCHAR(20)',
                    'proxy_host': 'VARCHAR(255)',
                    'proxy_port': 'INT',
                    'proxy_username': 'VARCHAR(255)',
                    'proxy_password': 'TEXT'
                }
                
                for field, field_type in proxy_fields.items():
                    if field not in columns:
                        cursor.execute(f"ALTER TABLE channels ADD COLUMN {field} {field_type}")
                        logger.info(f"Added column {field} to channels table")
                        
            conn.commit()
        except Exception as e:
            logger.warning(f"Migration warning (proxy fields may already exist): {e}")
    
    def _migrate_payload_config_field(self, cursor, conn):
        """迁移数据库，添加 payload_config 字段（如果不存在）"""
        try:
            if self.db_type == "sqlite":
                cursor.execute("PRAGMA table_info(channels)")
                columns = [row[1] for row in cursor.fetchall()]
                if 'payload_config' not in columns:
                    cursor.execute("ALTER TABLE channels ADD COLUMN payload_config TEXT")
                    logger.info("Added column payload_config to channels table")
            elif self.db_type == "mysql":
                cursor.execute("SHOW COLUMNS FROM channels")
                columns = [row['Field'] for row in cursor.fetchall()]
                if 'payload_config' not in columns:
                    cursor.execute("ALTER TABLE channels ADD COLUMN payload_config TEXT")
                    logger.info("Added column payload_config to channels table")
            conn.commit()
        except Exception as e:
            logger.warning(f"Migration warning (payload_config field may already exist): {e}")

    def _migrate_target_format_fields(self, cursor, conn):
        """迁移数据库，添加 target format 相关字段（如果不存在）"""
        try:
            if self.db_type == "sqlite":
                cursor.execute("PRAGMA table_info(channels)")
                columns = [row[1] for row in cursor.fetchall()]
                if 'default_target_format' not in columns:
                    cursor.execute("ALTER TABLE channels ADD COLUMN default_target_format TEXT")
                    logger.info("Added column default_target_format to channels table")
                if 'supported_formats' not in columns:
                    cursor.execute("ALTER TABLE channels ADD COLUMN supported_formats TEXT")
                    logger.info("Added column supported_formats to channels table")
            elif self.db_type == "mysql":
                cursor.execute("SHOW COLUMNS FROM channels")
                columns = [row['Field'] for row in cursor.fetchall()]
                if 'default_target_format' not in columns:
                    cursor.execute("ALTER TABLE channels ADD COLUMN default_target_format VARCHAR(100)")
                    logger.info("Added column default_target_format to channels table")
                if 'supported_formats' not in columns:
                    cursor.execute("ALTER TABLE channels ADD COLUMN supported_formats TEXT")
                    logger.info("Added column supported_formats to channels table")
            conn.commit()
        except Exception as e:
            logger.warning(f"Migration warning (target format fields may already exist): {e}")
    
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
        channel_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        models_mapping_json = json.dumps(models_mapping) if models_mapping else None
        supported_formats_json = json.dumps(supported_formats) if supported_formats else None
        payload_config_json = json.dumps(payload_config) if payload_config else None
        
        # 验证API密钥不是明显的JavaScript错误信息
        if api_key.startswith('script.js:') or 'Uncaught TypeError' in api_key:
            logger.error(f"Rejecting JavaScript error message as API key: {api_key[:50]}...")
            raise ValueError("Invalid API key: JavaScript error message detected")
        
        # 加密API密钥
        encrypted_api_key = encryption_manager.encrypt_api_key(api_key)
        
        # 加密代理密码（如果存在）
        encrypted_proxy_password = None
        if proxy_password:
            encrypted_proxy_password = encryption_manager.encrypt_api_key(proxy_password)
        
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, '''
                    INSERT INTO channels 
                    (id, name, provider, base_url, api_key, custom_key, timeout, max_retries, 
                     enabled, models_mapping, default_target_format, supported_formats, payload_config, use_proxy, proxy_type, proxy_host, proxy_port, 
                     proxy_username, proxy_password, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    channel_id, name, provider, base_url, encrypted_api_key, custom_key,
                    timeout, max_retries, True, models_mapping_json, default_target_format, supported_formats_json, payload_config_json, use_proxy, proxy_type,
                    proxy_host, proxy_port, proxy_username, encrypted_proxy_password, now, now
                ))
                
                conn.commit()
                logger.info(f"Added new channel: {name} ({provider}) with ID: {channel_id}")
                return channel_id
            except (sqlite3.IntegrityError, pymysql.IntegrityError) as e:
                if "custom_key" in str(e):
                    raise ValueError(f"Custom key '{custom_key}' already exists")
                raise ValueError(f"Database integrity error: {e}")
    
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
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if base_url is not None:
            updates.append("base_url = ?")
            params.append(base_url)
        if api_key is not None:
            # 验证API密钥不是明显的JavaScript错误信息
            if api_key.startswith('script.js:') or 'Uncaught TypeError' in api_key:
                logger.error(f"Rejecting JavaScript error message as API key: {api_key[:50]}...")
                raise ValueError("Invalid API key: JavaScript error message detected")
            
            updates.append("api_key = ?")
            # 加密API密钥
            encrypted_api_key = encryption_manager.encrypt_api_key(api_key)
            params.append(encrypted_api_key)
        if custom_key is not None:
            updates.append("custom_key = ?")
            params.append(custom_key)
        if timeout is not None:
            updates.append("timeout = ?")
            params.append(timeout)
        if max_retries is not None:
            updates.append("max_retries = ?")
            params.append(max_retries)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(enabled)
        if models_mapping is not None:
            updates.append("models_mapping = ?")
            params.append(json.dumps(models_mapping))
        if default_target_format is not None:
            updates.append("default_target_format = ?")
            params.append(default_target_format)
        if supported_formats is not None:
            updates.append("supported_formats = ?")
            params.append(json.dumps(supported_formats))
        if payload_config is not None:
            updates.append("payload_config = ?")
            params.append(json.dumps(payload_config))
        if use_proxy is not None:
            updates.append("use_proxy = ?")
            params.append(use_proxy)
        if proxy_type is not None:
            updates.append("proxy_type = ?")
            params.append(proxy_type)
        if proxy_host is not None:
            updates.append("proxy_host = ?")
            params.append(proxy_host)
        if proxy_port is not None:
            updates.append("proxy_port = ?")
            params.append(proxy_port)
        if proxy_username is not None:
            updates.append("proxy_username = ?")
            params.append(proxy_username)
        if proxy_password is not None:
            updates.append("proxy_password = ?")
            # 加密代理密码
            encrypted_proxy_password = encryption_manager.encrypt_api_key(proxy_password)
            params.append(encrypted_proxy_password)
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(channel_id)
        
        with self.get_connection() as conn:
            try:
                cursor = self._execute_query(conn, f'''
                    UPDATE channels 
                    SET {", ".join(updates)}
                    WHERE id = ?
                ''', tuple(params))
                
                if cursor.rowcount == 0:
                    return False
                
                conn.commit()
                logger.info(f"Updated channel: {channel_id}")
                return True
            except (sqlite3.IntegrityError, pymysql.IntegrityError) as e:
                if "custom_key" in str(e):
                    raise ValueError(f"Custom key '{custom_key}' already exists")
                raise ValueError(f"Database integrity error: {e}")
    
    def delete_channel(self, channel_id: str) -> bool:
        """删除渠道"""
        with self.get_connection() as conn:
            cursor = self._execute_query(conn, "DELETE FROM channels WHERE id = ?", (channel_id,))
            if cursor.rowcount == 0:
                return False
            
            conn.commit()
            logger.info(f"Deleted channel: {channel_id}")
            return True
    
    def get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """获取渠道信息"""
        with self.get_connection() as conn:
            cursor = self._execute_query(conn, "SELECT * FROM channels WHERE id = ?", (channel_id,))
            row = cursor.fetchone()
            
            if row:
                channel = dict(row)
                if channel['models_mapping']:
                    channel['models_mapping'] = json.loads(channel['models_mapping'])
                if channel.get('supported_formats'):
                    channel['supported_formats'] = json.loads(channel['supported_formats'])
                if channel.get('payload_config'):
                    channel['payload_config'] = json.loads(channel['payload_config'])
                # 解密API密钥
                if channel['api_key']:
                    channel['api_key'] = encryption_manager.decrypt_api_key(channel['api_key'])
                # 解密代理密码
                if channel.get('proxy_password'):
                    channel['proxy_password'] = encryption_manager.decrypt_api_key(channel['proxy_password'])
                return channel
            return None
    
    def get_channel_by_custom_key(self, custom_key: str) -> Optional[Dict[str, Any]]:
        """根据自定义key获取渠道信息"""
        with self.get_connection() as conn:
            cursor = self._execute_query(conn, 
                "SELECT * FROM channels WHERE custom_key = ? AND enabled = 1", 
                (custom_key,)
            )
            row = cursor.fetchone()
            
            if row:
                channel = dict(row)
                if channel['models_mapping']:
                    channel['models_mapping'] = json.loads(channel['models_mapping'])
                if channel.get('supported_formats'):
                    channel['supported_formats'] = json.loads(channel['supported_formats'])
                if channel.get('payload_config'):
                    channel['payload_config'] = json.loads(channel['payload_config'])
                # 解密API密钥
                if channel['api_key']:
                    channel['api_key'] = encryption_manager.decrypt_api_key(channel['api_key'])
                # 解密代理密码
                if channel.get('proxy_password'):
                    channel['proxy_password'] = encryption_manager.decrypt_api_key(channel['proxy_password'])
                return channel
            return None
    
    def get_all_channels(self) -> List[Dict[str, Any]]:
        """获取所有渠道"""
        with self.get_connection() as conn:
            cursor = self._execute_query(conn, "SELECT * FROM channels ORDER BY created_at DESC")
            channels = []
            for row in cursor.fetchall():
                channel = dict(row)
                if channel['models_mapping']:
                    channel['models_mapping'] = json.loads(channel['models_mapping'])
                if channel.get('supported_formats'):
                    channel['supported_formats'] = json.loads(channel['supported_formats'])
                if channel.get('payload_config'):
                    channel['payload_config'] = json.loads(channel['payload_config'])
                # 解密API密钥
                if channel['api_key']:
                    channel['api_key'] = encryption_manager.decrypt_api_key(channel['api_key'])
                # 解密代理密码
                if channel.get('proxy_password'):
                    channel['proxy_password'] = encryption_manager.decrypt_api_key(channel['proxy_password'])
                channels.append(channel)
            return channels
    
    def get_enabled_channels(self) -> List[Dict[str, Any]]:
        """获取所有启用的渠道"""
        with self.get_connection() as conn:
            cursor = self._execute_query(conn, "SELECT * FROM channels WHERE enabled = 1 ORDER BY created_at DESC")
            channels = []
            for row in cursor.fetchall():
                channel = dict(row)
                if channel['models_mapping']:
                    channel['models_mapping'] = json.loads(channel['models_mapping'])
                if channel.get('supported_formats'):
                    channel['supported_formats'] = json.loads(channel['supported_formats'])
                if channel.get('payload_config'):
                    channel['payload_config'] = json.loads(channel['payload_config'])
                # 解密API密钥
                if channel['api_key']:
                    channel['api_key'] = encryption_manager.decrypt_api_key(channel['api_key'])
                # 解密代理密码
                if channel.get('proxy_password'):
                    channel['proxy_password'] = encryption_manager.decrypt_api_key(channel['proxy_password'])
                channels.append(channel)
            return channels
    
    def get_channels_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """按提供商获取渠道列表"""
        with self.get_connection() as conn:
            cursor = self._execute_query(conn,
                "SELECT * FROM channels WHERE provider = ? AND enabled = 1 ORDER BY created_at DESC", 
                (provider,)
            )
            channels = []
            for row in cursor.fetchall():
                channel = dict(row)
                if channel['models_mapping']:
                    channel['models_mapping'] = json.loads(channel['models_mapping'])
                if channel.get('supported_formats'):
                    channel['supported_formats'] = json.loads(channel['supported_formats'])
                if channel.get('payload_config'):
                    channel['payload_config'] = json.loads(channel['payload_config'])
                # 解密API密钥
                if channel['api_key']:
                    channel['api_key'] = encryption_manager.decrypt_api_key(channel['api_key'])
                # 解密代理密码
                if channel.get('proxy_password'):
                    channel['proxy_password'] = encryption_manager.decrypt_api_key(channel['proxy_password'])
                channels.append(channel)
            return channels
    
    def set_config(self, key: str, value: str):
        """设置系统配置"""
        now = datetime.now().isoformat()
        
        with self.get_connection() as conn:
            if self.db_type == "sqlite":
                cursor = self._execute_query(conn, '''
                    INSERT OR REPLACE INTO system_config (key, value, created_at, updated_at)
                    VALUES (?, ?, 
                        COALESCE((SELECT created_at FROM system_config WHERE key = ?), ?),
                        ?)
                ''', (key, value, key, now, now))
            elif self.db_type == "mysql":
                cursor = self._execute_query(conn, '''
                    INSERT INTO system_config (`key`, `value`, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON DUPLICATE KEY UPDATE
                    `value` = VALUES(`value`),
                    updated_at = VALUES(updated_at)
                ''', (key, value, now, now))
            conn.commit()
    
    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """获取系统配置"""
        with self.get_connection() as conn:
            if self.db_type == "sqlite":
                cursor = self._execute_query(conn, "SELECT value FROM system_config WHERE key = ?", (key,))
            elif self.db_type == "mysql":
                cursor = self._execute_query(conn, "SELECT `value` FROM system_config WHERE `key` = ?", (key,))
            row = cursor.fetchone()
            return row['value'] if row else default
    
    def delete_config(self, key: str) -> bool:
        """删除系统配置"""
        with self.get_connection() as conn:
            if self.db_type == "sqlite":
                cursor = self._execute_query(conn, "DELETE FROM system_config WHERE key = ?", (key,))
            elif self.db_type == "mysql":
                cursor = self._execute_query(conn, "DELETE FROM system_config WHERE `key` = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_configs_by_prefix(self, prefix: str) -> List[Dict[str, str]]:
        """获取指定前缀的所有配置"""
        with self.get_connection() as conn:
            if self.db_type == "sqlite":
                cursor = self._execute_query(conn,
                    "SELECT key, value FROM system_config WHERE key LIKE ?", 
                    (f"{prefix}%",)
                )
            elif self.db_type == "mysql":
                cursor = self._execute_query(conn,
                    "SELECT `key`, `value` FROM system_config WHERE `key` LIKE ?", 
                    (f"{prefix}%",)
                )
            results = cursor.fetchall()
            return [{"key": row["key"], "value": row["value"]} for row in results]
    
    def has_encrypted_api_keys(self) -> bool:
        """检查数据库中是否存在加密的API密钥"""
        try:
            with self.get_connection() as conn:
                cursor = self._execute_query(conn,
                    "SELECT COUNT(*) as count FROM channels WHERE api_key LIKE ?"
                    , ("encrypted:%",)
                )
                row = cursor.fetchone()
                return row['count'] > 0 if row else False
        except Exception:
            # 如果表不存在或查询失败，返回False
            return False
    



# 全局数据库管理器实例（懒加载）
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """获取数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

# 向后兼容的属性访问
class _DBManagerProxy:
    def __getattr__(self, name):
        return getattr(get_db_manager(), name)

db_manager = _DBManagerProxy()
