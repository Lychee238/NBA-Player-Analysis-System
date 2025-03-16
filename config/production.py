import os
from .default import Config as DefaultConfig

class Config(DefaultConfig):
    """生产环境配置类"""
    # Flask应用配置
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'production-secret')
    
    # 缓存配置
    CACHE_TYPE = "RedisCache"
    CACHE_REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    CACHE_REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    CACHE_REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    CACHE_DEFAULT_TIMEOUT = 7200  # 2小时
    
    # 日志配置
    LOG_LEVEL = 'WARNING'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(process)d] - %(message)s'
    
    # 安全配置
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # 性能配置
    BATCH_SIZE = 5000
    MIN_SAMPLES = 500
    
    # 模型配置
    MODEL_PARAMS = {
        'n_estimators': 200,
        'random_state': 42,
        'n_jobs': -1  # 使用所有CPU核心
    } 