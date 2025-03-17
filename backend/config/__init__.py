import os
from .default import Config as DefaultConfig
from .production import Config as ProductionConfig

def get_config():
    """根据环境变量返回相应的配置类"""
    env = os.getenv('FLASK_ENV', 'development')
    if env == 'production':
        return ProductionConfig
    return DefaultConfig 