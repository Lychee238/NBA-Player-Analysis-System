import os
from pathlib import Path

class Config:
    """默认配置类"""
    # 基础路径配置
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # 数据目录配置
    MODEL_DIR = BASE_DIR / "model"
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Flask应用配置
    DEBUG = True
    SECRET_KEY = 'dev'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 文件上传配置
    ALLOWED_EXTENSIONS = {'csv', 'json'}
    
    # 缓存配置
    CACHE_TYPE = "FileSystemCache"
    CACHE_DEFAULT_TIMEOUT = 3600
    CACHE_THRESHOLD = 1000
    CACHE_DIR = str(BASE_DIR / "cache")  # 确保缓存目录是字符串类型
    
    # 数据处理配置
    BATCH_SIZE = 1000
    MIN_SAMPLES = 100
    
    # 模型配置
    MODEL_PARAMS = {
        'n_estimators': 100,
        'random_state': 42
    }
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = BASE_DIR / 'app.log'
    
    @classmethod
    def init_app(cls, app):
        """初始化应用配置"""
        # 确保必要的目录存在
        directories = [
            Path(cls.MODEL_DIR),
            Path(cls.UPLOAD_FOLDER),
            Path(cls.CACHE_DIR)
        ]
        for directory in directories:
            directory.mkdir(exist_ok=True)
            
        # 配置日志
        import logging
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(str(cls.LOG_FILE))
            ]
        ) 