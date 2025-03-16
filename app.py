from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_caching import Cache
from services.data_loader import DataLoader
from services.model_trainer import ModelTrainer
from routes.player_routes import player_bp, PlayerRoutes
from config import get_config
from utils.logger import init_app as init_logger
from utils.decorators import log_route_access, log_errors
import logging
from jsonschema import ValidationError
import os
from werkzeug.utils import secure_filename

# 获取配置
config = get_config()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """创建并配置Flask应用"""
    app = Flask(__name__)
    
    # 配置应用
    app.config.from_object(config)
    config.init_app(app)
    
    # 禁用浏览器缓存
    @app.after_request
    def add_header(response):
        if request.path.startswith('/api'):
            response.headers.update({
                'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0'
            })
        return response
    
    # 配置缓存
    app.config.update(
        CACHE_TYPE='filesystem',
        CACHE_DIR='cache',
        CACHE_DEFAULT_TIMEOUT=300,  # 5分钟过期
        CACHE_THRESHOLD=1000  # 最大缓存条目数
    )
    cache = Cache(app)
    
    # 添加缓存清理路由
    @app.route('/api/cache/clear', methods=['POST'])
    def clear_cache():
        try:
            cache.clear()
            return jsonify({'message': '缓存已清除'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    # 配置日志
    init_logger(app)
    logger.info("开始初始化应用程序...")

    try:
        # 初始化数据加载器
        logger.info("初始化数据加载器...")
        data_loader = DataLoader()
        if data_loader.df is None or data_loader.df.empty:
            logger.error("数据加载失败")
            raise RuntimeError("Failed to load data")
        logger.info(f"数据加载完成，共 {len(data_loader.df)} 条记录")

        # 初始化模型训练器
        logger.info("初始化模型训练器...")
        model_trainer = ModelTrainer()
        model_trainer.train_prediction_models(data_loader.df)
        logger.info("模型训练完成")

        # 注册路由
        logger.info("注册路由...")
        PlayerRoutes(data_loader, model_trainer, cache)
        app.register_blueprint(player_bp)
        logger.info("路由注册完成")

        # CORS设置
        CORS(app)
        
        # 全局错误处理
        @app.errorhandler(ValidationError)
        @log_errors
        def handle_validation_error(error):
            return jsonify({
                'error': 'Validation failed',
                'message': error.message
            }), 400

        @app.errorhandler(404)
        @log_errors
        def handle_not_found(error):
            return jsonify({
                'error': 'Resource not found',
                'message': str(error)
            }), 404

        @app.errorhandler(500)
        @log_errors
        def handle_server_error(error):
            logger.error(f"Server error: {str(error)}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'Please try again later'
            }), 500

        def allowed_file(filename):
            """检查文件扩展名是否允许"""
            return '.' in filename and \
                   filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

        @app.route('/upload', methods=['POST'])
        @log_route_access
        @log_errors
        def handle_upload():
            """处理文件上传"""
            try:
                if 'file' not in request.files:
                    logger.warning('No file part')
                    return jsonify({'error': 'No file part'}), 400
                    
                file = request.files['file']
                if file.filename == '':
                    logger.warning('No file selected')
                    return jsonify({'error': 'No file selected'}), 400
                    
                if not allowed_file(file.filename):
                    logger.warning(f'Unsupported file type: {file.filename}')
                    return jsonify({'error': 'Unsupported file type, only CSV files are allowed'}), 400
                    
                # 安全地获取文件名并保存文件
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                logger.info(f'Saving file: {filename}')
                file.save(filepath)
                
                # 清除缓存
                cache.clear()
                logger.info('Cache cleared')
                
                # 重新初始化数据加载器
                try:
                    logger.info('Reinitializing data loader')
                    data_loader = DataLoader(filepath)
                    data_loader.preprocess_data()
                    
                    logger.info('Retraining models')
                    model_trainer = ModelTrainer()
                    model_trainer.train_prediction_models(data_loader.df)
                    
                    logger.info(f'File {filename} processed successfully')
                    return jsonify({
                        'success': True,
                        'message': 'File uploaded successfully, data updated',
                        'filename': filename
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
                    # 如果处理失败，删除上传的文件
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.info(f'Deleted failed upload file: {filename}')
                    return jsonify({
                        'error': 'File processing failed, please ensure correct format',
                        'details': str(e)
                    }), 500
                    
            except Exception as e:
                logger.error(f"Error during file upload: {str(e)}", exc_info=True)
                return jsonify({
                    'error': 'File upload failed',
                    'details': str(e)
                }), 500
        
        logger.info("应用程序初始化完成")
        return app
    except Exception as e:
        logger.error(f"应用程序初始化失败: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    app = create_app()
    app.run()