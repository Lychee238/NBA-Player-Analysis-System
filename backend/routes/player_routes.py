from flask import Blueprint, jsonify, request, render_template
from flask_expects_json import expects_json
from jsonschema import ValidationError
from .schemas import evaluate_schema, search_schema, year_schema
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

player_bp = Blueprint('player', __name__)

# 错误处理装饰器
@player_bp.errorhandler(400)
def bad_request(error):
    """处理错误请求"""
    if isinstance(error.description, ValidationError):
        return jsonify({
            'error': 'Validation failed',
            'message': error.description.message
        }), 400
    return jsonify({
        'error': 'Bad request',
        'message': str(error)
    }), 400

class PlayerRoutes:
    def __init__(self, data_loader, model_trainer, cache):
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.cache = cache
        self.setup_routes()

    def setup_routes(self):
        @player_bp.route('/')
        @self.cache.cached(timeout=3600)
        def index():
            return render_template('index.html', players=self.data_loader.get_all_players())

        @player_bp.route('/player/<player_name>')
        def player_detail(player_name):
            if not player_name or len(player_name.strip()) == 0:
                return jsonify({'error': 'Invalid player name'}), 400
            return render_template('player_detail.html', player_name=player_name)

        @player_bp.route('/api/players', methods=['GET'])
        @self.cache.cached(timeout=3600)
        def get_all_players():
            try:
                # 将 DataFrame 中的 NaN 值替换为 None，确保 JSON 序列化正确
                players = self.data_loader.df.where(pd.notnull(self.data_loader.df), None).to_dict('records')
                logger.info(f"Successfully retrieved {len(players)} player records")
                return jsonify(players)
            except Exception as e:
                logger.error(f"Error getting player list: {str(e)}", exc_info=True)
                return jsonify({'error': 'Failed to get player list'}), 500

        @player_bp.route('/api/players/<player_name>', methods=['GET'])
        @self.cache.memoize(timeout=3600)
        def get_player_by_name(player_name):
            if not player_name or len(player_name.strip()) == 0:
                return jsonify({'error': 'Please provide a valid player name'}), 400
            
            try:
                player_data = self.data_loader.df[
                    self.data_loader.df['Player'].str.contains(player_name, case=False, na=False)
                ]
                if player_data.empty:
                    return jsonify({'error': 'Player not found'}), 404
                return jsonify(player_data.to_dict('records'))
            except Exception as e:
                logger.error(f"Error querying player {player_name}: {str(e)}")
                return jsonify({'error': 'Failed to query player data'}), 500

        @player_bp.route('/api/players/evaluate', methods=['POST'])
        @expects_json(evaluate_schema)
        def evaluate_player():
            """评估球员数据并返回预测结果"""
            try:
                data = request.get_json()
                player_name = data.get('player_name')
                
                if not player_name:
                    return jsonify({'error': '未提供球员姓名'}), 400
                
                logger.info(f"开始评估球员: {player_name}")
                
                # 获取球员数据
                player_data = self.data_loader.get_player_data(player_name)
                if player_data is None or player_data.empty:
                    return jsonify({'error': f'未找到球员数据: {player_name}'}), 404
                
                # 计算生涯数据
                career_stats = self.data_loader.calculate_career_stats(player_data)
                
                # 计算联盟平均数据
                league_averages = self.data_loader.calculate_league_averages()
                
                # 生成预测
                prediction_result = self.model_trainer.predict(player_data)
                
                if prediction_result is None:
                    return jsonify({'error': '预测失败'}), 500
                
                logger.info(f"成功生成预测: {prediction_result}")
                
                # 处理yearly_stats中的NaN值
                yearly_stats = player_data.replace({float('nan'): None, float('inf'): None, float('-inf'): None})
                yearly_stats = yearly_stats.where(yearly_stats.notnull(), None)
                yearly_stats = yearly_stats.to_dict('records')
                
                # 处理career_stats中的数值
                processed_career_stats = {}
                for k, v in career_stats.items():
                    if isinstance(v, (int, float)):
                        if pd.isna(v) or np.isinf(v):
                            processed_career_stats[k] = None
                        else:
                            processed_career_stats[k] = float(v)
                    else:
                        processed_career_stats[k] = v
                
                # 处理league_averages中的数值
                processed_league_averages = {}
                for k, v in league_averages.items():
                    if isinstance(v, (int, float)):
                        if pd.isna(v) or np.isinf(v):
                            processed_league_averages[k] = None
                        else:
                            processed_league_averages[k] = float(v)
                    else:
                        processed_league_averages[k] = v
                
                # 准备返回数据
                response_data = {
                    'player_name': player_name,
                    'career_stats': processed_career_stats,
                    'yearly_stats': yearly_stats,
                    'predictions': prediction_result['predictions'],
                    'position_averages': prediction_result['position_averages'],
                    'league_averages': processed_league_averages,
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"评估球员时出错: {str(e)}", exc_info=True)
                return jsonify({'error': f'评估失败: {str(e)}'}), 500

        @player_bp.route('/api/players/names', methods=['GET'])
        @self.cache.cached(timeout=3600)
        def get_player_names():
            try:
                player_names = self.data_loader.get_all_players()
                if not player_names:
                    return jsonify({
                        'success': False,
                        'error': '没有可用的球员数据'
                    }), 500
                    
                return jsonify({
                    'success': True,
                    'players': player_names
                })
            except Exception as e:
                logger.error(f"获取球员名单时发生错误: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': '获取球员名单失败'
                }), 500

        # 添加缓存清理路由
        @player_bp.route('/api/cache/clear', methods=['POST'])
        def clear_cache():
            try:
                self.cache.clear()
                return jsonify({
                    'success': True,
                    'message': '缓存已清除'
                })
            except Exception as e:
                logger.error(f"清除缓存时发生错误: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': '清除缓存失败'
                }), 500 