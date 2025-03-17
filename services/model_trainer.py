import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from quantile_forest import RandomForestQuantileRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from pathlib import Path
import logging
import tempfile
import hashlib
import json

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.features = ['Age', 'G', 'MP', 'FG%', '3P%', 'FT%']
        self.model_path = Path('model')
        self.cache_dir = Path(tempfile.gettempdir()) / "player_analysis_cache"
        self.X_train_dict = {}
        self.y_train_dict = {}
        self.feature_selectors = {}  # 存储每个指标的特征选择器
        self.df = None  # 存储完整的数据集
        
        # 新增模型配置字典
        self.model_config = {
            'rf': {
                'model': RandomForestQuantileRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1),
                'requires_scaling': True
            },
            'knn': {
                'model': KNeighborsRegressor(n_neighbors=5),
                'requires_scaling': True
            },
            'tree': {
                'model': DecisionTreeRegressor(max_depth=5, random_state=42),
                'requires_scaling': False
            },
            'ensemble': {
                'model': VotingRegressor([
                    ('rf', RandomForestQuantileRegressor(n_estimators=100, max_depth=7, random_state=42)),
                    ('knn', KNeighborsRegressor(n_neighbors=5)),
                    ('tree', DecisionTreeRegressor(max_depth=5, random_state=42))
                ]),
                'requires_scaling': True
            }
        }
        self.selected_model_type = 'rf'  # 默认模型类型
        
        # 创建缓存目录
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        
        # 创建模型目录
        if not self.model_path.exists():
            self.model_path.mkdir()
            
        # 检查并加载已保存的模型
        logger.info("正在检查已保存的模型...")
        metrics = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'PER', 'BPM', 'VORP']
        for metric in metrics:
            model_file = self.model_path / f'{metric}_model.joblib'
            if model_file.exists():
                try:
                    self.models[metric] = joblib.load(model_file)
                    logger.info(f"成功加载{metric}模型")
                except Exception as e:
                    logger.error(f"加载{metric}模型时出错: {str(e)}")
            else:
                logger.info(f"未找到{metric}模型文件，将在首次预测时训练新模型")

    def add_advanced_features(self, df):
        """添加高级特征"""
        logger.info("正在添加高级特征...")
        
        # 赛季负荷特征
        df['MPG'] = np.where(df['G'] > 0, df['MP'] / df['G'], 0)
        
        # 效率复合指标
        df['EFF'] = np.where(df['MP'] > 0,
                            (df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK']) / df['MP'],
                            0)
        
        # 处理无穷大和NaN值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # 位置编码
        position_dummies = pd.get_dummies(df['Pos'], prefix='Pos')
        df = pd.concat([df, position_dummies], axis=1)
        
        # 更新特征列表
        self.features.extend(['MPG', 'EFF'])
        self.features.extend(position_dummies.columns.tolist())
        
        logger.info(f"添加了 {len(self.features) - 6} 个新特征")
        return df

    def evaluate_model(self, metric, X, y):
        """评估所有模型类型的性能"""
        logger.info(f"正在评估{metric}的模型性能...")
        best_model = None
        best_score = float('inf')
        
        # 特征选择（保留前8个重要特征）
        selector = SelectKBest(f_regression, k=8)
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors[metric] = selector
        
        # 获取选中的特征名称
        selected_features = [self.features[i] for i in selector.get_support(indices=True)]
        logger.info(f"{metric}选中的特征: {selected_features}")
        
        # 评估所有模型类型
        for model_type, config in self.model_config.items():
            try:
                model = config['model']
                if config['requires_scaling']:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_selected)
                else:
                    X_scaled = X_selected
                
                # 交叉验证评估
                scores = cross_val_score(
                    model, X_scaled, y, 
                    cv=3, scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                rmse = np.sqrt(-scores.mean())
                
                if rmse < best_score:
                    best_score = rmse
                    best_model = model_type
                    
                logger.info(f"{model_type}模型RMSE: {rmse:.3f}")
                
            except Exception as e:
                logger.error(f"评估{model_type}模型时出错: {str(e)}")
        
        self.selected_model_type = best_model
        logger.info(f"选择{metric}的最佳模型: {best_model}，RMSE: {best_score:.3f}")
        return best_model

    def _get_cache_key(self, df):
        """生成数据的缓存键"""
        # 使用数据的基本特征生成哈希值
        data_hash = hashlib.md5()
        
        # 添加数据形状
        data_hash.update(str(df.shape).encode())
        
        # 添加列名
        data_hash.update(str(sorted(df.columns.tolist())).encode())
        
        # 添加数据的统计特征
        stats = df.describe()
        data_hash.update(str(stats.to_dict()).encode())
        
        return data_hash.hexdigest()

    def _get_cached_data(self, df):
        """获取缓存的预处理数据"""
        cache_key = self._get_cache_key(df)
        cache_file = self.cache_dir / f"preprocessed_data_{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                logger.info("找到缓存的预处理数据，正在加载...")
                cached_data = pd.read_pickle(cache_file)
                # 验证缓存数据的完整性
                required_features = set(self.features)
                if set(cached_data.columns).issuperset(required_features):
                    logger.info("成功加载缓存数据")
                    return cached_data
                else:
                    logger.warning("缓存数据特征不完整，需要重新预处理")
            except Exception as e:
                logger.error(f"加载缓存数据时出错: {str(e)}")
        
        return None

    def _cache_preprocessed_data(self, df):
        """缓存预处理后的数据"""
        try:
            cache_key = self._get_cache_key(df)
            cache_file = self.cache_dir / f"preprocessed_data_{cache_key}.pkl"
            df.to_pickle(cache_file)
            logger.info("成功缓存预处理数据")
        except Exception as e:
            logger.error(f"缓存数据时出错: {str(e)}")

    def train_prediction_models(self, df):
        """训练多个预测模型"""
        try:
            # 保存完整的数据集
            self.df = df.copy()
            logger.info(f"保存完整数据集，共 {len(self.df)} 条记录")
            
            # 尝试从缓存加载预处理数据
            preprocessed_df = self._get_cached_data(df)
            
            if preprocessed_df is None:
                logger.info("未找到缓存数据，开始预处理...")
                # 添加高级特征
                preprocessed_df = self.add_advanced_features(df)
                
                # 处理所有特征的异常值
                for feature in self.features:
                    if feature in preprocessed_df.columns and preprocessed_df[feature].dtype in [np.float64, np.int64]:
                        preprocessed_df[feature] = preprocessed_df[feature].replace([np.inf, -np.inf], 0)
                        preprocessed_df[feature] = preprocessed_df[feature].fillna(0)
                
                # 缓存预处理后的数据
                self._cache_preprocessed_data(preprocessed_df)
            
            # 计算所有场均数据
            metrics = {
                'PPG': 'PTS',
                'RPG': 'TRB',
                'APG': 'AST',
                'SPG': 'STL',
                'BPG': 'BLK'
            }
            
            for per_game, total in metrics.items():
                preprocessed_df[per_game] = np.where(preprocessed_df['G'] > 0, preprocessed_df[total] / preprocessed_df['G'], 0)
            
            # 为每个指标训练模型
            for metric in ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'PER', 'BPM', 'VORP']:
                try:
                    logger.info(f"正在训练{metric}预测模型...")
                    model_file = self.model_path / f'{metric}_model.joblib'
                    
                    # 准备训练数据
                    X = preprocessed_df[self.features]
                    y = preprocessed_df[metric]
                    
                    # 评估并选择最佳模型
                    best_model_type = self.evaluate_model(metric, X, y)
                    model_config = self.model_config[best_model_type]
                    
                    # 使用特征选择器选择特征
                    X_selected = self.feature_selectors[metric].transform(X)
                    
                    # 准备训练数据
                    if model_config['requires_scaling']:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_selected)
                    else:
                        scaler = None
                        X_scaled = X_selected
                    
                    # 克隆并训练最终模型
                    final_model = clone(model_config['model'])
                    final_model.fit(X_scaled, y)
                    
                    # 保存模型信息
                    model_info = {
                        'model_type': best_model_type,
                        'model': final_model,
                        'scaler': scaler,
                        'feature_selector': self.feature_selectors[metric],
                        'features': self.features,
                        'requires_scaling': model_config['requires_scaling']
                    }
                    
                    joblib.dump(model_info, model_file)
                    self.models[metric] = model_info
                    logger.info(f"{metric}模型训练完成并保存")
                    
                except Exception as e:
                    logger.error(f"训练{metric}模型时出错: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"训练模型时出错: {str(e)}")

    def calculate_position_averages(self, position):
        """计算同位置球员的平均数据"""
        try:
            # 获取主要位置（第一个位置）
            main_position = position.split('-')[0]
            logger.info(f"使用主要位置 {main_position} 计算平均数据")
            
            # 从self.df中筛选具有相同主要位置的球员数据
            position_mask = self.df['Pos'].apply(lambda x: str(x).split('-')[0] == main_position)
            all_position_players = self.df[position_mask]['Player'].unique()
            logger.info(f"找到 {len(all_position_players)} 名 {main_position} 位置的球员")
            
            # 从完整数据集中获取这些球员的所有数据
            position_data = self.df[self.df['Player'].isin(all_position_players)]
            
            if position_data.empty:
                logger.warning(f"未找到位置{main_position}的球员数据")
                return {}
            
            # 计算关键指标的平均值
            metrics = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'PER', 'BPM', 'VORP']
            averages = {}
            
            # 按球员分组计算每个球员的生涯平均值，然后再计算所有同位置球员的平均值
            for metric in metrics:
                if metric in position_data.columns:
                    # 对于基础数据，先计算每个球员的场均值
                    if metric in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
                        # 按球员分组计算场均值
                        player_averages = position_data.groupby('Player').apply(
                            lambda x: x[metric].sum() / x['G'].sum() if x['G'].sum() > 0 else 0
                        )
                        # 计算所有同位置球员的平均值
                        averages[metric] = float(player_averages.mean())
                    else:
                        # 对于高级数据，计算每个球员的平均值
                        player_averages = position_data.groupby('Player')[metric].mean()
                        averages[metric] = float(player_averages.mean())
            
            logger.info(f"已计算{main_position}位置的平均数据: {averages}")
            return averages
            
        except Exception as e:
            logger.error(f"计算位置平均值时出错: {str(e)}")
            return {}

    def predict(self, player_data):
        """预测球员数据"""
        try:
            logger.info("开始进行预测...")
            
            # 确保输入数据是DataFrame格式
            if isinstance(player_data, pd.Series):
                input_data = player_data.to_frame().T
            else:
                input_data = player_data.copy()
            
            # 获取最新一个赛季的数据
            latest_season = input_data.iloc[-1:].copy()
            
            # 获取球员位置
            player_position = latest_season['Pos'].iloc[0] if 'Pos' in latest_season.columns else None
            
            # 计算同位置球员平均数据
            position_averages = {}
            if player_position:
                # 使用self.df计算位置平均值
                position_averages = self.calculate_position_averages(player_position)
                logger.info(f"计算得到位置{player_position}的平均数据")
            
            # 尝试从缓存加载预处理数据
            preprocessed_data = self._get_cached_data(latest_season)
            
            if preprocessed_data is None:
                # 处理NaN值
                latest_season = latest_season.replace({np.nan: 0, np.inf: 0, -np.inf: 0})
                
                # 添加高级特征
                try:
                    latest_season['MPG'] = latest_season.apply(lambda x: x['MP'] / x['G'] if x['G'] > 0 else 0, axis=1)
                    latest_season['EFF'] = latest_season.apply(
                        lambda x: (x['PTS'] + x['TRB'] + x['AST'] + x['STL'] + x['BLK']) / x['MP'] if x['MP'] > 0 else 0,
                        axis=1
                    )
                except Exception as e:
                    logger.error(f"计算高级特征时出错: {str(e)}")
                    raise
                
                # 处理位置编码
                pos_features = [col for col in self.features if col.startswith('Pos_')]
                if 'Pos' in latest_season.columns:
                    pos = latest_season['Pos'].iloc[0]
                    
                    # 初始化所有位置特征为0
                    for pos_feature in pos_features:
                        latest_season[pos_feature] = 0
                        
                    # 设置当前位置为1
                    pos_feature = f'Pos_{pos}'
                    if pos_feature in pos_features:
                        latest_season[pos_feature] = 1
                
                # 确保所有特征都存在
                for feature in self.features:
                    if feature not in latest_season.columns:
                        latest_season[feature] = 0
                        logger.warning(f"特征{feature}不存在，设置为0")
                
                # 缓存预处理后的数据
                self._cache_preprocessed_data(latest_season)
                preprocessed_data = latest_season
            
            # 准备预测数据
            X = preprocessed_data[self.features]
            
            # 再次检查并处理NaN值
            X = X.replace({np.nan: 0, np.inf: 0, -np.inf: 0})
            
            predictions = {}
            metrics_mapping = {
                'PPG': 'PTS',
                'RPG': 'TRB',
                'APG': 'AST',
                'SPG': 'STL',
                'BPG': 'BLK',
                'PER': 'PER',
                'BPM': 'BPM',
                'VORP': 'VORP'
            }
            
            for model_key, output_key in metrics_mapping.items():
                try:
                    # 如果模型不存在，先训练模型
                    if model_key not in self.models:
                        logger.info(f"模型{model_key}不存在，开始训练...")
                        self.train_prediction_models(input_data)
                        
                    if model_key in self.models:
                        model_info = self.models[model_key]
                        
                        # 使用特征选择器选择特征
                        X_selected = model_info['feature_selector'].transform(X)
                        
                        # 数据处理
                        if model_info['requires_scaling']:
                            X_scaled = model_info['scaler'].transform(X_selected)
                        else:
                            X_scaled = X_selected
                        
                        # 预测
                        prediction = model_info['model'].predict(X_scaled)[0]
                        
                        # 计算置信区间（使用预测值的±10%）
                        lower_bound = prediction * 0.9
                        upper_bound = prediction * 1.1
                        
                        # 对于BPM允许负值，其他指标保持非负
                        if output_key != 'BPM':
                            prediction = max(0, prediction)
                            lower_bound = max(0, lower_bound)
                            upper_bound = max(0, upper_bound)
                        
                        predictions[output_key] = {
                            'value': float(prediction),
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound)
                        }
                        
                        logger.info(f"{output_key}预测值: {prediction:.2f} (置信区间: {lower_bound:.2f} - {upper_bound:.2f})")
                    else:
                        logger.error(f"无法加载或训练{model_key}模型")
                        predictions[output_key] = None
                        
                except Exception as e:
                    logger.error(f"预测{model_key}时出错: {str(e)}")
                    predictions[output_key] = None
            
            # 添加位置平均数据到返回结果
            return {
                'predictions': predictions,
                'position_averages': position_averages
            }
            
        except Exception as e:
            logger.error(f"预测过程出错: {str(e)}")
            return None 