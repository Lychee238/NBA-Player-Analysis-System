import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from quantile_forest import RandomForestQuantileRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.features = ['Age', 'G', 'MP', 'FG%', '3P%', 'FT%']
        self.model_path = Path('model')
        self.X_train_dict = {}
        self.y_train_dict = {}
        
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
        
        if not self.model_path.exists():
            self.model_path.mkdir()

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
        
        # 评估所有模型类型
        for model_type, config in self.model_config.items():
            try:
                model = config['model']
                if config['requires_scaling']:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X.values
                
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

    def train_prediction_models(self, df):
        """训练多个预测模型"""
        # 添加高级特征
        df = self.add_advanced_features(df)
        scaler = StandardScaler()
        X_global = scaler.fit_transform(df[self.features])
        
        # 处理所有特征的异常值
        for feature in self.features:
            if feature in df.columns and df[feature].dtype in [np.float64, np.int64]:
                df[feature] = df[feature].replace([np.inf, -np.inf], 0)
                df[feature] = df[feature].fillna(0)
        
        # 计算所有场均数据
        metrics = {
            'PPG': 'PTS',
            'RPG': 'TRB',
            'APG': 'AST',
            'SPG': 'STL',
            'BPG': 'BLK'
        }
        
        for per_game, total in metrics.items():
            df[per_game] = np.where(df['G'] > 0, df[total] / df['G'], 0)
            
        # 为每个指标训练模型
        for metric in ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'PER', 'BPM', 'VORP']:
            try:
                logger.info(f"正在训练{metric}预测模型...")
                model_file = self.model_path / f'{metric}_model.joblib'
                
                # 准备训练数据
                X = df[self.features]
                y = df[metric]
                
                # 评估并选择最佳模型
                best_model_type = self.evaluate_model(metric, X, y)
                model_config = self.model_config[best_model_type]
                
                # 准备训练数据
                if model_config['requires_scaling']:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    scaler = None
                    X_scaled = X.values
                
                # 克隆并训练最终模型
                final_model = clone(model_config['model'])
                final_model.fit(X_scaled, y)
                
                # 保存模型信息
                model_info = {
                    'model_type': best_model_type,
                    'model': final_model,
                    'scaler': scaler,
                    'features': self.features,
                    'requires_scaling': model_config['requires_scaling']
                }
                
                joblib.dump(model_info, model_file)
                self.models[metric] = model_info
                logger.info(f"{metric}模型训练完成并保存")
                
            except Exception as e:
                logger.error(f"训练{metric}模型时出错: {str(e)}")
                continue

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
            
            # 准备预测数据
            X = latest_season[self.features]
            
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
                if model_key in self.models:
                    try:
                        model_info = self.models[model_key]
                        
                        # 数据处理
                        if model_info['requires_scaling']:
                            X_scaled = model_info['scaler'].transform(X)
                        else:
                            X_scaled = X.values
                        
                        # 预测
                        prediction = model_info['model'].predict(X_scaled)[0]
                        
                        # 计算置信区间（使用预测值的±10%）
                        lower_bound = prediction * 0.9
                        upper_bound = prediction * 1.1
                        
                        # 确保所有值非负
                        prediction = max(0, prediction)
                        lower_bound = max(0, lower_bound)
                        upper_bound = max(0, upper_bound)
                        
                        predictions[output_key] = {
                            'value': float(prediction),
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound)
                        }
                        
                        logger.info(f"{output_key}预测值: {prediction:.2f} (置信区间: {lower_bound:.2f} - {upper_bound:.2f})")
                        
                    except Exception as e:
                        logger.error(f"预测{model_key}时出错: {str(e)}")
                        predictions[output_key] = None
            
            return predictions
            
        except Exception as e:
            logger.error(f"预测过程出错: {str(e)}")
            return None 