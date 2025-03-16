import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from quantile_forest import RandomForestQuantileRegressor
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
        # 添加训练数据存储
        self.X_train_dict = {}
        self.y_train_dict = {}
        if not self.model_path.exists():
            self.model_path.mkdir()

    def add_advanced_features(self, df):
        """添加高级特征"""
        logger.info("Adding advanced features...")
        
        # 赛季负荷特征 - 避免除以零
        df['MPG'] = np.where(df['G'] > 0, df['MP'] / df['G'], 0)
        
        # 效率复合指标 - 避免除以零
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
        
        logger.info(f"Added {len(self.features) - 6} new features")
        return df

    def evaluate_model(self, metric, X, y):
        """评估模型性能"""
        logger.info(f"Evaluating model for {metric}...")
        
        # 创建分位数回归森林
        model = RandomForestQuantileRegressor(
            n_estimators=100,
            max_depth=7,
            n_jobs=-1,
            random_state=42
        )
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练模型
        model.fit(X_scaled, y)
        
        # 预测并计算评估指标
        y_pred = model.predict(X_scaled)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        
        metrics = {'mae': mae, 'rmse': rmse}
        logger.info(f"Model evaluation results for {metric}: MAE={mae:.3f}, RMSE={rmse:.3f}")
        return metrics

    def train_prediction_models(self, df):
        """训练多个预测模型或加载已有模型"""
        # 添加高级特征
        df = self.add_advanced_features(df)
        
        # 处理所有特征的异常值
        for feature in self.features:
            if feature in df.columns and df[feature].dtype in [np.float64, np.int64]:
                # 将异常值替换为0
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
            # 安全地计算场均数据
            df[per_game] = np.where(df['G'] > 0, df[total] / df['G'], 0)
            
        # 为每个指标训练或加载模型
        for metric in ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'PER', 'BPM', 'VORP']:
            model_file = self.model_path / f'{metric}_model.joblib'
            scaler_file = self.model_path / f'{metric}_scaler.joblib'
            data_file = self.model_path / f'{metric}_training_data.joblib'
            
            if all(f.exists() for f in [model_file, scaler_file, data_file]):
                logger.info(f"Loading existing model and data: {metric}")
                self.models[metric] = joblib.load(model_file)
                self.scalers[metric] = joblib.load(scaler_file)
                training_data = joblib.load(data_file)
                self.X_train_dict[metric] = training_data['X']
                self.y_train_dict[metric] = training_data['y']
            else:
                logger.info(f"Training new model: {metric}")
                # 处理缺失值
                df_clean = df[self.features + [metric]].fillna(0)
                
                # 分割数据
                X = pd.DataFrame(df_clean[self.features])
                y = df_clean[metric]
                
                # 确保数据有效
                X = X.replace([np.inf, -np.inf], 0)
                y = y.replace([np.inf, -np.inf], 0)
                
                # 评估模型（在标准化之前）
                model_metrics = self.evaluate_model(metric, X, y)
                logger.info(f"Model performance for {metric}: {model_metrics}")
                
                # 存储原始训练数据
                self.X_train_dict[metric] = X
                self.y_train_dict[metric] = y
                
                # 标准化数据用于训练
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # 训练最终模型
                model = RandomForestQuantileRegressor(
                    n_estimators=100,
                    max_depth=7,
                    n_jobs=-1,
                    random_state=42
                )
                model.fit(X_scaled, y)
                
                # 保存模型、标准化器和训练数据
                joblib.dump(model, model_file)
                joblib.dump(scaler, scaler_file)
                joblib.dump({
                    'X': X,
                    'y': y
                }, data_file)
                
                self.models[metric] = model
                self.scalers[metric] = scaler

    def predict(self, player_data):
        """为球员预测下一赛季的数据"""
        try:
            logger.info("Starting prediction with player data")
            predictions = {}
            
            # 确保输入数据是DataFrame格式
            if isinstance(player_data, pd.Series):
                input_data = player_data.to_frame().T
            else:
                input_data = player_data.copy()
            
            # 获取最新一个赛季的数据
            latest_season = input_data.iloc[-1:].copy()
            
            logger.debug(f"Initial features: {list(latest_season.columns)}")
            
            # 检查必需的基础特征是否存在
            required_features = ['Age', 'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'Pos']
            missing_base_features = [f for f in required_features if f not in latest_season.columns]
            if missing_base_features:
                logger.error(f"Missing base features: {missing_base_features}")
                raise ValueError(f"Missing required base features: {missing_base_features}")
            
            # 添加高级特征
            try:
                latest_season['MPG'] = latest_season.apply(lambda x: x['MP'] / x['G'] if x['G'] > 0 else 0, axis=1)
                latest_season['EFF'] = latest_season.apply(
                    lambda x: (x['PTS'] + x['TRB'] + x['AST'] + x['STL'] + x['BLK']) / x['MP'] if x['MP'] > 0 else 0,
                    axis=1
                )
                logger.debug(f"Advanced features added: MPG={latest_season['MPG'].values[0]}, EFF={latest_season['EFF'].values[0]}")
            except Exception as e:
                logger.error(f"Error calculating advanced features: {str(e)}")
                raise
            
            # 处理位置编码
            pos_features = [col for col in self.features if col.startswith('Pos_')]
            if 'Pos' in latest_season.columns:
                pos = latest_season['Pos'].iloc[0]
                logger.warning(f"Position {pos} not found in training features")
                
                # 初始化所有位置特征为0
                for pos_feature in pos_features:
                    latest_season[pos_feature] = 0
                    
                # 设置当前位置为1
                pos_feature = f'Pos_{pos}'
                if pos_feature in pos_features:
                    latest_season[pos_feature] = 1
            
            # 确保所有必需的特征都存在
            for feature in self.features:
                if feature not in latest_season.columns:
                    latest_season[feature] = 0
                    logger.warning(f"Feature {feature} not found, setting to 0")
            
            # 选择需要的特征
            X = latest_season[self.features]
            
            # 对每个指标进行预测
            metrics_mapping = {
                'PPG': 'PTS',
                'RPG': 'TRB',
                'APG': 'AST',
                'SPG': 'STL',
                'BPG': 'BLK'
            }
            
            for model_key, output_key in metrics_mapping.items():
                try:
                    if model_key in self.models:
                        # 标准化数据
                        X_scaled = self.scalers[model_key].transform(X)
                        # 使用训练好的模型进行预测
                        prediction = self.models[model_key].predict(X_scaled)[0]
                        predictions[output_key] = float(prediction)  # 转换为Python float类型
                        logger.info(f"Prediction for {output_key}: {prediction:.2f}")
                except Exception as e:
                    logger.error(f"Error processing metric {model_key}: {str(e)}")
                    predictions[output_key] = 0.0
            
            # 添加其他指标的预测
            for metric in ['PER', 'BPM', 'VORP']:
                try:
                    if metric in self.models:
                        # 标准化数据
                        X_scaled = self.scalers[metric].transform(X)
                        # 使用训练好的模型进行预测
                        prediction = self.models[metric].predict(X_scaled)[0]
                        predictions[metric] = float(prediction)  # 转换为Python float类型
                        logger.info(f"Prediction for {metric}: {prediction:.2f}")
                except Exception as e:
                    logger.error(f"Error processing metric {metric}: {str(e)}")
                    predictions[metric] = 0.0
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise 