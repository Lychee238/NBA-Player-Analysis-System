import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_path='Seasons_Stats.csv'):
        self.file_path = file_path
        self.df = None
        self.player_names = []
        self.cache_path = Path('cache')
        if not self.cache_path.exists():
            self.cache_path.mkdir()
        self.load_data()

    def load_data(self):
        """加载CSV数据文件，优先使用缓存"""
        cache_file = self.cache_path / 'processed_data.joblib'
        names_cache_file = self.cache_path / 'player_names.joblib'

        try:
            if cache_file.exists() and names_cache_file.exists():
                logger.info("Loading data from cache...")
                self.df = joblib.load(cache_file)
                self.player_names = joblib.load(names_cache_file)
                logger.info("Successfully loaded data from cache")
            else:
                logger.info("Loading data from CSV file...")
                self.df = pd.read_csv(self.file_path)
                logger.info("Successfully read CSV file")
                # 确保Player列是字符串类型
                self.df['Player'] = self.df['Player'].astype(str)
                # 获取唯一的球员名字并排序
                self.player_names = sorted(self.df['Player'].unique().tolist())
                logger.info(f"Total of {len(self.player_names)} unique players")
                
                # 保存到缓存
                logger.info("Saving data to cache...")
                joblib.dump(self.df, cache_file)
                joblib.dump(self.player_names, names_cache_file)
                logger.info("Data caching complete")

        except FileNotFoundError:
            logger.error(f"Error: File not found {self.file_path}")
            self.df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            self.df = pd.DataFrame()

    def preprocess_data(self):
        """数据预处理"""
        if self.df.empty:
            return self.df
        
        logger.info("Starting data preprocessing...")
        # 处理缺失值
        self.df = self.df.fillna(0)
        
        # 确保数值列的类型正确
        numeric_columns = ['PER', 'WS/48', 'TS%', 'USG%', 'BPM', 'VORP']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 按球员和年份排序
        logger.info("Sorting data by time sequence...")
        self.df = self.df.sort_values(['Player', 'Year'])
        
        # 添加赛季索引
        logger.info("Adding season index...")
        self.df['SeasonIndex'] = self.df.groupby('Player').cumcount() + 1
        
        # 计算赛季间隔
        logger.info("Calculating season intervals...")
        self.df['YearDiff'] = self.df.groupby('Player')['Year'].diff()
        
        # 计算生涯赛季数
        logger.info("Calculating career seasons...")
        self.df['TotalSeasons'] = self.df.groupby('Player')['Year'].transform('count')
        
        logger.info("Data preprocessing complete")
        return self.df

    def get_player_data(self, player_name):
        """获取指定球员的数据"""
        logger.info(f"正在获取球员数据: {player_name}")
        logger.debug(f"数据框大小: {self.df.shape}")
        logger.debug(f"数据框列: {list(self.df.columns)}")
        
        # 检查数据框是否为空
        if self.df is None or self.df.empty:
            logger.error("数据框为空")
            return pd.DataFrame()
            
        # 检查Player列是否存在
        if 'Player' not in self.df.columns:
            logger.error("数据框中缺少Player列")
            return pd.DataFrame()
            
        # 获取球员数据
        player_data = self.df[self.df['Player'] == player_name].sort_values('Year')
        
        # 记录结果
        if player_data.empty:
            logger.warning(f"未找到球员数据: {player_name}")
        else:
            logger.info(f"找到球员数据: {len(player_data)}行")
            logger.debug(f"球员数据年份范围: {player_data['Year'].min()} - {player_data['Year'].max()}")
            
        return player_data

    def get_all_players(self):
        """获取所有球员名单"""
        return self.player_names

    def calculate_career_stats(self, player_data):
        """计算球员生涯数据"""
        total_games = player_data['G'].sum()
        return {
            'G': int(total_games),
            'Seasons': int(len(player_data)),
            'Year_Start': int(player_data['Year'].min()),
            'Year_End': int(player_data['Year'].max()),
            'PTS': float(player_data['PTS'].sum() / total_games) if total_games > 0 else 0,
            'TRB': float(player_data['TRB'].sum() / total_games) if total_games > 0 else 0,
            'AST': float(player_data['AST'].sum() / total_games) if total_games > 0 else 0,
            'STL': float(player_data['STL'].sum() / total_games) if total_games > 0 else 0,
            'BLK': float(player_data['BLK'].sum() / total_games) if total_games > 0 else 0,
        }

    def calculate_league_averages(self):
        """计算联盟平均数据"""
        total_games = self.df['G'].sum()
        return {
            'PTS': self.df['PTS'].sum() / total_games if total_games > 0 else 0,
            'TRB': self.df['TRB'].sum() / total_games if total_games > 0 else 0,
            'AST': self.df['AST'].sum() / total_games if total_games > 0 else 0,
            'STL': self.df['STL'].sum() / total_games if total_games > 0 else 0,
            'BLK': self.df['BLK'].sum() / total_games if total_games > 0 else 0,
        }

    def calculate_percentile_ranks(self, career_stats):
        """计算球员数据的百分位排名"""
        def calculate_percentile_rank(series, value):
            return (series < value).mean() * 100

        return {
            'PPG': calculate_percentile_rank(self.df['PTS'] / self.df['G'], career_stats['PTS']),
            'RPG': calculate_percentile_rank(self.df['TRB'] / self.df['G'], career_stats['TRB']),
            'APG': calculate_percentile_rank(self.df['AST'] / self.df['G'], career_stats['AST']),
            'SPG': calculate_percentile_rank(self.df['STL'] / self.df['G'], career_stats['STL']),
            'BPG': calculate_percentile_rank(self.df['BLK'] / self.df['G'], career_stats['BLK']),
        }

    def get_yearly_stats(self, player_data):
        """获取球员的年度数据"""
        return {
            'years': player_data['Year'].astype(int).tolist(),
            'PTS': (player_data['PTS'] / player_data['G']).astype(float).tolist(),
            'TRB': (player_data['TRB'] / player_data['G']).astype(float).tolist(),
            'AST': (player_data['AST'] / player_data['G']).astype(float).tolist(),
            'STL': (player_data['STL'] / player_data['G']).astype(float).tolist(),
            'BLK': (player_data['BLK'] / player_data['G']).astype(float).tolist(),
            'FG%': (player_data['FG%'] * 100).astype(float).tolist(),
            '3P%': (player_data['3P%'] * 100).astype(float).tolist(),
            'FT%': (player_data['FT%'] * 100).astype(float).tolist(),
            'PER': player_data['PER'].astype(float).tolist(),
            'BPM': player_data['BPM'].astype(float).tolist(),
            'VORP': player_data['VORP'].astype(float).tolist(),
        } 