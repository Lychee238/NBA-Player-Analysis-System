# NBA球员数据分析与预测系统

这是一个基于Python的NBA球员数据分析与预测系统，能够分析球员历史数据并预测未来表现。

## 功能特点

- 📊 **数据可视化**
  - 基础数据趋势图（得分、篮板、助攻）
  - 投篮命中率走势图
  - 防守数据分析图
  - 高级数据统计图
  - 球员能力雷达图

- 🔮 **数据预测**
  - 下赛季数据预测
  - 包含得分、篮板、助攻等核心数据
  - 使用随机森林分位数回归模型

- 📈 **统计分析**
  - 生涯数据统计
  - 赛季间表现对比
  - 联盟平均值对比

## 安装说明

1. 克隆项目
```bash
git clone [项目地址]
cd PlayerAnalysis
```

2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动应用
```bash
python app.py
```

2. 访问网页界面
- 打开浏览器访问 `http://localhost:5000`
- 在搜索框中输入球员姓名
- 点击查看详细数据分析

## 技术栈

- 后端：Python, Flask
- 数据分析：pandas, numpy, scikit-learn
- 机器学习：quantile-forest
- 前端：HTML, JavaScript, Chart.js
- 数据存储：CSV

## 数据来源

- NBA官方统计数据
- Basketball Reference

## 开发环境要求

- Python 3.8+
- 操作系统：Windows/Linux/MacOS
- 内存：4GB+
- 硬盘空间：500MB+

## 注意事项

- 首次运行时会自动下载并处理数据，可能需要一些时间
- 预测功能需要足够的历史数据支持
- 建议使用现代浏览器（Chrome、Firefox、Edge等）访问

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。 