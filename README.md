# 技术指标计算库（Technical Index）

一个用于技术指标计算的 Python 项目，支持多种主流技术分析指标。

## 安装说明

### 1. 克隆项目并安装依赖

```bash
git clone https://github.com/AshinGau/technical-index.git
cd technical-index
pip install -e .
```

### 2. 开发环境安装

```bash
pip install -e ".[dev]"
```

## TA-Lib 依赖说明

本项目部分高级技术指标（如K线形态识别）依赖 [TA-Lib](https://mrjbq7.github.io/ta-lib/)。

### 安装系统级 TA-Lib C 库

**macOS:**
```bash
brew install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ta-lib
```

**CentOS/RHEL:**
```bash
sudo yum install -y epel-release
sudo yum install -y ta-lib
```

### 安装 Python 依赖

已在 `requirements.txt` 中添加：
```
TA-Lib>=0.4.28
```

推荐使用如下命令安装所有依赖：
```bash
pip install -r requirements.txt
```

如遇到 `ta-lib` 相关编译或找不到头文件等问题，请先确保系统已正确安装 ta-lib C 库。

---

## 示例代码

请参考 `examples/technical_indicators_example.py`，包含：
- 如何获取币安ETHUSDT数据
- 如何计算所有技术指标
- 如何自定义参数（如ma_periods、RSI等）
- 如何获取所有可用指标和参数

## 开发相关

### 运行测试

```bash
pytest
```

### 代码格式化

```bash
black .
```

### 代码检查

```bash
flake8 .
```

### 类型检查

```bash
mypy .
```

## 项目结构

```
technical-index/
├── main.py              # 主入口
├── technical_index/     # 主包
│   └── __init__.py
├── tests/               # 测试文件
│   └── __init__.py
├── pyproject.toml       # 项目配置
├── README.md           # 本文件
└── requirements.txt    # 生产依赖
```

## 许可证

MIT License
