#!/usr/bin/env python3
"""
获取币安交易数据
"""

import logging
import sys

import pandas as pd
import requests

# 配置日志记录器
logger = logging.getLogger(__name__)


def _handle_request_exception(e, response=None):
    """
    根据币安API错误代码处理不同类型的请求异常。

    :param e: 异常对象
    :param response: 响应对象（如果可用）
    """
    if response is not None:
        status_code = response.status_code

        if status_code == 429:
            # 请求频率超限
            logger.error(f"请求频率超限 (429)。请求被拒绝。错误: {e}")
            sys.exit(1)
        elif status_code == 418:
            # IP被封禁
            logger.error(f"IP已被自动封禁 (418)。错误: {e}")
            sys.exit(1)
        elif status_code == 400:
            logger.warning(f"错误请求 (400): 无效参数。错误: {e}")
        elif status_code == 401:
            logger.warning(f"未授权 (401): 无效的API密钥或签名。错误: {e}")
        elif status_code == 403:
            logger.warning(f"禁止访问 (403): 访问被拒绝。错误: {e}")
        elif status_code == 404:
            logger.warning(f"未找到 (404): 端点未找到。错误: {e}")
        elif status_code >= 500:
            logger.warning(f"服务器错误 ({status_code}): 币安服务器错误。错误: {e}")
        else:
            logger.warning(f"HTTP错误 ({status_code}): {e}")
    else:
        # 网络或其他连接问题
        logger.warning(f"网络或连接错误: {e}")


def get_futures_market_data(symbol, interval, limit=500):
    """
    从币安USDT-M合约获取市场数据（K线）。

    :param symbol: 交易对，例如 'ETHUSDT'
    :param interval: K线间隔，例如 '15m', '1h', '4h', '1d'
    :param limit: 数据点数量，最大1500
    :return: Pandas DataFrame 或 None
    """
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 对于错误响应抛出HTTPError
        data = response.json()

        # 将JSON数据转换为Pandas DataFrame
        df = pd.DataFrame(data)

        # 定义列名
        columns = [
            "Open_Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close_Time",
            "Quote_Asset_Volume",
            "Number_Of_Trades",
            "Taker_Buy_Base_Asset_Volume",
            "Taker_Buy_Quote_Asset_Volume",
            "Ignore",
        ]
        df.columns = columns

        # 将时间戳转换为可读格式（UTC）
        df["Open_Time"] = pd.to_datetime(df["Open_Time"], unit="ms")
        df["Close_Time"] = pd.to_datetime(df["Close_Time"], unit="ms")

        # 将数据类型转换为数值型
        convert_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Quote_Asset_Volume",
            "Number_Of_Trades",
            "Taker_Buy_Base_Asset_Volume",
            "Taker_Buy_Quote_Asset_Volume",
        ]
        for col in convert_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 仅保留核心数据列
        core_columns = [
            "Open_Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Number_Of_Trades",
            "Taker_Buy_Base_Asset_Volume",
            "Taker_Buy_Quote_Asset_Volume",
        ]
        return df[core_columns].set_index("Open_Time")

    except requests.exceptions.RequestException as e:
        # 处理不同类型的请求异常
        response = getattr(e, "response", None)
        _handle_request_exception(e, response)
        return None
