# 球员评估接口的验证模式
evaluate_schema = {
    "type": "object",
    "properties": {
        "player_name": {
            "type": "string",
            "minLength": 1,
            "description": "球员姓名"
        },
        "_timestamp": {
            "type": "number",
            "description": "时间戳"
        },
        "_nonce": {
            "type": "string",
            "description": "随机数"
        }
    },
    "required": ["player_name"],
    "additionalProperties": True
}

# 球员搜索接口的验证模式
search_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "搜索关键词"
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "description": "限制结果数量"
        }
    },
    "required": ["query"],
    "additionalProperties": False
}

# 年份筛选接口的验证模式
year_schema = {
    "type": "object",
    "properties": {
        "year": {
            "type": "integer",
            "minimum": 1946,
            "maximum": 2024,
            "description": "赛季年份"
        }
    },
    "required": ["year"],
    "additionalProperties": False
} 