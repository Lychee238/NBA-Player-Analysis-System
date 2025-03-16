import functools
import time
from flask import current_app, request

def log_execution_time(f):
    """记录函数执行时间的装饰器"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        current_app.logger.debug(
            f'Function {f.__name__} executed in {(end_time - start_time):.2f} seconds'
        )
        return result
    return wrapper

def log_route_access(f):
    """记录路由访问信息的装饰器"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        current_app.logger.info(
            f'Route accessed: {request.method} {request.path}'
        )
        return f(*args, **kwargs)
    return wrapper

def log_errors(f):
    """记录异常信息的装饰器"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            current_app.logger.error(
                f'Error in {f.__name__}: {str(e)}',
                exc_info=True
            )
            raise
    return wrapper 