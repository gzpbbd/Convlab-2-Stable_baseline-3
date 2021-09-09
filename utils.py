from functools import wraps
import time
import logging


# 装饰器：在定义其他函数时在前一行加入 "@calculate_time"
def calculate_time(func):
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        start = time.time()
        _result = func(*args, **kwargs)
        total_time = time.time() - start

        rest = total_time - int(total_time)
        total_time = int(total_time)
        logging.debug(
            'Running function \"{}\" spent time {:02}:{:02}:{:02}.{:03}'.format(func.__name__, total_time // 3600,
                                                                                total_time % 3600 // 60,
                                                                                total_time % 60, int(rest * 1000)))
        return _result

    return wrapped_function
