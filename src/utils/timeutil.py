# -*- coding: utf-8 -*-
import re
from datetime import datetime, timedelta


class TimeUtil(object):
    @classmethod
    def parse_timezone(cls, timezone):
        """
        解析时区表示
        :param timezone: str eg: +8
        :return: dict{symbol, offset}
        """
        result = re.match(r'(?P<symbol>[+-])(?P<offset>\d+)', timezone)
        symbol = result.groupdict()['symbol']
        offset = int(result.groupdict()['offset'])

        return {
            'symbol': symbol,
            'offset': offset
        }

    @classmethod
    def convert_timezone(cls, dt, timezone="+0"):
        """默认是utc时间，需要"""
        result = cls.parse_timezone(timezone)
        symbol = result['symbol']

        offset = result['offset']

        if symbol == '+':
            return dt + timedelta(hours=offset)
        elif symbol == '-':
            return dt - timedelta(hours=offset)
        else:
            raise Exception('dont parse timezone format')


def now_str():
    utc_now = datetime.utcnow()
    # now = datetime.now()
    convert_now = TimeUtil.convert_timezone(utc_now, '+8')

    # print('utc_now    ', utc_now)
    # print('now        ', now)
    # print('convert_now', convert_now)

    """
    utc_now     2021-01-27 03:26:13.132189
    now         2021-01-27 11:26:13.132198
    convert_now 2021-01-27 11:26:13.132189
    """
    return convert_now.strftime('%Y-%m-%d %H-%M-%S')
