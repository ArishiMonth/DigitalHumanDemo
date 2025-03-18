# -*- coding: UTF-8 -*-
import pymysql
from dbutils.pooled_db import PooledDB

from yj_config.env_config import config

'''
@功能：PT数据库连接池
'''


class PTConnectionPool(object):
    __pool = None

    def __enter__(self):
        self.conn = self.__getConn()
        self.cursor = self.conn.cursor()
        print
        u"PT数据库创建con和cursor"
        return self

    def __getConn(self):
        if self.__pool is None:
            print(f'环境配置:', config)
            self.__pool = PooledDB(creator=pymysql, mincached=config.DB_MIN_CACHED, maxcached=config.DB_MAX_CACHED,
                                   maxshared=config.DB_MAX_SHARED, maxconnections=config.DB_MAX_CONNECYIONS,
                                   blocking=config.DB_BLOCKING, maxusage=config.DB_MAX_USAGE,
                                   setsession=config.DB_SET_SESSION,
                                   host=config.DB_HOST, port=config.DB_PORT,
                                   user=config.DB_USER, passwd=config.DB_PASSWORD,
                                   db=config.DB_DBNAME, use_unicode=False, charset=config.DB_CHARSET,
                                   connect_timeout=60, read_timeout=30, write_timeout=20)
        return self.__pool.connection()

    """
    @summary: 释放连接池资源
    """

    def __exit__(self, type, value, trace):
        self.cursor.close()
        self.conn.close()
        print(u"PT连接池释放con和cursor")

    # 重连接池中取出一个连接
    def getconn(self):
        conn = self.__getConn()
        cursor = conn.cursor()
        return cursor, conn


# 关闭连接归还给连接池
# def close(self):
#     self.cursor.close()
#     self.conn.close()
#     print u"PT连接池释放con和cursor";


def getPTConnection():
    return PTConnectionPool()
