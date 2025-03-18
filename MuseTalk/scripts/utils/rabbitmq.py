# import pika
#
# from configs.env_config import config
#


# -*- coding: utf-8 -*-
# pylint: disable=C0111,C0103,R0205

import functools
import logging
import time

import pika
from retry import retry

from configs.env_config import config

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)


class ExampleConsumer(object):
    _connection = None

    def __init__(self):
        connectionParameters = pika.ConnectionParameters(
            host=config.rabbit_mq_host,
            port=config.rabbit_mq_port,
            connection_attempts=3,
            retry_delay=5,
            credentials=pika.PlainCredentials(config.rabbit_mq_username, config.rabbit_password)
        )

        self._connection = pika.BlockingConnection(connectionParameters)

    @property
    def connection(self):
        if self._connection is not None:
            return self._connection
        return reconnect()

    def setListener(self, func):
        # 创建通道
        channel = self._connection.channel()
        channel.exchange_declare(exchange='digitalHumanTopic', exchange_type='x-delayed-message', durable=True)
        # 声明一个队列
        queue_name = 'digitalHumanTopic.talk'
        channel.queue_declare(queue=queue_name)
        # 绑定队列到交换机（这里使用默认交换机）
        channel.queue_bind(
            exchange='digitalHumanTopic',
            queue=queue_name,
            routing_key='#'
        )
        channel.basic_qos(prefetch_count=1)
        channel.add_on_cancel_callback(self.cancel)
        channel.add_on_return_callback(self.on_return)
        on_message_callback = functools.partial(func)
        channel.basic_consume(queue_name, on_message_callback)
        return channel

    def reconnect(self):
        # 连接参数
        connectionParameters = pika.ConnectionParameters(
            host=config.rabbit_mq_host,
            port=config.rabbit_mq_port,
            connection_attempts=3,
            retry_delay=5,
            credentials=pika.PlainCredentials(config.rabbit_mq_username, config.rabbit_password)
        )

        self._connection = pika.BlockingConnection(connectionParameters)
        return self._connection
    def cancel(self,ch, method_frame, _header_frame, body):
        print('mq已经断连')
        if self._connection.is_closed:
            self.reconnect()
        else:
            self._connection.close()
            self.reconnect()
    def on_return(self,ch, method_frame, _header_frame, body):
        print('mq返回回调消息',body)



    def consume(self, func):
        print("retry connect")
        while(True):
            if self._connection.is_closed:
                self.reconnect()
            try:
                channel = self.setListener(func)
                channel.start_consuming()
            except Exception as ex:
                print('出现异常，可能是网络原因')
                LOGGER.error('Failed to prepare AMQP consumer',ex)
                time.sleep(30)
