import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Header, BackgroundTasks
from pubsub import pub
from starlette.requests import Request
from starlette.responses import JSONResponse

from configs.env_config import config
from entity.musetalkParam import MuseTalkTemplateParam, MuseTalkParam
from scripts.utils.mysqlhelp import mysqlHelper
from scripts.utils.rabbitmq import ExampleConsumer
from talk_generate import generate_template_upload, generate_human_upload

# load model weights

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_access_token_by_sql()
    consumer.consume(callback)
    yield
    # 设置回调函数，处理接收到的消息
    # try:
    #     channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    #     channel.start_consuming()
    # except Exception as ex:
    #     logger.error('消费配置出现错误,',ex)
    # yield


app = FastAPI(lifespan=lifespan)


# =====================================
# 获取accessToken
# =====================================
def get_access_token_by_sql():
    sql = "select prop_value from tb_sys_config where prop_name='user.auth.accessToken' "
    rows = mysqlHelper.selectone(sql)
    if rows is None:
        pass
    else:
        config.ACCESS_TOKEN = rows[0].decode('utf-8')
        print(f"access_token:", config.ACCESS_TOKEN)


# 自定义中间件处理 OPTIONS 请求
@app.middleware("http")
async def options_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        # 返回允许的 HTTP 方法和其他相关头部信息
        headers = {
            "Access-Control-Allow-Origin": config.cors_allowOrigins,
            "Access-Control-Allow-Methods": config.cors_allowMethods,
            "Access-Control-Allow-Headers": config.cors_allowHeaders
        }
        # encoded_list = [item.encode('utf-8') for item in headers]
        return JSONResponse(content=None, status_code=200, headers=headers)
    # 如果不是 OPTIONS 请求，则继续处理请求
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = config.cors_allowOrigins
    response.headers["Access-Control-Allow-Methods"] = config.cors_allowMethods
    response.headers["Access-Control-Allow-Headers"] = config.cors_allowHeaders
    return response


def callback(ch, method_frame, _header_frame, body):
    json_obj = json.loads(body.decode('utf-8'))
    print(f" [x] Received {json_obj}")
    try:
        if json_obj['humanEventType'] == 1:
            param = MuseTalkTemplateParam(id=json_obj['id'], type=json_obj['type'], bboxShift=json_obj['bboxShift'],sourceUrl=json_obj['sourceUrl'], audioUrl=json_obj['audioUrl'])
            generate_template_upload(param)
        else:
            param = MuseTalkParam(id=json_obj['id'], audioUrl=json_obj['audioUrl'],backgroundUrl=json_obj['backgroundUrl'])
            generate_human_upload(param, '')
    except Exception as ex:
        logger.error('消费出现错误,', ex)
    ch.basic_ack(delivery_tag=method_frame.delivery_tag)

consumer = ExampleConsumer()
if __name__ == '__main__':
    # get_access_token_by_sql()


    # threading.Thread(target=,args=callback).start()
    uvicorn.run("main:app", host='0.0.0.0', port=config.port, reload=False, log_level="info")


# =====================================
# 权限校验
# =====================================
def valid_permission(access_token):
    if access_token is None or access_token != config.ACCESS_TOKEN:
        return False
    return True

# =====================================
# 生成语言模板文件并上传到服务器
# =====================================
@app.post("/generateTemplateSync")
def generate_template_sync(params: MuseTalkTemplateParam, background_tasks: BackgroundTasks,
                           access_token: Optional[str] = Header(None)):
    if not valid_permission(access_token):
        return {"errorCode": "00001", "message": "没有访问该接口的权限"}
    print('收到的参数：', params)
    background_tasks.add_task(generate_template_upload, param=params)
    return {"errorCode": "00000", "message": "success"}


# =====================================
# 生成语言文件并上传到服务器,并更新数据库
# =====================================
@app.post("/generateSync")
def generate_sync(params: MuseTalkParam,access_token: Optional[str] = Header(None)):
    if not valid_permission(access_token):
        return {"errorCode": "00001", "message": "没有访问该接口的权限"}
    print('收到的参数：', params)
    # 发布事件
    pub.sendMessage('topic', arg1=params)
    # background_tasks.add_task(generate_human_upload, param=params)
    return {"errorCode": "00000", "message": "success"}


# 执行创建MP4

# def generate_template_upload(param):
#     sql = 'update tb_digital_human_template set video_path = %s,human_status = %s,human_fail = %s where id = %s'
#     # 先将状态变为生成中
#     mysqlHelper.update(sql, ["", 1, "", param.id])
#     values = []
#     try:
#         result_path = main(param)
#         result_url = YjCommon.upload_file(result_path)
#         print(result_url)
#         file_path = os.path.join(config.result_dir, param.id + "_temp")
#         shutil.rmtree(file_path)
#         values = [result_url, 2, '', param.id]
#     except Exception as ex:
#         values = ["", 3, str(ex), param.id]
#     mysqlHelper.update(sql, values)
#
#
# lock = threading.Lock()
#
#
# def generate_human_upload(param: MuseTalkParam, arg2):
#     sql = 'update tb_digital_human set url = %s,human_status = %s,human_fail = %s where id = %s'
#     try:
#         tmp_param = query_digital_human(param)
#         print(tmp_param)
#         if tmp_param is None:
#             # 查询数据失败，生成失败
#             mysqlHelper.update(sql, ["", 3, "获取数字人信息失败", tmp_param.id])
#             pass
#         if tmp_param:
#             pass
#         else:
#             # 先将状态变为生成中
#             mysqlHelper.update(sql, ["", 1, "", tmp_param.id])
#             values = []
#             lock.acquire()
#             try:
#                 resutlPath = main(tmp_param)
#                 result_url = YjCommon.upload_file(resutlPath)
#                 print(result_url)
#                 file_path = os.path.join(config.result_dir, tmp_param.id + "_temp")
#                 shutil.rmtree(file_path)
#                 values = [result_url, 2, '', tmp_param.id]
#             except Exception as ex:
#                 values = ["", 3, str(ex), tmp_param.id]
#             finally:
#                 lock.release()
#             mysqlHelper.update(sql, values)
#     except Exception as ex:
#         logger.exception(ex)
#         logger.error(f'生成数字人失败：%s', ex)
#
#
# def query_digital_human(param: MuseTalkParam):
#     human_info = mysqlHelper.selectone("select human_template_id,human_status from tb_digital_human where id=%s", [param.id])
#     if human_info is None:
#         raise Exception('数字人信息找不到')
#     if human_info[1] == 2:
#         return True
#     human_template = mysqlHelper.selectone(
#         "select type,bbox_shift,video_path from tb_digital_human_template where id=%s", [human_info[0]])
#     return MuseTalkTemplateParam(id=param.id, audioUrl=param.audioUrl, type=human_template[0],
#                                  bboxShift=human_template[1], sourceUrl=human_template[2].decode('utf-8'))
#     # return {
#     #     'id': param.id,
#     #     "audioUrl": param.audioUrl,
#     #     "type": human_template[0],
#     #     "bboxShift": human_template[1],
#     #     "sourceUrl": human_template[2].decode('utf-8')
#     # }
#
#
def async_listener(arg1):
    # generate_human_upload(arg1)
    thread = threading.Thread(target=generate_human_upload, args=(arg1, ''))
    thread.start()


pub.subscribe(async_listener, 'topic')
