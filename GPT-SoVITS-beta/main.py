# -*-coding:utf-8-*-
"""
工程入口
老代码-已废弃

"""
import base64
import json
import os
import shutil
import sys
import queue
import signal
import subprocess
import threading
<<<<<<< HEAD:GPT-SoVITS-beta/main.py
import traceback
from io import BytesIO

import numpy as np

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

from contextlib import asynccontextmanager
from typing import Optional
=======
from typing import Optional, Dict
>>>>>>> parent of ba45930... tts:main.py

import requests
import uvicorn
from fastapi import FastAPI, Header, BackgroundTasks,Response
from starlette.requests import Request
from starlette.responses import JSONResponse

import ttsHuman
from config.env_config import config, opt
from entity.digitalHumanParam import DigitalHumanParam, DigitalHumanRealParam, DigitalHumanSyncParam
from scheduleJob import scheduler
<<<<<<< HEAD:GPT-SoVITS-beta/main.py
from yj_utils.common import YjCommon
from yj_utils.mysqlhelp import MysqlHelp
import soundfile as sf
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_access_token_by_sql()
    yield


app = FastAPI(lifespan=lifespan)
=======
from utils.common import YjCommon
from utils.mysqlhelp import MysqlHelp

app = FastAPI()
>>>>>>> parent of ba45930... tts:main.py


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


args = ['-o', '', '-f', 'json', '', '-r', 'phonetic']
STATUS_SUCCESS = "1"
STATUS_FAIL = "0"
mysqlHelper = MysqlHelp.getInstance()
tts = ttsHuman.HumanTTS(config.DATA_PATH)
q = queue.Queue(10)


# =====================================
# 获取accessToken
# =====================================
def get_access_token_by_sql():
    sql = "select prop_value from tb_sys_config where prop_name='user.auth.accessToken' "
    rows = mysqlHelper.selectone(sql)
    if rows is None:
        pass
    else:
        config.ACCESS_TOKEN = rows[0]


<<<<<<< HEAD:GPT-SoVITS-beta/main.py
def get_tts():
    return ttsHuman.GPTSoVITSTTS(config.DATA_PATH)


tts = get_tts()

if __name__ == "__main__":
    try:
        scheduler.start()
        uvicorn.run("main:app", host='0.0.0.0', port=config.port, reload=False, workers=1, log_level="info")
        scheduler.shutdown()
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
=======
if __name__ == '__main__':
    scheduler.start()
    get_access_token_by_sql()
    uvicorn.run("main:app", host='0.0.0.0', port=config.port, reload=True, workers=4)
    scheduler.shutdown()

>>>>>>> parent of ba45930... tts:main.py

# =====================================
# 权限校验
# =====================================
def valid_permission(access_token):
    if access_token is None or access_token != config.ACCESS_TOKEN:
        return False
    return True

@app.get("/tts")
async def tts_get_endpoint(
                        text: str = None,
                        text_lang: str = 'zh',
                        ref_audio_path: str = 'output/xiaoxiao.wav',
                        aux_ref_audio_paths:list = None,
                        prompt_lang: str = 'zh',
                        prompt_text: str = "",
                        top_k:int = 5,
                        top_p:float = 1,
                        temperature:float = 1,
                        text_split_method:str = "cut0",
                        batch_size:int = 1,
                        batch_threshold:float = 0.75,
                        split_bucket:bool = True,
                        speed_factor:float = 1.0,
                        fragment_interval:float = 0.3,
                        seed:int = -1,
                        media_type:str = "wav",
                        streaming_mode:bool = False,
                        parallel_infer:bool = True,
                        repetition_penalty:float = 1.35,
                        gpt_path_str:str = None,
                        sovits_path_str:str=None
                        ):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size":int(batch_size),
        "batch_threshold":float(batch_threshold),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "fragment_interval":fragment_interval,
        "seed":seed,
        "media_type":media_type,
        "streaming_mode":streaming_mode,
        "parallel_infer":parallel_infer,
        "repetition_penalty":float(repetition_penalty)
    }
    return await tts_handle(req)


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
            }
    returns:
        StreamingResponse: audio stream response.
    """
    media_type='wav'
    try:
        tts_generator = ttsHuman.tts_pipeline.run(req)
        sr, audio_data = next(tts_generator)
        audio_data = pack_audio(BytesIO(), audio_data, sr).getvalue()
        return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})
def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int):
    sf.write(io_buffer, data, rate, format='wav')
    io_buffer.seek(0)
    return io_buffer
# =====================================
# 离线文本转语音生成和口型文件生成
# =====================================
@app.post("/pushSpeechMsg")
def pushSpeechMsg(params: DigitalHumanParam, access_token: Optional[str] = Header(None)):
    if not valid_permission(access_token):
        return {"errorCode": "00001", "message": "没有访问该接口的权限"}
    print('收到的参数：', params)
    q.put(params)
    return {"errorCode": "00000", "message": "success"}


# =====================================
# 只生成wave文件并上传到服务器
# =====================================
@app.post("/generateWaveFile")
def generate_wave_file(params: DigitalHumanSyncParam, background_tasks: BackgroundTasks,
                       access_token: Optional[str] = Header(None)):
    if not valid_permission(access_token):
        return {"errorCode": "00001", "message": "没有访问该接口的权限"}
    print('收到的参数：', params)
    background_tasks.add_task(generate_wave_file_upload, params=params)
    return {"errorCode": "00000", "message": "success"}


# =====================================
# 实时语音生成
# =====================================
@app.post("/generateLipAndWavFile")
def generateLipAndWavFile(params: DigitalHumanRealParam):
    print('收到的参数：', params)
    if params.id is not None:
        result = get_digital_human_by_sql(params.id)
        if result is not None:
            return {"errorCode": "00000", "data": {"videoUrl": result[0], "jsonObj": result[1]}}
    file_name = YjCommon.md5_encryption(params.msg)
    # 生成音频文件
    save_file = tts.generate_speech(params.msg, params.voiceName, file_name)
    print(save_file)
    parent_path = YjCommon.get_parent_path(save_file)
    save_json_file = parent_path + file_name + ".json"
    save_wave_file = parent_path + file_name + ".wav"
    generate_lip_json(save_json_file, save_wave_file)
    with open(save_json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(save_wave_file, 'rb') as file:
        audio_base64 = base64.b64encode(file.read()).decode('utf-8')
    return {"errorCode": "00000", "data": {"audioBase64": audio_base64, "jsonObj": data}}


@app.post("/demo1")
def demo1(params: Dict[str, str]):
    print('收到的参数：', params)
    return {"errorCode": "00000"}


def generate_wave_file_upload(params: DigitalHumanSyncParam):
    sync_resp = {
        "msg": "",
        "waveUrl": "",
        "id": params.id,
        "status": STATUS_SUCCESS,
        "failReason": ""
    }
    file_name = YjCommon.md5_encryption(params.msg)
    try:
        # 生成音频文件
        save_file = tts.generate_speech(params.msg, params.voiceName, file_name)
        print(save_file)
        wave_url = upload(save_file)
        sync_resp["waveUrl"] = wave_url
    except Exception as ex:
        sync_resp["status"] = STATUS_FAIL
        sync_resp["failReason"] = str(ex)
    notify_talk(params.callbackUrl, sync_resp)


# 文本生成音频与口型,并更新入库
def generate_human(params: DigitalHumanParam):
    try:
        file_name = YjCommon.md5_encryption(params.msg)
        # 生成音频文件
        save_file = tts.generate_speech(params.msg, params.voiceName, file_name)
        print(save_file)
        parent_path = YjCommon.get_parent_path(save_file)
        generate_lip_json_to_sql(parent_path, file_name, params.id)
    except Exception as ex:
        print("语音生成或嘴唇生成发生异常,", ex.args)
        sql = 'update tb_rasa_digital_human set status = %s,fail_reason = %s where id = %s'
        values = [2, str(ex), params.id]
        mysqlHelper.update(sql, values)
    pass


# 生成口型文件
def generate_lip_json(saveJsonFile, saveWaveFile):
    # 如果文件已经存在，直接返回
    if os.path.exists(saveJsonFile):
        return saveJsonFile
    args[1] = saveJsonFile
    args[4] = saveWaveFile
    print("参数：", args)
    print(opt.rhubarbPath)
    subprocess.call([opt.rhubarbPath] + args)


pass


# 生成口型文件并上传到nginx,更新入库
def generate_lip_json_to_sql(parent_path, file_name, id):
    save_json_file = parent_path + file_name + ".json"
    save_wave_file = parent_path + file_name + ".wav"
    generate_lip_json(save_json_file, save_wave_file)
    success = 1
    mave_url = upload(save_wave_file)
    if len(mave_url) == 0:
        success = 2
    print(f'是都成功：', success)
    with open(save_json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(data)
    # 如果存在一个失败，则本次生成失败，都成功则需要将数据库中值置为成功
    sql = 'update tb_rasa_digital_human set video_url = %s,json_obj = %s,status = %s where id = %s'
    values = [mave_url, json.dumps(data), success, id]
    mysqlHelper.update(sql, values)


pass


# 最后进行文件上传到nginx
def upload(file_path):
    print(config.BASE_URL + "upload-api/upload")

    with open(file_path, 'rb') as file:
        response = requests.post(config.BASE_URL + "upload-api/upload", files={'file': file},
                                 params={'domainId': 1, 'dir': "video"},
                                 headers={'access-token': config.ACCESS_TOKEN})
    # 输出返回信息
    print(f"返回的信息：", response.text)
    resp_obj = json.loads(response.text)
    if "00000" == resp_obj['errorCode']:
        return resp_obj['data']
    else:
        return ""


# =====================================
# 查询数据库关联信息
# =====================================
def get_digital_human_by_sql(id):
    sql = 'select video_url,json_obj,status,relation_id from tb_rasa_digital_human where relation_id = %s limit 1'
    result = mysqlHelper.selectone(sql, [id])
    if result is None:
        return None
    else:
        if result[2] == 1:
            # 如果是已经生成语音直接返回
            return result
        else:
            return None
    pass


def notify_talk(callback_url, sync_resp):
    print(f"地址：", callback_url)
    print(f"请求参数：", sync_resp)
    response = requests.post(callback_url, json=sync_resp,
                             headers={'access-token': config.ACCESS_TOKEN})
    # 输出返回信息
    print(f"返回的信息：", response.text)


def worker():
    while True:
        if q.empty():
            continue
        item = q.get(block=True, timeout=3)
        generate_human(item)
        q.task_done()


# Turn-on the worker thread.
threading.Thread(target=worker, daemon=True).start()
