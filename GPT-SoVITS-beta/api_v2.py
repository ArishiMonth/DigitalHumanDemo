"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                   # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": False,      # bool. whether to return a streaming response.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400
    
"""
import base64
import datetime
import json
import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import librosa
import requests

import ttsHuman
from entity.digitalHumanParam import LipsRealParam, DigitalHumanParam, DigitalHumanRealParam, DigitalHumanSyncParam
from scheduleJob import scheduler
from yj_utils.common import YjCommon
from yj_utils.mysqlhelp import MysqlHelp

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))
template_human = None
import subprocess
import signal
from fastapi import BackgroundTasks, Header
from fastapi import FastAPI
import uvicorn
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
from yj_config.env_config import config, opt
import soundfile as sf
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
logger = logging.getLogger(__name__)

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

# device = args.device
port = 9880
host = '0.0.0.0'
argv = sys.argv

args = ['-o', '', '-f', 'json', '', '-r', 'phonetic']
STATUS_SUCCESS = "1"
STATUS_FAIL = "0"
mysqlHelper = MysqlHelp.getInstance()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_access_token_by_sql()
    yield


APP = FastAPI(lifespan=lifespan)


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = 'zh'
    ref_audio_path: str = 'output/xiaoxiao.wav'
    aux_ref_audio_paths: list = None
    prompt_lang: str = 'zh'
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut1"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    gpt_path_str: str = None
    sovits_path_str: str = None


def get_tts():
    return ttsHuman.GPTSoVITSTTS(config.DATA_PATH)


# tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts = get_tts()


# tts_pipeline = TTS(tts_config)
# print(tts_config)
# gpt_path = tts_config.t2s_weights_path
# sovits_path = tts_config.vits_weights_path


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
    # init_rasa_template_model()


# =====================================
# 实时语音文件生成
# =====================================
@APP.post("/generateWavFileByRealtime")
def generate_wavfile_by_realtime(params: DigitalHumanRealParam):
    print('收到的参数：', params)
    file_name = YjCommon.md5_encryption(params.msg)
    # 生成音频文件
    try:
        save_file = tts.generate_speech(params, file_name, False)
        print(save_file)
        parent_path = YjCommon.get_parent_path(save_file)
        save_wave_file = parent_path + file_name + ".wav"
        with open(save_wave_file, 'rb') as file:
            audio_base64 = base64.b64encode(file.read()).decode('utf-8')
        # tts.delete_file(save_wave_file)
        return {"errorCode": "00000", "data": {"audioBase64": audio_base64}}
    except Exception as ex:
        logger.error('generateWavFileByRealtime fail', ex)
        # 生成失败
        return {"errorCode": "00001", "message": f'语音生成失败，原因【{str(ex)}】'}


# =====================================
# 实时语音及嘴唇文件生成
# =====================================
@APP.post("/generateLipAndWavFile")
def generateLipAndWavFile(param: LipsRealParam):
    print('收到的参数：', param)
    try:
        result = get_voice_template(param.id, param.msg)
        print(f'获取到声音模板：{result}')
        if result is None:
            return {"errorCode": "00001", "message": "语音模板不存在"}
        file_name = YjCommon.md5_encryption(result.json())
        start_time =int(round(time.time() * 1000))
        # 生成音频文件
        save_file = tts.generate_speech(result, file_name, True)
        print(f'-----------语音生成执行时间{int(round(time.time() * 1000)) - start_time  }')
        print(save_file)
        start_time = int(round(time.time() * 1000))
        parent_path = YjCommon.get_parent_path(save_file)
        save_json_file = parent_path + file_name + ".json"
        save_wave_file = parent_path + file_name + ".wav"
        print(f'---------------获取目录时间{int(round(time.time() * 1000)) - start_time}')
        start_time = int(round(time.time() * 1000))
        generate_lip_json(save_json_file, save_wave_file)
        print(f'-----------------口型生成时间{int(round(time.time() * 1000)) - start_time}')
        start_time = int(round(time.time() * 1000))
        with open(save_json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        with open(save_wave_file, 'rb') as file:
            audio_base64 = base64.b64encode(file.read()).decode('utf-8')
        print(f'----------------转base64及加载json文件时间{int(round(time.time() * 1000)) - start_time}')
        return {"errorCode": "00000", "data": {"audioBase64": audio_base64, "jsonObj": data}}
    except Exception as ex:
        logger.error('generateWavFileByRealtime fail', ex)
        # 生成失败
        return {"errorCode": "00001", "message": f'语音生成失败，原因【{str(ex)}】'}


# =====================================
# 只生成wave文件并上传到服务器
# =====================================
@APP.post("/generateWaveFile")
def generate_wave_file(params: DigitalHumanSyncParam, background_tasks: BackgroundTasks,
                       access_token: Optional[str] = Header(None)):
    if not valid_permission(access_token):
        return {"errorCode": "00001", "message": "没有访问该接口的权限"}
    print('收到的参数：', params)
    background_tasks.add_task(generate_wave_file_upload, params=params)
    return {"errorCode": "00000", "message": "success"}


@APP.post("/changeModel")
async def change_model(param: BaseModel):
    return init_rasa_template_model()


def init_rasa_template_model():
    # voice_name, audio_url, prompt_text,prompt_lang = get_rasa_voice_template()
    # if voice_name is None:
    #     logger.error(f'没有设置客服模版')
    #     return {"errorCode": "00001", "message": "没有设置客服模版"}
    # model_dic = tts.get_model_dic()
    # tmp = []
    # if voice_name in model_dic:
    #     tmp = model_dic[voice_name]
    # else:
    #     return {"errorCode": "00001", "message": "找不到选择的模型"}
    # GPT_model_path = os.path.abspath('GPT_weights_v2/' + tmp['gpt_path_str'])
    # SoVITS_model_path = os.path.abspath('SoVITS_weights_v2/' + tmp['sovits_path_str'])
    # tts_pipeline.init_t2s_weights(GPT_model_path)
    # tts_pipeline.init_vits_weights(SoVITS_model_path)
    # ref_audio_path = os.path.abspath(f'row/voicedemo/{YjCommon.get_file_name(audio_url)}')
    # if os.path.exists(ref_audio_path) is False:
    #     ref_audio_path = YjCommon.download_file(audio_url, ref_audio_path)
    # print(f'获得的完整音频地址：{ref_audio_path}')
    # tts_pipeline.set_ref_audio(ref_audio_path)
    # tts_pipeline.set_prompt_text(prompt_text, prompt_lang)
    # logger.info('客服模型初始化完成')
    return {"errorCode": "00000", "message": "success"}


def generate_wave_file_upload(params: DigitalHumanSyncParam):
    # 先将状态变为生成中
    sql = 'update tb_digital_human set audio_status = %s,audio_fail = %s,audio_url = %s where id = %s'
    mysqlHelper.update(sql, [1, "", '', params.id])
    sync_resp = {
        "audioUrl": "",
        "id": params.id
    }
    file_name = YjCommon.md5_encryption(params.msg)
    audio_success = 0
    try:
        # 生成音频文件
        save_file = tts.generate_speech(params, file_name, False)
        y, sr = librosa.load(save_file, sr=None)
        trimed_signal, _ = librosa.effects.trim(y)
        sf.write(save_file, trimed_signal, sr)
        print(save_file)
        wave_url = YjCommon.upload_file(save_file)
        print(wave_url)
        # 生成成功
        mysqlHelper.update(sql, [2, "", wave_url, params.id])
        sync_resp["audioUrl"] = wave_url
        audio_success = 1
    except Exception as ex:
        # 生成失败
        mysqlHelper.update(sql, [3, str(ex), '', params.id])
    if audio_success == 1:
        notify_talk(sync_resp)


# =====================================
# 权限校验
# =====================================
def valid_permission(access_token):
    print(f"access_token1:", access_token)
    print(f"access_token:", config.ACCESS_TOKEN)
    if access_token is None or access_token != config.ACCESS_TOKEN:
        return False
    return True


def notify_talk(sync_resp):
    print(f"地址：", config.BASE_URL + "/digital/human/notifyHuman")
    print(f"请求参数：", sync_resp)
    try:
        response = requests.post(config.BASE_URL + "/digital/human/notifyHuman", json=sync_resp,
                                 headers={'access-token': config.ACCESS_TOKEN}, timeout=(5, 10))
        # 输出返回信息
        print(f"返回的信息：", response.text)
    except requests.exceptions.RequestException as e:
        print(e)


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


# =====================================
# 查询数据库关联信息
# =====================================
def get_voice_template(id, msg):
    global template_human
    if template_human is not None:
        template_human.msg = msg
        return template_human;
    print(f'get_voice_template')
    sql = 'select voice_name,model_code,rate,audio_template_url,prompt_text,prompt_lang from tb_digital_voice_template where deleted=0 and status=1 and id = %s limit 1'
    result = mysqlHelper.selectone(sql, [id])
    if result is None:
        return None
    else:
        template_human = DigitalHumanParam(msg=msg, voiceName=result[0], id=id, rate=int(result[2]), audioTemplateUrl=result[3].decode('utf-8'),
                                 promptText=result[4].decode('utf-8'), promptLang=result[5].decode('utf-8'))
        return template_human
    pass


# =====================================
# 查询rasa客服关联声音信息
# =====================================
def get_rasa_voice_template():
    sql = 'select voice_name,audio_template_url,prompt_text,prompt_lang from tb_digital_voice_template where deleted=0 and status=1 and id in(select voice_template_id from tb_rasa_human_setting) limit 1'
    result = mysqlHelper.selectone(sql)
    if result is None:
        return None, None, None, None
    else:
        return result[0].decode('utf-8'), result[1].decode('utf-8'), result[2].decode('utf-8'), result[3].decode(
            'utf-8')
    pass


if __name__ == "__main__":
    try:
        if host == 'None':  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        scheduler.start()
        uvicorn.run(app=APP, host=host, port=port, workers=1)
        scheduler.shutdown()
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
