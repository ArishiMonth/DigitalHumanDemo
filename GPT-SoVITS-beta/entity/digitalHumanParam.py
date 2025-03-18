from typing import Optional

from pydantic import BaseModel


class DigitalHumanParam(BaseModel):
    msg: str
    textLang: Optional[str] = None
    voiceName: Optional[str] = "zh-CN-YunjianNeural"
    id: str
    '''
      音速
    '''
    rate: Optional[int] = 0
    '''
      音量
    '''
    volume: Optional[int] = 0
    '''
      音高
    '''
    pitch: Optional[int] = 0
    audioTemplateUrl: Optional[str] = None
    promptText: Optional[str] = None
    promptLang: Optional[str] = 'zh'

# 实时请求部分
class DigitalHumanRealParam(DigitalHumanParam):
    msg: Optional[str] = ""
    voiceName: Optional[str] = "zh-CN-YunjianNeural"
    id: Optional[str] = None


# 异步生成部分
class DigitalHumanSyncParam(DigitalHumanParam):
    msg: Optional[str] = ""
    voiceName: Optional[str] = "zh-CN-YunjianNeural"
    id: Optional[str] = None


# 口型文件及语音生成
class LipsRealParam(BaseModel):
    msg: Optional[str] = ""
    text_lang: str = 'zh'
    # 语音模板id
    id: Optional[str] = None


class GPT_TTS_Request(BaseModel):
    text: str = None
    text_lang: str = 'zh'
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = 'zh'
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 24
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
    return_fragment: bool = False


