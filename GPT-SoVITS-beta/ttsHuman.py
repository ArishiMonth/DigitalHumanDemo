import json
import os
import re
import shutil
import sys
import time
import uuid

import torch
from pydub import AudioSegment

file_separator = os.sep
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import datetime
import os
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf

from TTS_infer_pack.TTS import TTS_Config, TTS
from entity.digitalHumanParam import DigitalHumanParam, GPT_TTS_Request
from yj_utils.common import YjCommon
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if (os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if (name == "jieba.cache"): continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass


class HumanTTS:
    def __init__(self, path):
        self.savePath = path
        if not YjCommon.is_dir_exists(path):
            try:
                os.mkdir(path)
                print(f"文件目录 {path} 不存在，已创建。")
            except OSError as e:
                print(f"文件目录失败: {e}")

    def generate_speech(self, param: DigitalHumanParam, file_name, tmp_flag):
        return ""

    # 删除文件
    def delete_file(self, file):
        try:
            os.remove(file)
            print(f"文件 {file} 已被删除。")
        except OSError as e:
            print(f"文件删除失败: {e}")


# 设置静音阈值，默认是-16dB
def get_silence_end(y, sr, threshold=-16):
    # 分割音频，默认按静音分割
    split = librosa.effects.split(y, top_db=threshold)

    # 获取最后一个分割的结束时间
    end_time = split[-1][1] if len(split) > 0 else len(y) / sr

    # 计算最后一个分割结束的索引
    end_index = int(end_time * sr)

    return end_index


'''
  GPT-SoVITS
'''


class GPTSoVITSTTS(HumanTTS):
    tts_list = {}

    def __init__(self, path):
        super().__init__(path)

        model_dic = self.get_model_dic()
        for key, model in model_dic.items():
            tts_config1 = TTS_Config(os.path.abspath("GPT_SoVITS/configs/tts_infer.yaml"))
            temp = TTS(tts_config1)
            GPT_model_path = os.path.abspath('GPT_weights_v2/' + model['gpt_path_str'])
            SoVITS_model_path = os.path.abspath('SoVITS_weights_v2/' + model['sovits_path_str'])
            temp.init_t2s_weights(GPT_model_path)
            temp.init_vits_weights(SoVITS_model_path)
            self.tts_list[key] = temp

    def get_model_dic(self):
        with open('yj_config/model_dic.json', 'r', encoding='utf-8') as file:
            model_dic = json.load(file)
        return model_dic

    def generate_speech(self, param: DigitalHumanParam, file_name, tmp_flag):

        # tts_pipeline = TTS(tts_config1)
        model_dic = self.get_model_dic()
        tmp = []
        if param.voiceName in model_dic:
            tmp = model_dic[param.voiceName]
        else:
            raise Exception('找不到选择的模型')
        current_date = datetime.datetime.today().date()
        parent_path = self.savePath + file_separator + current_date.strftime("%Y%m%d")
        req = GPT_TTS_Request(text_lang=param.textLang, text_split_method='cut5',
                              ref_audio_path=tmp['ref_audio_path'], text=param.msg,
                              prompt_text=tmp['prompt_text'], prompt_lang=tmp['prompt_lang'], top_k=tmp['top_k']).dict()
        if param.audioTemplateUrl is not None:
            ref_audio_path = os.path.abspath(f'{self.savePath + file_separator}/voicedemo/{YjCommon.get_file_name(param.audioTemplateUrl)}')
            if os.path.exists(ref_audio_path) is False:
                ref_audio_path = YjCommon.download_file(param.audioTemplateUrl, ref_audio_path)
            req['ref_audio_path'] = ref_audio_path
        if param.promptText is not None:
            req['prompt_text'] = param.promptText
            req['prompt_lang'] = param.promptLang
        if param.textLang is not None:
            req['text_lang'] = param.textLang
        print('在获取模型之前')
        tts_pipeline = self.tts_list[param.voiceName]
        print(tts_pipeline.configs)
        return self.run(param, file_name, tmp_flag, tts_pipeline, req, parent_path)

    def run(self, param: DigitalHumanParam, file_name, tmp_flag, tts_pipeline: TTS, req, parent_path):
        save_file = parent_path + file_separator + file_name + ".wav"
        tmp_file = self.savePath + file_separator + str(uuid.uuid4()) + ".wav"
        # 如果文件已经存在，直接返回
        if tmp_flag and os.path.exists(save_file):
            return os.path.abspath(save_file)
        if not YjCommon.is_dir_exists(parent_path):
            try:
                os.mkdir(parent_path)
                print(f"文件目录 {parent_path} 不存在，已创建。")
            except OSError as e:
                print(f"文件目录创建失败: {e}")
                raise e
        speed = (100 + param.rate) / 100
        print(f'语速：{speed}')
        req["speed_factor"] = speed
        print(f'拿到的参数：{req}')
        text_list, break_seconds = self.break_text(req['text'])
        audio_data_all = np.zeros(0, dtype=np.int16)
        sr = None
        for index, sub_text in enumerate(text_list):
            req['text'] = sub_text
            start_time = int(round(time.time() * 1000))
            tts_generator = tts_pipeline.run(req)
            sr, audio_data = next(tts_generator)
            print(f'-----------语音生成执行时间:::{int(round(time.time() * 1000)) - start_time}')
            if audio_data_all.size == 0:
                audio_data_all = audio_data
            else:
                audio_data_all = np.concatenate((audio_data_all, audio_data), 0)
            if len(break_seconds) > index and break_seconds[index] > 0.1:
                empty = self.generate_silence(break_seconds[index], sr)
                audio_data_all = np.concatenate((audio_data_all, empty), 0)
        sf.write(save_file, audio_data_all, sr, format='wav')
        # self.synthesize(GPT_model_path, SoVITS_model_path, tmp['ref_audio_path'], tmp['ref_text'], tmp['text_lang'],
        #                 param.msg, tmp['text_lang'], save_file)
        # y, sr = librosa.load(tmp_file, sr=None)
        # trimed_signal, _ = librosa.effects.trim(y)
        # sf.write(save_file, trimed_signal, sr)
        #  self.delete_file(tmp_file)
        return save_file

    def pack_audio(self, io_buffer: BytesIO, data: np.ndarray, rate: int):
        sf.write(io_buffer, data, rate, format='wav')
        io_buffer.seek(0)
        return io_buffer

    # 语音合成前分割文本
    def break_text(self, text):
        pattern = r'\[break_\d+(?:\.\d+)\]|\[break_\d+]'
        text_list = re.split(pattern, text)
        find_list = re.findall(pattern, text)
        break_seconds = []
        for str in find_list:
            break_seconds.append(float(str.replace('break_', '').replace('[', '').replace(']', '')))
        print(f'停顿分割后{text_list}，{break_seconds}')
        return text_list, break_seconds

    def generate_silence(self, duration_seconds, sr):
        silence = AudioSegment.silent(duration=duration_seconds * 1000, frame_rate=sr)
        return silence.get_array_of_samples()

    # def synthesize(self, GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text, ref_language, target_text,
    #                target_language, output_wav_path):
    #
    #     # Change model weights
    #     change_gpt_weights(gpt_path=GPT_model_path)
    #     change_sovits_weights(sovits_path=SoVITS_model_path)
    #
    #     # Synthesize audio
    #     synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path,
    #                                    prompt_text=ref_text,
    #                                    prompt_language=i18n('中英混合'),
    #                                    text=target_text,
    #                                    text_language=i18n('中英混合'), top_p=1, temperature=1)
    #
    #     result_list = list(synthesis_result)
    #
    #     if result_list:
    #         last_sampling_rate, last_audio_data = result_list[-1]
    #         sf.write(output_wav_path, last_audio_data, last_sampling_rate)
    #         print(f"Audio saved to {output_wav_path}")
