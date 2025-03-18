import copy
import glob
import logging
import os
import pickle
import shutil
import threading

import cv2
import numpy as np
import torch
from tqdm import tqdm

from configs.env_config import config
from entity.musetalkParam import MuseTalkTemplateParam, MuseTalkParam
from musetalk.utils.blending import get_image
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.utils import load_all_model
from scripts.utils.common import YjCommon
from scripts.utils.mysqlhelp import mysqlHelper

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
logger = logging.getLogger(__name__)

audio_processor, vae, unet, pe = load_all_model()
ngpu = torch.cuda.device_count()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

timesteps = torch.tensor([0], device=device)
lock = threading.Lock()


@torch.no_grad()
def generate(args):
    global pe
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    # 访问 TaskConfig 对象的属性
    video_path = args.sourceUrl
    audio_path = args.audioUrl
    bbox_shift = args.bboxShift

    resultbasepath = os.path.join(args.result_dir, args.id + "_temp")
    os.makedirs(resultbasepath, exist_ok=True)
    local_video_path = os.path.join(resultbasepath, 'video')
    local_audio_path = os.path.join(resultbasepath, 'audio')
    os.makedirs(local_video_path, exist_ok=True)
    os.makedirs(local_audio_path, exist_ok=True)

    # 下载视频和音频文件
    local_video_path = os.path.join(local_video_path, YjCommon.get_file_name(video_path))
    local_audio_path = os.path.join(local_audio_path,YjCommon.get_file_name(audio_path))
    print(f'下载的视频1：{os.path.basename(video_path)}')
    print(f'下载的视频2：{local_video_path}')
    video_path = YjCommon.download_file(video_path, local_video_path)
    audio_path = YjCommon.download_file(audio_path, local_audio_path)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]

    output_basename = 'result'
    result_img_save_path = os.path.join(resultbasepath, output_basename)  # related to video & audio inputs
    crop_coord_save_path = os.path.join(result_img_save_path,
                                        input_basename + ".pkl")  # only related to video input
    os.makedirs(result_img_save_path, exist_ok=True)
    args.result_dir = resultbasepath

    output_vid_name = os.path.join(args.result_dir, args.id + ".mp4")

    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        os.system(cmd)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    elif get_file_type(video_path) == "image":
        input_img_list = [video_path, ]
        fps = args.fps
    elif os.path.isdir(video_path):  # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    else:
        raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

    # print(input_img_list)
    ############################################## extract audio feature ##############################################
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
    ############################################## preprocess input image  ##############################################
    background_img = None
    background_path = os.path.join(resultbasepath, 'audio')
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path, 'rb') as f:
            coord_list = pickle.load(f)
        if not args.is_template and args.back_img is not None and args.back_img != '':
            background_img_path = YjCommon.download_file(args.back_img,  os.path.join(background_path, os.path.basename(args.back_img)))
            background_img = cv2.imread(background_img_path)
        frame_list = read_imgs(input_img_list,background_img)
    else:
        print("extracting landmarks...time consuming")
        if not args.is_template and args.back_img is not None and args.back_img != '':
            background_img_path = YjCommon.download_file(args.back_img,  os.path.join(background_path, os.path.basename(args.back_img)))
            background_img = cv2.imread(background_img_path)
        coord_list, frame_list,remark = get_landmark_and_bbox(input_img_list, bbox_shift,background_img)
        if args.is_template:
            print(f'建议调参范围【{remark}】')
            sql = 'update tb_digital_human_template set suggestion = %s where id = %s'
            # 先将状态变为生成中
            mysqlHelper.update(sql, [remark, args.id])


        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)

    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    ############################################## inference batch by batch ##############################################
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
    res_frame_list = []
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))):
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                     dtype=unet.model.dtype)  # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)

        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)

    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i % (len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i % (len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except:
            #                 print(bbox)
            continue

        combine_frame = get_image(ori_frame, res_frame, bbox)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
    print(cmd_img2video)
    os.system(cmd_img2video)

    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
    print(cmd_combine_audio)
    os.system(cmd_combine_audio)

    os.remove("temp.mp4")
    shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")
    return output_vid_name

lock1 = threading.Lock()
def generate_template_upload(param):
    human_info = mysqlHelper.selectone("select human_status, bbox_shift,audio_url,source_url, video_url from tb_digital_human_template where id=%s",
                                       [param.id])
    print(f'查询数字人模板的信息{human_info}')
    video_url = ""
    if human_info[4] is not None:
        video_url = human_info[4].decode('utf-8')
    if human_info[0] == 1:
        pass
    else:
        if human_info[0] == 2 and int(human_info[1].decode('utf-8')) == param.bboxShift and human_info[2].decode('utf-8') == param.audioUrl and human_info[3].decode('utf-8') == param.sourceUrl:
            pass
        else:
            sql = 'update tb_digital_human_template set video_url = %s,human_status = %s,human_fail = %s where id = %s'
            # 先将状态变为生成中
            mysqlHelper.update(sql, [video_url, 1, "", param.id])
            values = []
            lock1.acquire()
            try:
                result_path = generate(param)
                result_url = YjCommon.upload_file(result_path)
                print(result_url)
                file_path = os.path.join(config.result_dir, param.id + "_temp")
                shutil.rmtree(file_path)
                values = [result_url, 2, '', param.id]
            except Exception as ex:
                logger.exception(ex)
                values = ["", 3, str(ex), param.id]
            mysqlHelper.update(sql, values)
            lock1.release()

lock = threading.Lock()
def generate_human_upload(param: MuseTalkParam, arg2):
    sql = 'update tb_digital_human set video_url = %s,human_status = %s,human_fail = %s where id = %s'
    try:
        tmp_param = query_digital_human(param)
        print(tmp_param)
        if tmp_param is None:
            pass
        else:
            # 先将状态变为生成中
            mysqlHelper.update(sql, ["", 1, "", tmp_param.id])
            lock.acquire()
            values = []
            try:
                resutlPath = generate(tmp_param)
                result_url = YjCommon.upload_file(resutlPath)
                print(result_url)
                file_path = os.path.join(config.result_dir, tmp_param.id + "_temp")
                shutil.rmtree(file_path)
                values = [result_url, 2, '', tmp_param.id]
            except Exception as ex:
                logger.exception(ex)
                values = ["", 3, str(ex), tmp_param.id]
            mysqlHelper.update(sql, values)
            lock.release()
    except Exception as ex:
        logger.exception(ex)
        logger.error(f'生成数字人失败：%s', ex)


def query_digital_human(param: MuseTalkParam):
    human_info = mysqlHelper.selectone("select human_template_id,human_status from tb_digital_human where id=%s",
                                       [param.id])
    if human_info is None:
        raise Exception('数字人信息找不到')
    if human_info[1] == 2:
        return None
    human_template = mysqlHelper.selectone(
        "select type,bbox_shift,source_url from tb_digital_human_template where id=%s", [human_info[0]])
    return MuseTalkTemplateParam(id=param.id, audioUrl=param.audioUrl, type=human_template[0],
                                 bboxShift=human_template[1], sourceUrl=human_template[2].decode('utf-8'),is_template=False,back_img=param.backgroundUrl)
