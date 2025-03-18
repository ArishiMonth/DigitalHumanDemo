import hashlib
import os

<<<<<<< HEAD:MuseTalk/scripts/utils/common.py
import requests

from configs.env_config import config


=======
>>>>>>> parent of ba45930... tts:utils/common.py
class YjCommon:
    # =====================================
    # MD5 加密
    # =====================================
    @staticmethod
    def md5_encryption(data):
        # 创建一个md5对象
        md5 = hashlib.md5()
        # 使用utf-8编码数据
        md5.update(data.encode('utf-8'))
        # 返回加密后的十六进制字符串
        return md5.hexdigest()


    # 判断目录是否存在
    @staticmethod
    def is_dir_exists(path):
        return os.path.exists(path) and os.path.isdir(path)


    # 获取文件所在完整路径
    @staticmethod
    def get_parent_path(file_path):
        index = file_path.rfind("/")
        if index == -1:
            index = file_path.rfind("\\")
        folder_path = file_path[:index + 1]
        print(folder_path)
        return folder_path
<<<<<<< HEAD:MuseTalk/scripts/utils/common.py

    # 上传文件的函数
    @staticmethod
    def upload_file(file_path, chunk_size=5 * 1024 * 1024):
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_url = ''
        chunk_number = 0
        with open(file_path, 'rb') as file:
            total = math.ceil(file_size / chunk_size)
            task_id = ''
            print(f'总片数：{total}')
            for offset in range(0, total):
                chunk = file.read(chunk_size)
                headers = {'access-token': config.ACCESS_TOKEN}
                response = requests.post(config.BASE_URL + "upload-api/chunkUpload", files={'file': chunk},
                                         params={'domainId': 1, 'dir': "video", 'chunk': chunk_number,
                                                 'chunkTotal': total,
                                                 'taskId': task_id, "fileName": file_name},
                                         headers=headers)
                if response.status_code == 200:
                    resp_obj = json.loads(response.text)
                    print('分片上传成功', response.json())
                    if "00000" == resp_obj['errorCode']:
                        data = resp_obj['data']
                        if "300" == data['status']:
                            task_id = data['taskId']
                            chunk_number = chunk_number + 1
                        else:
                            file_url = data['url']
                    else:
                        raise Exception(f"文件上传失败，{resp_obj['message']}")
                else:
                    print("分片上传失败", response.status_code)
                    raise Exception(f"文件上传失败，{response.status_code}")
            return file_url

    @staticmethod
    def download_file(url, local_path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_path

    @staticmethod
    def get_file_name(url):
        file_name = os.path.basename(url)
        print(file_name)
        if '?' in file_name:
            file_name = file_name[:file_name.index('?')]
        return file_name
=======
>>>>>>> parent of ba45930... tts:utils/common.py
