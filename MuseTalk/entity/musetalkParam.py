from typing import Dict, Any

from pydantic import BaseModel


class VideoCreateParam(BaseModel):
    wavUrl: str
    videoUrl: str
    dataId: str
    callBackUrl: str


class TaskConfig:
    def __init__(self, video_path: str, audio_path: str, bbox_shift: int = 0):
        self.video_path = video_path
        self.audio_path = audio_path
        self.bbox_shift = bbox_shift


class InferenceConfig:
    def __init__(self, tasks: Dict[str, TaskConfig]):
        self.tasks = tasks


class MusetalkCreateParam():
    result_dir: str = './results'
    fps: int = 25
    dataId: str
    batch_size: int = 8
    output_vid_name: str = None
    use_saved_coord: bool = False
    use_float16: bool = False
    inference_config: Dict[str, TaskConfig] = None
    callBackUrl: str


class MuseTalkBaseParam(BaseModel):
    result_dir: str = './results'
    fps: int = 25
    batch_size: int = 8
    use_saved_coord: bool = False
    use_float16: bool = True
    is_template: bool = False


class MuseTalkTemplateParam(MuseTalkBaseParam):
    id: str
    type: int
    bboxShift: int
    sourceUrl: str
    audioUrl: str
    humanEventType: str = None
    is_template: bool = True
    back_img: Any = None

    # def __init__(self, id, type, bboxShift, sourceUrl, audioUrl,humanEventType):
    #     self.id = id
    #     self.type = type
    #     self.bboxShift = bboxShift
    #     self.sourceUrl = sourceUrl
    #     self.audioUrl = audioUrl
    #     self.humanEventType = humanEventType

    # @staticmethod
    # def from_dict(self, json_obj):
    #     return MuseTalkTemplateParam(id=json_obj['id'], type=json_obj['type'], bboxShift=json_obj['bboxShift'],
    #                                  sourceUrl=json_obj['sourceUrl'], audioUrl=json_obj['audioUrl'])


class MuseTalkParam(MuseTalkBaseParam):
    id: str
    audioUrl: str
    backgroundUrl: Any = None
    # def __init__(self, id, audioUrl):
    #     self.id = id
    #     self.audioUrl = audioUrl

    # @staticmethod
    # def from_dict(self, json_obj):
    #     return MuseTalkParam(id=json_obj['id'], audioUrl=json_obj['audioUrl'])
