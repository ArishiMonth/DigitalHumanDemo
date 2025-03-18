import argparse


class Config():
    DB_HOST = "mysql"
    DB_PORT = 3306
    DB_USER = "root"
    DB_PASSWORD = "Xnh00<>?"
    DB_DBNAME = "db_standard_one_travel"
    DB_CHARSET = "utf8"
    # 启动时最小空闲连接数量
    DB_MIN_CACHED = 2
    # 允许的空闲最大连接数量
    DB_MAX_CACHED = 2
    # 共享连接允许的最大数量
    DB_MAX_SHARED = 5
    # 创建连接池的最大数量
    DB_MAX_CONNECYIONS = 20
    DB_BLOCKING = True
    DB_MAX_USAGE = None
    DB_SET_SESSION = []
    ACCESS_TOKEN = "accessToken@@@9rSXGKmbqQKRNZ8o4DO7fNi2wyvzuztQBAmLyjjvKrA1"
    BASE_URL = "https://dev.ymygz.com/sot/"
    port = 8080
    cors_allowMethods = "GET, POST, PUT, DELETE, OPTIONS, PATCH, FETCH"
    cors_allowHeaders = "Origin,Accept,Accept-Language,Authorization,Content-Type,UserCode,Timestamp,Nonce,Signature,Content-Disposition,right-token,access-token"
    cors_allowOrigins = "*"
    DATA_PATH = "data"


class DevConfig(Config):
    BASE_URL = "https://dev.ymygz.com/sot/"


class LocalConfig(Config):
    DB_HOST = "220.197.15.175"
    DB_PORT = 31616
<<<<<<< HEAD:GPT-SoVITS-beta/yj_config/env_config.py
    BASE_URL = "https://dev.ymygz.com/digital-admin-api"
=======
    BASE_URL = "https://dev.ymygz.com/sot/"
>>>>>>> parent of ba45930... tts:config/env_config.py
    port = 8090


class TestConfig(Config):
    BASE_URL = "http://test.ymygz.com/standard-onetravel-2/"


class ProdConfig(Config):
    BASE_URL = "https://test.ymygz.com/standard-onetravel-demo/"


mapping = {
    'local': LocalConfig,
    'dev': DevConfig,
    'test': TestConfig,
    'prod': ProdConfig
}
# APP_ENV = os.environ.get('APP_ENV', 'local').lower()
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='local')
parser.add_argument('--rhubarbPath', type=str, default='rhubarb-win')
opt = parser.parse_args()
config = mapping[opt.env]()
