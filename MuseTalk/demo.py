import os

url = 'https://dev.ymygz.com/oss/sot-bucket/video/202411/2414101278E0F58C9A8DAD29208FEE85.mp4?'

file_name = os.path.basename(url)
print(file_name)
if '?' in file_name:
    file_name = file_name[:file_name.index('?')]
    print(file_name)
else:
    print(file_name)
