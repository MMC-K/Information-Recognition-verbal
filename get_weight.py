import os
import gdown

path_weight = './model/T5-mean-m_KETI-AIR_ke-t5-base_default/weights/'

if not os.path.isdir(path_weight):
   os.makedirs(path_weight)
   

url = 'https://drive.google.com/file/d/13madUWxOHblPwXtu7aNeu8TtLckAUOiG/view?usp=sharing'
output = f'{path_weight}best.pth'
gdown.download(url, output, quiet=False, fuzzy=True)