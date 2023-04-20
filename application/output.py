import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
import imageio
from PIL import Image
# from application.seoptimization.caption_image_beam_search import caption_image_beam_search
# from application.seoptimization.cosine import Doct2vect
from application.seoptimization.cosine import Semantic_sim
# from seoptimization.cosine import Doct2vect
# from seoptimization.cosine import Semantic_sim

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")



def run_query(text):
    result = pd.read_csv(('output/vid_results.csv'))
    reference_string = text    
    corpus = [reference_string]+list(result['seq_text'])
    print(list(result['seq_text']))
    print(corpus)
    # capturing similarties between seq_text with reference_string
    # result['cosine_values'] = Doct2vect(corpus)[1:]  
    result['cosine_values'] = Semantic_sim(corpus)[1:]
    result['cosine_values']= result['cosine_values'].apply(lambda x: '%.2f' % float(x*100))
    result = result.values.tolist()
    # print(result)
    result.sort(key=lambda x: float(x[3]) , reverse = True)
    

    return result
 

if __name__ == '__main__':
    run_query("cooking video")