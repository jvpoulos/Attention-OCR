__author__ = 'moonkey'

import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from data_util.bucketdata import BucketData

class DataGen(object):
    GO = 1
    EOS = 2

    def __init__(self,
            data_root, annotation_fn,
            evaluate = False,
            valid_target_len = float('inf'),
           img_width_range = (83, 2083), # iamdb train set
            word_len = 81):
            # img_width_range = (354,1990), # sgdb
            # word_len = 74): 
            # img_width_range = (306,911), # pardb
            # word_len = 70): 
           # img_width_range = (135,2358), # rimes
           # word_len = 100): 
            # img_width_range = (175,1801), # gwdb
            # word_len = 87): 
       
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """

        img_height = 64
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)

        if evaluate:
            self.bucket_specs = [(int(math.ceil(img_width_range[0])), int(math.ceil(img_width_range[1] / 8))),
                             (int(math.ceil(img_width_range[1]/8 )), int(math.ceil(img_width_range[1] / 6))),
                             (int(math.ceil(img_width_range[1]/6 )), int(math.ceil(img_width_range[1] / 4))), 
                             (int(math.ceil(img_width_range[1] / 4)), int(math.ceil(img_width_range[1] / 3))), 
                            (int(math.ceil(img_width_range[1] / 3)), int(math.ceil(img_width_range[1]/2)))] 
        else:
            self.bucket_specs = [(int(math.ceil(img_width_range[0])), int(math.ceil(img_width_range[1] / 8))),
                             (int(math.ceil(img_width_range[1]/8 )), int(math.ceil(img_width_range[1] / 6))),
                             (int(math.ceil(img_width_range[1]/6 )), int(math.ceil(img_width_range[1] / 4))), 
                             (int(math.ceil(img_width_range[1] / 4)), int(math.ceil(img_width_range[1] / 3))), 
                            (int(math.ceil(img_width_range[1] / 3)), int(math.ceil(img_width_range[1]/2)))] 
                            
        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def get_size(self):
        with open(self.annotation_path, 'r') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                img_path, lex = l.strip().split()
                try:
                    img_bw, word = self.read_data(img_path, lex)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-1]

                    # TODO:resize if > 320
                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            assert False, 'no valid bucket of width %d'%width
                except IOError:
                    pass # ignore error images
                    #with open('error_img.txt', 'a') as ef:
                    #    ef.write(img_path + '\n')
        self.clear()

    def read_data(self, img_path, lex):
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]


        # û: 251 # end RIMES
        # ....
        # À: 192
        # ' }':125
        # '|':124 # end IAMDB
        # 'a':97, 'z':122
        # '_':95
        # 'A':65, 'Z':90
        #"?":63
        # '<':60, '>':62
        #";":59
        #":":58
        # '0':48, '9':57
        #"/":47
        # '.':46
        #'-':45
        #',':44
        #"+":43
        #"*":42
        #"(":40, "(":41
        #"'":39
        #"&":38
        #"!":33 # start IAMDB/RIMES

        # 0: PADDING, 1: GO, 2: EOS, 3: UNK

        word = [self.GO]
        for c in lex:
            assert 32 < ord(c) < 126 or 191 < ord(c) < 252
            word.append(
                ord(c)+3-33 if ord(c) < 126 else ord(c)-192+96
            )
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word


def test_gen():
    print('testing gen_valid')
    # s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
    s_gen = EvalGen('../../data/evaluation_data/icdar13', 'test.txt')
    count = 0
    for batch in s_gen.gen(1):
        count += 1
        print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape[2:]))
        assert batch['data'].shape[2] == img_height
    print(count)


if __name__ == '__main__':
    test_gen()