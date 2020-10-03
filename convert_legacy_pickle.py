'''
convert a pickle file saved from python2 into a pickle file that can be
read from python3
'''

import pickle

def convert(input_file, output_file):
    with open(input_file, 'rb') as f:
        pkl = pickle.load(f, encoding="latin1")

    pickle.dump(pkl, open(output_file, 'wb'))


input_file = '/home/alta/BLTSpeaking/grd-kk492/mfcc13/GKTS4-D3/grader/BLXXXgrd02/data/BLXXXgrd02.pkl'
output_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'

'''
input_file = '/home/alta/BLTSpeaking/grd-kk492/mfcc13/GKTS4-D3/grader/BLXXXeval3/data/BLXXXeval3.pkl'
output_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXeval3/BLXXXeval3.pkl'
'''

convert(input_file, output_file)
