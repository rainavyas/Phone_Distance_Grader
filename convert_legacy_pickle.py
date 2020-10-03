'''
convert a pickle file saved from python2 into a pickle file that can be
read from python3
'''

import pickle
import dill

def convert(input_file, output_file):

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    with open(input_file, 'rb') as f:
        pkl = pickle.load(f, encoding="latin1")

    pickle.dump(pkl, open(output_file, 'wb'))


#input_file = '/home/alta/BLTSpeaking/grd-kk492/mfcc13/GKTS4-D3/grader/BLXXXgrd02/data/BLXXXgrd02.pkl'
input_file = '/home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/merger/adversarial/gradient_attck_mfcc_model/mypkl.pkl'
output_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'

'''
input_file = '/home/alta/BLTSpeaking/grd-kk492/mfcc13/GKTS4-D3/grader/BLXXXeval3/data/BLXXXeval3.pkl'
output_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXeval3/BLXXXeval3.pkl'
'''

convert(input_file, output_file)
