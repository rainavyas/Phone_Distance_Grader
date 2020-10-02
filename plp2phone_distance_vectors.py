import torch

class PLP_worker():
    def __init__(self, pkl):
        self.pkl = pkl
        self.features = []
        self.speakerids = []

    def get_all_speakers_features(self):
        '''
        Returns a num_speakers x 1081 dimension
        numpy array, where each speaker has 1081 phone
        distance features.
        Returns a second list of speakerids.
        '''

    def write_features_to_pkl(self, filename):
        '''
        Writes the phone distance features (as a list of list)
        and the speakerids to a pickle file.
        '''
