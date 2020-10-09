'''
Using the 13-dim mfcc vectors in the pickle file, calculate the pdf of each
phone for each speaker and then form the 1128 dim phone distance feature
vector for each speaker.
'''

import numpy as np
import pickle

class Pkl2Feat_worker():
    def __init__(self, pkl, output_file_name):
        self.pkl = pkl
        self.filename = output_file_name

    def get_phones(self, alphabet='arpabet'):
        if alphabet == 'arpabet':
            vowels = ['aa', 'ae', 'eh', 'ah', 'ea', 'ao', 'ia', 'ey', 'aw', 'ay', 'ax', 'er', 'ih', 'iy', 'uh', 'oh', 'oy', 'ow', 'ua', 'uw']
            consonants = ['el', 'ch', 'en', 'ng', 'sh', 'th', 'zh', 'w', 'dh', 'hh', 'jh', 'em', 'b', 'd', 'g', 'f', 'h', 'k', 'm', 'l', 'n', 'p', 's', 'r', 't', 'v', 'y', 'z'] + ['sil']
            phones = vowels + consonants
            return phones
        if alphabet == 'graphemic':
            vowels = ['a', 'e', 'i', 'o', 'u']
            consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'] + ['sil']
            phones = vowels + consonants
            return phones
        raise ValueError('Alphabet name not recognised: ' + alphabet)

    def kl_div(self, mu1, mu2, sig1, sig2):
        '''
        Computes symmetric KL divergence (Jensen-Shannon divergence)
         between Gaussian pdfs
        '''
        if len(mu1.shape) != 2 or mu1.shape[1] != 1:
            raise ValueError('Mu1 should have shape [X,1] but instead has shape ' + str(mu1.shape))
        d = mu1.shape[0]
        if mu2.shape != mu1.shape:
            raise ValueError('Mu2 should have shape ' + str(mu1.shape) + ' but instead has shape ' + str(mu2.shape))
        for j in range(2):
            sig_mat = [sig1, sig2][j]
            if sig_mat.shape != (d, d):
                raise ValueError(
                    'Sig'+str(j) + ' should have shape (' + str(d) + ',' + str(d) + ') but instead has shape ' + str(
                        sig_mat.shape))
            min_eig = np.min(np.real(np.linalg.eigvals(sig_mat)))
            if min_eig < 0:
                sig_mat += 1e-3 * np.eye(d)
        if np.linalg.matrix_rank(sig2) != sig2.shape[0] or np.linalg.matrix_rank(sig1) != sig1.shape[0]:
            return -1.0
        isig2 = np.linalg.inv(sig2)
        md = mu2 - mu1
        trace = np.trace(np.matmul(isig2, sig1))
        msm = np.matmul(np.matmul(np.transpose(md), isig2), md)[0][0]
        log_det_ratio = np.log(np.linalg.det(sig2)) - np.log(np.linalg.det(sig1))
        kld = 0.5 * (trace + msm + log_det_ratio - d)
        return -1.0 if (np.isnan(kld) or np.isinf(kld)) else kld

    def get_pdf(self, phones):
        obj = self.pkl
        n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector

        # Store all the means: num_spkrs x 48 x 13 x 1
        means = np.zeros((len(obj['plp']), len(phones)-1 , n, 1))

        # Store variances: num_spkrs x 48 x (13 x 13) -> ie 13x13 covariance matrix for each of the 48 phones for each speaker
        variances = np.zeros((len(obj['plp']), len(phones)-1, n, n))

        for spk in range(len(obj['plp'])):
            print("on speaker", spk)
            SX = np.zeros((len(phones) - 1, n, 1))
            N = np.zeros(len(phones) - 1)
            SX2 = np.zeros((len(phones) - 1, n, n))
            Sig = np.zeros((len(phones) - 1, n, n))

            for utt in range(len(obj['plp'][spk])):
                # Iterate through utterances
                for w in range(len(obj['plp'][spk][utt])):
                    # Iterate through words
                    for ph in range(len(obj['plp'][spk][utt][w])):
                        # Iterate through phones
                        for frame in range(len(obj['plp'][spk][utt][w][ph])):
                            # Iterate through frames
                            N[obj['phone'][spk][utt][w][ph]] += 1
                            X = np.reshape(np.array(obj['plp'][spk][utt][w][ph][frame]), [n, 1])
                            SX[obj['phone'][spk][utt][w][ph]] += X
                            SX2[obj['phone'][spk][utt][w][ph]] += np.matmul(X, np.transpose(X))

            for ph in range(len(phones)-1):
                if N[ph] !=0:
                    SX[ph] /= N[ph]
                    SX2[ph] /= N[ph]
                    m2 = np.matmul(SX[ph], np.transpose(SX[ph]))
                    Sig[ph] = SX2[ph] - m2

            k = 0
            for i in range(len(phones) - 1):
                for j in range(i + 1, len(phones) - 1):
                    obj['pdf'][spk][k] = -1 if N[i] == 0 or N[j] == 0 else self.kl_div(SX[i], SX[j], Sig[i], Sig[j])
                    k += 1


    def write_pkl_object(self):
        '''
        Writes the entire pickle object with all the data (including feature
        vectors for each phone per speaker) to a pickle file.
        '''
        pickle.dump(self.pkl, open(self.filename, "wb"))

    def work(self):
        '''
        Returns a num_speakers x 1128 dimension (48*47*0.5)
        numpy array, where each speaker has 1128 phone
        distance features.
        Returns a second list of speakerids.
        '''

        # Get the phones
        phones = self.get_phones()

        # Estimate the Gaussian pdfs for each speaker and each phone
        # Then calculate the phone distance features for each speaker
        self.get_pdf(phones)

        # Write the pickle object to a pickle file
        self.write_pkl_object()

# Load pickle file containing all the data
#pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXgrd02/BLXXXgrd02.pkl'
pkl_file = '/home/alta/BLTSpeaking/exp-vr313/data/mfcc13/GKTS4-D3/grader/BLXXXeval3/BLXXXeval3.pkl'
pkl = pickle.load(open(pkl_file, "rb"))

#output_file_name = 'BLXXXgrd02.pkl'
output_file_name = 'BLXXXeval3.pkl'

my_worker = Pkl2Feat_worker(pkl, output_file_name)
my_worker.work()
