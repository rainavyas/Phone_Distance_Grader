# Phone_Distance_Grader
Use phone distances to assess a speakers pronunciation for fluency

Kyriakopoulos, K., Gales, M., Knill, K. (2017) Automatic Characterisation of the Pronunciation of Non-native English Speakers using Phone Distance Features. Proc. 7th ISCA Workshop on Speech and Language Technology in Education, 59-64, DOI: 10.21437/SLaTE.2017-11.





Model Summary:

When attempting to grade a speaker on their English fluency by only considering their pronunciation, phone distances can be used. An ASR system will output a phone classification for each frame of a candidate's speech, using the mfcc vector input. For each speaker and for every phone (47 in English language) a multivariate Gaussian distribution can be estimated (by calculating a mean vector and covariance matrix from the mfcc vectors assigned to the particular phone for each speaker). To reduce the impact of voice quality or accent, the "distance" between these phones' distributions can be computed. A symmetric KL divergence distance is employed, meaning, we can summarise each speaker's phone distances in a 1128 dimensional feature vector (48 x 47 x 0.5). This vector can be passed through a fully connected deep model to predict a fluency grade (for example on a CEFR scale).
