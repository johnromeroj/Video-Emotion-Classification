def get_aud_dirs(): 
    ''' 
    Function to get directories, ensures all are the same for each notebook.
    '''
    import os
    
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'Data', 'Audio')
    raw_dir = os.path.join(data_dir, 'Raw')
    
    features_dir = os.path.join(data_dir, 'Features')
    if not os.path.isdir(features_dir): os.mkdir(features_dir)
    
    metadata_dir = os.path.join(data_dir, 'Metadata')
    if not os.path.isdir(metadata_dir): os.mkdir(metadata_dir)
    
    eda_dir = os.path.join(data_dir, 'EDA')  
    if not os.path.isdir(eda_dir): os.mkdir(eda_dir)
    
    fig_dir = os.path.join(cwd, 'Figures')  
    if not os.path.isdir(fig_dir): os.mkdir(fig_dir)
    
    models_dir = os.path.join(cwd, 'Models')
    if not os.path.isdir(models_dir): os.mkdir(models_dir)

    return cwd, data_dir, raw_dir, features_dir, metadata_dir, eda_dir, fig_dir, models_dir

def get_aud_feats(data, sr=48000, frame_len_ms=25, hop_len_ms=10): 
    '''
    Function to return a tuple of feature arrays from the audio data.
    '''
    import librosa 
    import numpy as np 
    import scipy.signal as signal
    # Calculate Frame length and overlap (ms to samps)
    fr_len = int(sr / 1000 * frame_len_ms)
    hop_len = int(sr / 1000 * hop_len_ms)
    
    # MFCC
    y_emph = librosa.effects.preemphasis(data, coef=0.97)
    mfcc = librosa.feature.mfcc(y=y_emph, sr=sr, 
                                n_mfcc=20, 
                                hop_length = hop_len, 
                                win_length = fr_len, 
                                window = signal.windows.hamming, center=True, 
                                n_mels=128)   
    
    # CHROMA 
    chroma = librosa.feature.chroma_stft(y=data, sr=sr, 
                                         hop_length=hop_len, n_chroma=12, 
                                         n_fft=fr_len, win_length=fr_len)

    return mfcc, chroma

def augment_aud_data(y, sr): 
    '''
    Data to return augmented data from raw audio data.
    Pitch shift changes the pitch +/- 2 pitch levels. 
    Time Stretch makes the audio up to 20% faster or slower. 
    Both does both. 
    '''
    import numpy as np
    import librosa 
    augments = np.random.choice(['freq', 'time', 'both'])
    if augments == 'freq': 
        y_aug = librosa.effects.pitch_shift(y=y, sr=sr, scale=True, n_steps= np.random.choice([1,2]) * np.random.choice([-1,1]))
    if augments == 'time':  
        y_aug = librosa.effects.time_stretch(y=y, rate = 1 + (round(np.random.uniform(0.01, 0.20), 2) * np.random.choice([-1,1])))
    elif augments == 'both': 
        y_aug = librosa.effects.pitch_shift(y=y, sr=sr, scale=True, n_steps= np.random.choice([1,2]) * np.random.choice([-1,1]))
        y_aug = librosa.effects.time_stretch(y=y_aug, rate = 1 + (round(np.random.uniform(0.01, 0.20), 2) * np.random.choice([-1,1])))
    y_aug = librosa.utils.normalize(y_aug)
    return y_aug

def load_aud_data(fn, mode, data, metadata, augment = False, n_feats=12, sr=48000, frame_len_ms=25, hop_len_ms=10):
    '''
    Function to load the audio data, raw or features. Can read the data from or write it to pickle files. 
    '''
    import librosa
    import os 
    import numpy as np
    import scipy.signal as signal
    
    cwd, data_dir, raw_dir, features_dir, metadata_dir, eda_dir, fig_dir, models_dir = get_aud_dirs()
    
    fr_len = int(sr / 1000 * frame_len_ms)
    hop_len = int(sr / 1000 * hop_len_ms)
    
    target = metadata[metadata['fname'] == os.path.basename(fn)].category.values[0]

    if data == 'raw': 
        # Load the Audio Signal 
        y, sr = librosa.load(fn, sr=sr, mono=True) 
        if augment:
            aug1 = aumgent(y)
            aug2 = augment(y)
            return [(target, dat) for dat in [y, aug1, aug2]]
        elif not augment: 
            y_re = librosa.resample(y =y, orig_sr = 48000, target_sr = 16000)
            y_trimmed, _ = librosa.effects.trim(y_re, frame_length=fr_len, hop_length=hop_len, aggregate=np.max)
            return target, y_trimmed 
        
    elif data == 'feats': 
        
        out_fn      = os.path.join(features_dir, os.path.basename(fn) + 'feats.dat')
        out_fn_aug1 = os.path.join(features_dir, os.path.basename(fn) + 'feats_aug1.dat')
        out_fn_aug2 = os.path.join(features_dir, os.path.basename(fn) + 'feats_aug2.dat')
        
        if mode == 'w':
            # Load the Audio Signal 
            y, sr = librosa.load(fn, sr=sr, mono=True) 
            
            feats = np.concatenate(get_aud_feats(data=y))
            if augment: 
                aug1, aug2 = augment_aud_data(y, sr), augment_aud_data(y, sr)
                feats_aug1, feats_aug2 = np.concatenate(get_aud_feats(data=aug1)), np.concatenate(get_aud_feats(data=aug2)) 
                feats.dump(out_fn)
                feats_aug1.dump(out_fn_aug1)
                feats_aug2.dump(out_fn_aug2)
                return [(target, f.T) for f in [feats, feats_aug1, feats_aug2]]
            
            elif not augment:          
                feats.dump(out_fn)
                return target, feats.T 
            
        elif mode =='r': 
            feats = np.load(out_fn, allow_pickle=True)
            if augment: 
                feats_aug1 =  np.load(out_fn_aug1, allow_pickle=True)
                feats_aug2 =  np.load(out_fn_aug2, allow_pickle=True)
                return [(target, f.T) for f in [feats, feats_aug1, feats_aug2]]
            elif not augment: 
                return target, feats.T

def get_aud_data(df, metadata, mode='r', data='feats', augment=False): 
    '''
    Function to get the desired data from a dataframe of metadata. 
    Useful to load sets after train_test_split.
    '''
    
    import os 
    
    cwd, data_dir, raw_dir, features_dir, metadata_dir, eda_dir, fig_dir, models_dir = get_aud_dirs()
    
    fnames = [os.path.join(raw_dir, f"Actor_{df['actor'][i]}", df['fname'][i]) if df['actor'][i] >= 10
              else os.path.join(raw_dir, f"Actor_0{df['actor'][i]}", df['fname'][i]) 
              for i in df.index]
 
    data_tups = [load_aud_data(fn, mode=mode, data=data, metadata=metadata, augment=augment) for fn in fnames]
    if augment: 
        data_tups = [item for sublist in data_tups for item in sublist]    
    
    target, data = zip(*data_tups)
    return target, data

def aud_data_from_vis_fn(fn:str, aud_metadata):

    '''
    Function to get the audio data from a video filename, helps to match audio data to video data. 
    '''

    import os
    from glob import glob

    id = fn[fn.find('-') + 1 : fn.find('.')]

    aud_fname = '03-' + id + '.wav'

    data_dir = os.path.join(os.getcwd(), 'Data')
    
    full_aud_fname = glob(os.path.join(data_dir, 'Audio', 'Raw', '*', aud_fname))[0]
    
    tar, aud_feats = load_aud_data(full_aud_fname, mode='r', data='feats', metadata=aud_metadata)

    return aud_feats  