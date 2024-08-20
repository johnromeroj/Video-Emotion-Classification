def get_vis_dirs(): 

    '''
    Function to get the directories for 
    '''
    
    vis_data_dir = os.path.join(os.getcwd(), 'Data', 'Video')
    full_vis_data = os.path.join(vis_data_dir, 'Full_Data')
    vis_raw_dir = os.path.join(vis_data_dir, 'Raw')
    
    vis_proc_dir = os.path.join(vis_data_dir, 'Processed')
    if not os.path.isdir(vis_proc_dir): os.mkdir(vis_proc_dir)
        
    lmrk_dir = os.path.join(vis_data_dir, 'Landmarks')
    if not os.path.isdir(lmrk_dir): os.mkdir(lmrk_dir) 
        
    fig_dir = os.path.join(os.getcwd(), 'Figures')
    if not os.path.isdir(fig_dir): os.mkdir(fig_dir)
        
    model_dir = os.path.join(os.getcwd(), 'Models')
    if not os.path.isdir(model_dir): os.mkdir(model_dir)

    return vis_data_dir, full_vis_data, vis_raw_dir, vis_proc_dir, lmrk_dir, fig_dir, model_dir

def get_metadata(mode: str, type:str): 
    
    '''
    Function to load the Metadata for both audio and video data. 
    '''
    
    import os 
    import pandas as pd 
    from glob import glob 
    from sklearn.model_selection import train_test_split
    
    meta_dir = os.path.join(os.getcwd(), 'Data', type.title(), 'Metadata')

    if mode == 'w': 

        raw_dir = os.path.join(os.getcwd(), 'Data', type.title(), 'Raw')

        if type.lower() == 'visual':
            files = glob(os.path.join(raw_dir, '*', '*.mp4'))
        elif type.lower() == 'audio': 
            files = glob(os.path.join(raw_dir, '*', '*.wav'))

        meta_ls = [[os.path.basename(fn)] + list(map(int, os.path.basename(fn).split('.')[0].split('-')))[2:] for fn in files]

        meta_cols = ['fname', 'emotion', 'intensity', 'statement', 'rep', 'actor']
        metadata = pd.DataFrame(data=meta_ls, columns=meta_cols)

        label_dict = {1:'baseline', 2:'baseline', 3:'happy', 4:'sad', 
                      5:'angry',6:'fearful', 7:'disgust',8:'surprised'}

        metadata['category'] = metadata['emotion'].apply(lambda r: label_dict[r])
        
        metadata['sex'] = metadata['actor'].apply(lambda r: 'm' if r % 2 == 1 else 'f')
        metadata['strat'] = metadata[['emotion', 'sex']].apply(lambda r: '_'.join(r.values.astype(str)), axis=1)

        train, test = train_test_split(metadata, stratify = metadata['strat'],
                               test_size=0.15, random_state=505)
        
        train, val = train_test_split(train, stratify = train['strat'],
                                      test_size=0.15/0.85, random_state=808)
        partition = []
        for fn in metadata.fname.to_list(): 
            if fn in test.fname.tolist(): partition.append('test')
            elif fn in val.fname.tolist(): partition.append('val')
            else: partition.append('train')
                
        metadata['partition'] = partition
        
        metadata.to_csv(os.path.join(meta_dir, f'{type.title()}_metadata.csv'), header=True, index=False)
    
    elif mode == 'r': 
        metadata = pd.read_csv(os.path.join(meta_dir, f'{type.title()}_metadata.csv'))
    
    return metadata

def frames_from_vid(path, stretch, target_fps=None, n_frames=None):

    '''
    Function to extract the frames from a video.
    Can be done at a target FPS or a target number of frames.
    Also, can change the number of pixels by any factor. 
    '''

    import cv2
    import numpy as np
    
    frames = []
    cap = cv2.VideoCapture(path)
    while True: 
        ret, frame = cap.read()
        # Check if an image was read
        if ret: 
            frame = cv2.resize(frame, (int(frame.shape[1] * stretch), int(frame.shape[0] * stretch)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)   
        else: 
            break
    orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    if n_frames is not None:
        if n_frames == 1: 
            frames = frames[len(frames) // 2]
        else: 
            frame_list = np.linspace(0, len(frames)-1, n_frames, endpoint=True).astype(int).tolist()
            frames = [frames[i] for i in frame_list]
    elif target_fps is not None:
        frame_list = list(range(0, orig_fps, int(orig_fps / target_fps)))
        frames = [frames[i] for i in frame_list]
        
    return frames
    
def get_frame_dataset(name: str, stretch, mode:str, target_fps = None, n_frames=None): 

    '''
    Function to extract a TF dataset from videos, basically a complicated wrapper for the frame extraction function. 
    '''
    
    import os
    from glob import glob 
    import tensorflow as tf
    import numpy as np

    if n_frames == 1: 
        image_ds_dir = os.path.join(os.getcwd(), 'Data', 'Video', 'Processed', 'Image')
        if not os.path.isdir(image_ds_dir): os.mkdir(image_ds_dir)
        ds_path = os.path.join(image_ds_dir, name)
    elif n_frames > 1 or n_frames == None: 
        video_ds_dir = os.path.join(os.getcwd(), 'Data', 'Video', 'Processed', 'Video')
        if not os.path.isdir(video_ds_dir): os.mkdir(video_ds_dir)
        ds_path = os.path.join(video_ds_dir, name)
    
    if not os.path.isdir(ds_path): os.mkdir(ds_path)

    if mode == 'w':
        print(f'Extracting {name} Dataset') 
        
        metadata = get_metadata('r', type = 'Video')
        cat_lookup = tf.keras.layers.StringLookup(num_oov_indices=0, 
                                                  vocabulary = np.unique(metadata.category).tolist(), 
                                                  output_mode='one_hot')
        
        train_meta = metadata.loc[metadata.partition == name].reset_index()
        
        data, labels = [], []
        
        for idx, row in train_meta.iterrows(): 
            
            path = glob(os.path.join(os.getcwd(), 'Data', 'Video', 'Raw', '*', row.fname))[0]
            
            frames = frames_from_vid(path=path,  n_frames=n_frames, stretch=stretch)
            frames = [ frame / 255 for frame in frames]
            frames = tf.stack(frames, axis=0)
            data.append(frames)
            
            label = cat_lookup(row.category)
            labels.append(label)

            if idx == 299: 
                print('Saving Initial DS')
                data, labels = tf.stack(data), tf.stack(labels)
                dataset = tf.data.Dataset.from_tensor_slices((data, labels))
                dataset.save(ds_path)
                data, labels = [], []
            elif idx > 300 and (idx + 1) % 300 == 0: 
                print('Concatenating to DS')
                data, labels = tf.stack(data), tf.stack(labels)
                subdataset = tf.data.Dataset.from_tensor_slices((data, labels))
                full_dataset = tf.data.Dataset.load(ds_path)
                dataset = full_dataset.concatenate(subdataset)
                dataset.save(ds_path)
                data, labels = [], []

        print('Saving Full DS')
        data, labels = tf.stack(data), tf.stack(labels)
        subdataset = tf.data.Dataset.from_tensor_slices((data, labels))
        full_dataset = tf.data.Dataset.load(ds_path)
        dataset = full_dataset.concatenate(subdataset)

        dataset.save(ds_path)

    elif mode == 'r': 
        
        print(f'Loading {name} Dataset')
        
        if n_frames == 1: 
            ds_path = os.path.join(os.getcwd(), 'Data', 'Video', 'Processed', 'Image', name)
        elif n_frames > 1: 
            ds_path = os.path.join(os.getcwd(), 'Data', 'Video', 'Processed', 'Video', name)

        dataset = tf.data.Dataset.load(ds_path)
    
    return dataset

def inv_label(lab): 

    '''
    Function to invert the labels from one-hot encoded to the category. 
    Mostly used to make sure it's always done in the same way. 
    '''
    
    import tensorflow as tf
    import numpy as np

    metadata = get_metadata(mode='r', type = 'Video')
    
    inv_cat_lookup = tf.keras.layers.StringLookup(num_oov_indices=0, 
                                                  vocabulary = np.unique(metadata.category).tolist(), 
                                                  invert = True)
    emotion = inv_cat_lookup(np.argmax(lab))
    
    return bytes.decode(emotion.numpy())
    
def display_video(dataset, rows=4, columns=3): 

    '''
    Function to display the frames of a random video. In the default case, expects 10 frames. 
    '''
    
    import matplotlib.pyplot as plt
    import os 

    data, lab = list(dataset.take(1))[0]

    emotion = inv_label(lab)
    
    vid = data.numpy()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.title(f'Training Example: {emotion.title()}')
    for idx, frame in enumerate(list(vid)): 
        fig.add_subplot(rows, columns, idx+1)
        plt.imshow(frame)
        plt.axis('off')
    ax.axis('off')
    plt.tight_layout()
    
    plt_fn = os.path.join(os.getcwd(), 'Figures', f'{emotion}_example.png') 
    
    fig.savefig(plt_fn, dpi=1200)
    plt.show()

def extract_landmarks(actor: int, mode: str):

    '''
    Function to extract the facial landmarks from the video frames. Takes about 24 sec per file. 
    '''

    import dlib 
    import cv2
    import numpy as np
    import pandas as pd 
    import os 
    from glob import glob

    lmrk_dir = os.path.join(os.getcwd(), 'Data', 'Landmarks')
        
    if actor < 10: 
        actPath = os.path.join(os.getcwd(), 'Data', 'Raw', f'Actor_0{actor}')
        landpath = os.path.join(lmrk_dir, f'Actor_0{actor}_landmarks.csv')
    else: 
        actPath = os.path.join(os.getcwd(), 'Data', 'Raw', f'Actor_{actor}')
        landpath = os.path.join(lmrk_dir, f'Actor_{actor}_landmarks.csv')

    if mode != 'o' and os.path.isfile(landpath): 
        return

    files = glob(os.path.join(actPath, '02*.mp4'))

    pt_list = ['pt' + str(i+1) for i in range(68)]
    x_names = [pt + '.x' for pt in pt_list]
    y_names = [pt + '.y' for pt in pt_list]
        
    landmarks = pd.DataFrame(columns = ['fname', 'frame_ct'] +  x_names + y_names)
    
    for idx, path in enumerate(files): 

        basename = os.path.basename(path)
        cap = cv2.VideoCapture(path)
        
        predictor = dlib.shape_predictor(os.path.join('models', 'shape_predictor_68_face_landmarks.dat'))
        detector = dlib.get_frontal_face_detector()
        
        frame_ct = 0

        while True: 
            ret, frame = cap.read()
            if ret:  
                try: 
                    graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rect = detector(graysc, 1)[0]
                    
                    shape = predictor(graysc, rect)
                    x_coord_dict = { f'pt{i+1}.x': shape.part(i).x for i in range(68) }
                    y_coord_dict = { f'pt{i+1}.y': shape.part(i).y for i in range(68) }  
                    coords = x_coord_dict | y_coord_dict
    
                    coords['fname'] = basename
                    coords['frame_ct'] = frame_ct
                    
                    landmarks = pd.concat( [landmarks, pd.DataFrame(coords, index=[idx])])
                    frame_ct += 1
                except: 
                    print(path)
                    print(frame_ct)
                    continue
            else: 
                break
        cap.release()

    landmarks.to_csv(landpath)
    
    return 

def plot_landmarks(path):

    '''
    Function to plot the landmarks onto a video frame and save it to file in the figure directory. 
    '''

    import dlib 
    import cv2
    import numpy as np
    import os

    frames = []
    cap = cv2.VideoCapture(path)
    fig_dir = os.path.join(os.getcwd(), 'Figures')

    predictor = dlib.shape_predictor(os.path.join('models', 'shape_predictor_68_face_landmarks.dat'))
    detector = dlib.get_frontal_face_detector()
    frame_n = np.random.randint(20, 30)
    counter = 0
    while True: 
        counter += 1
        ret, frame = cap.read()

        if ret: 
            if counter == 20:
    
                graysc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rect = detector(graysc, 1)[0]
                
                shape = predictor(graysc, rect)
                coords = [(shape.part(i).x, shape.part(i).y) for i in range(68) ]
                
                x, y = rect.left(), rect.top()
                w, h = rect.right() - x, rect.bottom() - y
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                for (x, y) in coords:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                cv2.imwrite(os.path.join(fig_dir, 'example_landmarks.png'), frame)  
                cv2.imshow('Example Landmarks', frame)
                break
        else: 
            break
    cap.release()
    
def load_landmarks(): 

    '''
    Function to load the landmark data for all files. 
    Filenames are included in the resulting data frame for identification.
    Frame numbers are also included to index the frames.
    '''
    
    from glob import glob 
    import os
    import pandas as pd 
    
    lmrk_dir = os.path.join(os.getcwd(), 'Data', 'Video', 'Landmarks', '*.csv')
    
    files = glob(lmrk_dir)

    landmark_data = pd.read_csv(files[0])

    for f in files[1:]: 
        landmark_data = pd.concat([landmark_data, pd.read_csv(f)])

    return landmark_data

def get_landmark_dataset(name: str): 

    '''
    Function to get the Landmark dataset that corresponds to the certain partition. 
    i.e. train, val, test
    '''

    from glob import glob 
    import os 
    import pandas as pd 
    import numpy as np 
    import tensorflow as tf

    lmrk_dir = os.path.join(os.getcwd(), 'Data', 'Video', 'Landmarks')
    
    files = glob(os.path.join(lmrk_dir, '*.csv'))
    
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    metadata = get_metadata('r', 'Video')
    
    fnames = metadata[metadata['partition'] == name].fname.to_list()
    df = data[data['fname'].isin(fnames)]

    cat_lookup = tf.keras.layers.StringLookup(num_oov_indices=0, 
                                                  vocabulary = np.unique(metadata.category).tolist(), 
                                                  output_mode='one_hot')
    arrs_ls = []
    
    for fn, group in df.groupby('fname'): 
        
        group.set_index('frame_ct', inplace=True)
        
        dat = np.array(group[[col for col in group.columns.to_list() if 'pt' in col]].values)
    
        label_dict = {1:'baseline', 2:'baseline', 3:'happy', 4:'sad', 
                      5:'angry',6:'fearful', 7:'disgust',8:'surprised'} 

        emo = label_dict[int(fn.split('-')[2])]
        lab = cat_lookup(emo)
        arrs_ls.append((fn, lab, dat))

    return arrs_ls

def normalize_landmark_ds(arrs: list): 

    '''
    Function to normalize the landmarks for each file. 
    '''
    
    from sklearn.preprocessing import StandardScaler 

    norm_arrs = []

    for arr in arrs: 
        
        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(arr)
        norm_arrs.append(scaled_arr)

    return norm_arrs
    
def plot_history(history, loss_fn, acc_fn, loss_title, acc_title): 

    '''
    Function to plot the history from fitting a TF model. 
    Also saves the figures to files.
    '''

    import matplotlib.pyplot as plt

    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], color='black')
    plt.plot(history.history['val_loss'], color='blue')
    plt.title(loss_title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Val Loss'], loc='lower left')
    plt.savefig(loss_fn)
    plt.show()
    
    # Plot training Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], color='black')
    plt.plot(history.history['val_accuracy'], color='blue')
    plt.title(acc_title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Val Accuacy'], loc='upper left')
    plt.savefig(acc_fn)
    plt.show()  









    
    