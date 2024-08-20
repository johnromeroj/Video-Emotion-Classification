# Video Emotion Classification

Repository for the video emotion classification project.

Note that the raw audio and visual data are not included in this repository, though can be found at https://zenodo.org/record/1188976. This project uses data originally presented in: 

S. R. Livingstone and F. A. Russo, “The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)”, PLoS ONE, vol. 13, no. 5. Zenodo, p. e0196391, Apr. 05, 2018. doi: 10.5281/zenodo.1188976.

The full model is trained in the notebook john_romero_full_model.ipynb and depends on the training of the audio model (john_romero_audio_classification.ipynb) and the visual model (john_romero_landmark_classification.ipynb). Note that these notebooks rely on functions from the files john_romero_audio_functions.py and john_romero_video_functions.py. 

All models are saved in the model directory, which also houses the model used for facial landmark detection. 
