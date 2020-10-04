# heart.ai-ml

A machine learning model for classifying heartbeat audio data as normal, abnormal, or null.

`gcloud_training_test.py` can be used to train the machine learning model on a local computer or in a Jupyter Notebook on Google Cloud's Deep Learning VMs.

`classification_test.py` can be used to classify individual audio files with the trained machine learning model.

`weights_gcloud.h5` is a model trained on a Google Cloud Deep Learning VM.

The training data was obtained via Kaggle's [heatbeat dataset](https://www.kaggle.com/kinguistics/heartbeat-sounds).