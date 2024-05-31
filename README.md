# VuQuocDung20225488
Sign Language Classification using CNN 
This is our source code, including four main files. 
Firstly, due to the large size of our training and testing data(more than 2GB), we uploaded it to these 2 links:
- "https://drive.google.com/file/d/1FOj1n1BoU1U9MDuT0rjww6xI76ON6GlV/view?usp=sharing", this one is for train and validity dataset
- "https://drive.google.com/drive/folders/1gZ7esm-I1as9vo-rbCKwsFHITHo7z_XV?usp=sharing", this one is our testing dataset.

"model.py" file is used to run our model, please change the directory in loading data part to before run it, model after training will be stored in "gray_model.h5".

However, it takes nearly 2 hours for all of the process; so we have already saved our model in the "gray_model.h5" so that you can use it to predict images by yourself without training process.

"test_for_newdata.py" is used to return model's accuracy for a total new dataset, you only have to change the directory of the folder you want to predict on the code.

"predict_image.py" is the fie used for predicting single hand sign image. You need to copy the link of the picture you want to classify and paste it to the directory. It will then rescale the image and predict the letter.
