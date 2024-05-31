# VuQuocDung20225488
Sign Language Classification using CNN 
This is our source code, including four main files. 
Firstly, due to the large size of our training and testing data(more than 2GB), we uploaded it to these 2 links:
- "https://drive.google.com/file/d/1FOj1n1BoU1U9MDuT0rjww6xI76ON6GlV/view?usp=sharing", this one is for train and validity dataset
- "https://drive.google.com/drive/folders/1gZ7esm-I1as9vo-rbCKwsFHITHo7z_XV?usp=sharing", this one is our testing dataset.
  
After downloading data, if you want to run through our model, you need to open "model.py" file, change the directory of the loading data phase in our source code and then run thr code
However, it takes nearly 2 hours for all of the process; so we have stored our model in the "gray_model.h5" so that you can use it to predict images by yourself without training the model.

If you want our model to run through a totally new dataset to test for the accuray, you need to open "test_for_newdata.py" file, change the directory to the dataset you want on your code and run it. It will return the model's accuracy for your dataset.
"predict_image.py" is the fie used for predicting single hand sign image. You need to copy the link of the picture you want to classify and paste it to the directory. It will then rescale the image and predict the letter.
