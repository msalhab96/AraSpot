# AraSpot: Arabic Spoken Command Spotting

This is the official implementation of AraSpot paper, which achieved SOTA on the ASC dataset 99.59% accuracy on the test set, by introducing the ConformerGRU as shown in the image below, applying online data augmentation, and synthetic data generation.

![model](https://user-images.githubusercontent.com/61272193/207247142-7a667756-2aea-4519-ac4b-ba86221464dc.png)


# Dataset
You can find the ASC dataset [here](https://www.google.com/)

# Results
The below image shows the results across different experiments as illustrated in the paper, while the black horizontal line is the best-performing model on the literature.

![results](https://user-images.githubusercontent.com/61272193/207247264-68f7ac99-dd0e-4a92-b719-ddc1aeb1368f.png)

# Try it out
1. Install the requiremnts using the command below
```bash
pip install -r requirements.txt
```
2. Download the model from [here](https://drive.google.com/drive/folders/1p7GM39U08bFlg1LTs_CPIXdbAX9uMJsR?usp=sharing), or the best performing model from [here](https://drive.google.com/file/d/1WvRoesQDzAeZI_Vx3L6_EeLrp5vMzuJI/view?usp=sharing)

3. Use the function ```single_predict``` in ```predict.py``` file.
