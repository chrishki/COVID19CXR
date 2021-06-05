# COVID19CXR

The following model is a convolutional neural network (CNN) that seeks to detect COVID-19 from a chest radiograph (CXR). Model weights can be downloaded from https://drive.google.com/drive/folders/1BdK0IZNSq0mU9SVsk5e4--QpLHyS_6pl?usp=sharing. To use the COVID-19 diagnosis model, run the line below. On a standard CPU, predictions should take less than 10 seconds to generate. 
  
```python
python run.py [PATH_TO_DCM_FILE]
```
  
Training and testing scripts assume a CUDA device is installed. Images must be padded and resized to 224x224 prior to model training and testing.
  

For all inquiries, please contact chris_kim@alumni.brown.edu.
