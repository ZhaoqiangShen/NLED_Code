Paper: Image Denoising via Sequential Ensemble Learning, TIP, 2020.

Test Enviroment: Matlab 2018a

How to run?


--- 
Quick demo

Change the current directory to './demo'. 

Run test_demo.m to use a pre-trained model to denoise.


--- 
Complete training and test

Change the current directory to './src'. 

Set parameters (e.g., sigma, filter size, number of training images) in train.m and then run
>>train
Trained models will be saved in './result'.

If models are already saved, you need only run
>>test(model_path)


---
Note：The current version of training code is being optimized, it may lead to slight performance difference from that of paper.




