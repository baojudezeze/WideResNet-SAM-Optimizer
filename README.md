# EE5438HW2

The structure of the program I wrote in this CIFAR-10 classified coding assignment can be mainly summarized as WideResNet + SAM + Cutout. Since I'm not very comfortable programming in Jupyter Notebooks, I rewrote the program in my local IDE. The project file consists of three main files called main.py, model.py, and utils.py. The project runs the main program within main.py, the model (WideResNet) are in the model file, and the utils file contains all other functions needed for model training (such as StepLR, SAM optimizer, and Cutout Func for data augmentation). To improve readability, I removed parts of the code that are not important and only show the key results.

As for my model's specific implementation, Firstly, I used the WideResNet (WRN-16-8), an enhanced deep residual network architecture which increases channels (or, widths) within the fundamental residual blocks for superior learning capacity. In my experiments,WRN-16-8 got similar results to WRN-28-10, but WRN-28-10 required almost twice as many parameters to train. In addition, I also tried PyramidNet, which is a pyramid-like network structure. Theoretically, the PyramidNet-272 can achieve higher accuracy than WRN-28-10, but I gave up due to the huge model training time; During my trials, PyramidNet-110 was about 0.8 percentage point less accurate than WRN-28-10. 

Secondly, the SAM algorithm was employed as the optimizer. SAM simultaneously minimizes loss value and loss sharpness, it needs two forward-backward passes to estimate the "sharpness-aware" gradient. Benchmarking the SGD Optimizer, the SAM Optimizer enhances the model's precision by about 0.5 to 1 percentage points within the CIFAR-10 data set.

Lastly, the implementation of the Cutout technique was applied for data augmentation. In my experiments, the model without data augmentation was about 2 percentage points less accurate than the model with data augmentation. Furthermore, the model's learning rate was tweaked via the StepLR Function, which diminishes the learning rate according to an increase in iterations.

Cumulatively speaking, this structure yielded a testing accuracy exceeding 97.21%. Theoretically, the combination of this model and optimization algorithm can achieve an accuracy of over 98%, but this requires a longer number of iterations and data augmentation tricks, which is also a big burden on the GPU. The highest accuracy I know of that can be achieved today on the CIFAR-10 dataset is over 99%, but that is achieved with a Vision Transformer (ViT) and requires extra training data.

## Reference
https://arxiv.org/pdf/2011.14660v4.pdf

https://arxiv.org/pdf/2010.01412v3.pdf
