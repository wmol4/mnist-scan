# mnist-scan

This program does four things:
1. Creates a dataset by stitching together various MNIST images randomly.
2. Classifies MNIST dataset with over 99% accuracy (neural network #1)
3. Determines if a digit is centered or not (neural network #2)
4. Scans over a stitched together image, finds centered digits, and classifies the digit.

The entire process can be viewed below. The code for generating this image can be found in find-digits.py

An example of a stitched together image:
![image](https://cloud.githubusercontent.com/assets/24555661/25556458/1a39776a-2cba-11e7-8e8b-b1818a98e75f.png)

The found digits:
![image](https://cloud.githubusercontent.com/assets/24555661/25556465/329a5720-2cba-11e7-80a7-4db639fa2dd8.png)

![image](https://cloud.githubusercontent.com/assets/24555661/25556473/4d17869a-2cba-11e7-87a0-d5345e2164a3.png)

![image](https://cloud.githubusercontent.com/assets/24555661/25556474/52575d74-2cba-11e7-905f-a75d809ffc14.png)

![image](https://cloud.githubusercontent.com/assets/24555661/25556477/57bb5e1e-2cba-11e7-98d2-45302ea2bef1.png)

![image](https://cloud.githubusercontent.com/assets/24555661/25556478/5ca5b7c6-2cba-11e7-9895-37c08aa96df3.png)

![image](https://cloud.githubusercontent.com/assets/24555661/25556479/62495e4e-2cba-11e7-9adc-a529172c83de.png)

![image](https://cloud.githubusercontent.com/assets/24555661/25556481/74c69afa-2cba-11e7-849c-5bc5a85f6e9a.png)

The two neural networks used in this program both utilize GPU computing. Computations will go significantly faster on a strong graphics card. The total time spent scanning, finding, and classifying the digits in one image takes a GTX 980 a little less than one second.
