# <center> Luna-Tensorflow MNIST Tutorial </center>

## Cloning repository.

```bash
git clone -b MNIST_tutorial https://github.com/Luna-Tensorflow/Luna-Tensorflow.git
cd Luna-Tensorflow/Tutorial
```

## Building libraries.
```bash
cd local_libs/Tensorflow/native_libs/
mkdir build
cd build
cmake ../src
make
cd ../../../..
```

## Downloading and preprocesing data.
Unfortunatelly, full dataset is quite heavy, so we need to cut it a little, with additional preprocessing.
```bash
chmod +x get_data.sh
./get_data.sh
venv/bin/python3 mnist_to_png.py
```

## Let's start with Luna Studio.

In the beggining we need some imports.

```
import Std.Base
import Tensorflow.Tensor
import Tensorflow.Types
import Tensorflow.Layers.Input
import Tensorflow.Layers.Dense
import Tensorflow.Layers.Reshape
import Tensorflow.Operations
import Tensorflow.Losses.Crossentropy
import Tensorflow.Optimizers.Adam
import Tensorflow.Model
```

The size of dataset labels is the number of different digits to distinguish.

```
def labelsCount:
    10
```

There is need to use some helper functions, to load dataset and work with it, like creating list of given value.

```
def nTimes n val:
    def helper acc m:
        if m > 0 then helper (acc.prepend val) (m - 1) else acc
    helper [] n
```

The training and testing labels are one hot encoded. It's simply list of length `labelsCount`, filled with 0, with 1 on index corresponding to presented value.

```
def oneHot label:
    oneHotList = 0.upto (labelsCount - 1) . each l: if l == label then 1.0 else 0.0
    Tensors.fromList FloatType [labelsCount] oneHotList
```

![](Screenshots/oneHot/oneHot.png)

Function to load training and testing images from png format will be neccessary too.
<b> TODO </b> More explanations needed.

```
def getData path:
    labels = 0.upto (labelsCount - 1)
    labelTensors = labels.each oneHot
    tensorLists = labels.each label: Tensors.fromPngDir (path + "/" + label.toText)
    ys = labelTensors.zip tensorLists . foldLeft [] ((label, tList): acc: (nTimes tList.length label) + acc)
    xs = tensorLists.foldLeft [] (acc: tList: tList + acc)
    (xs, ys)

```

![](Screenshots/getData/getData.png)

And finally helper function to prepare optimizer.

```
def prepareOptimizer:
    beta1 = 0.9
    beta1Power = beta1
    beta2 = 0.999
    beta2Power = beta2
    lr = 0.001
    epsilon = 0.00000001
    useNesterov = False

    AdamOptimizer.create beta1Power beta2Power lr beta1 beta2 epsilon useNesterov
```

![](Screenshots/prepareOptimizer/prepareOptimizer.png)

## Now we can handle model training and testing.

Let's focus on smaller details of Luna Tensorflow API.

<table>

<tr><th> Code </th><th> Node editor </th></tr>

<tr><td>

```
def main:
    (xTrain, yTrain) = getData 
        "data/train"
    (xTest, yTest) = getData 
        "data/test"

    xTrainBatch = Tensors.batchFromList
        xTrain
    yTrainBatch = Tensors.batchFromList 
        yTrain

    xTestBatch = Tensors.batchFromList 
        xTest
    yTestBatch = Tensors.batchFromList 
        yTest

```
</td><td>

Data loading.
![](Screenshots/main/loadData.png)

</td></tr> 

<tr><td>

```

    input = Input.create 
        FloatType 
        [28, 28, 3]

    reshape = Reshape.flatten 
        input

    dense1 = Dense.createWithActivation 
        128 
        Operations.relu 
        reshape

    dense2 = Dense.createWithActivation 
        10 
        Operations.softmax 
        dense1

```
</td><td>

Adding fully connected hidden and output layers.
![](Screenshots/main/layers.png)

</td></tr> 

<tr><td>

```
    optimizer = prepareOptimizer

    loss = Losses.categoricalCrossEntropy

    model = Models.make 
        input 
        dense2 
        optimizer 
        loss

```
</td><td>

Building model with its parameters.
![](Screenshots/main/model.png)

</td></tr> 

<tr><td>

```

    untrainedAccuracy = accuracy 
        model 
        xTestBatch 
        yTestBatch

    (h, trained) = model.train 
        [xTrainBatch] 
        [yTrainBatch] 
        30 
        (ValidationFraction 0.1) 
        0

    history = h

    trainedAccuracy = accuracy 
        trained 
        xTestBatch 
        yTestBatch

    None
```
</td><td>

Training model and accuracy ratio comparision.
![](Screenshots/main/test.png)

</td></tr> 

</table>


Evaluated model let us observe the ratio of training process in the node named `trainedAccuracy`.

![](Screenshots/main/trainedAccuracy.png)

In Node editor we can look at `main` function in full effect.
<b> TODO </b> Better image.

![](Screenshots/main/main.png)


