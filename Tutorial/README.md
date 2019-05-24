 <center>

# Luna-Tensorflow Fashion MNIST Tutorial 

Based on: https://www.tensorflow.org/tutorials/keras/basic_classification

</center>

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
Unfortunatelly, Fashion MNIST dataset is in incompatible format, so we have to preprocess it.

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

The size of dataset labels is the number of different pictures to distinguish.

```
def labelsCount:
    10
```

There is need to use some helper functions, to load dataset and work with it, like creating list of given value, used later to make pictures ont hot encoded labels.

```
def nTimes n val:
    def helper acc m:
        if m > 0 then helper (acc.prepend val) (m - 1) else acc
    helper [] n
```

The training and testing labels are one hot encoded. It's simply list of length `labelsCount`, filled with 0, with 1 on index corresponding to presented object.

```
def oneHot label:
    oneHotList = 0.upto (labelsCount - 1) . each l: if l == label then 1.0 else 0.0
    Tensors.fromList FloatType [labelsCount] oneHotList
```

![](Screenshots/oneHot/oneHot.png)

Function to load training and testing images from png format will be neccessary too. It creates one hot encoded labels for each object type on `labelTensors`. Loads dataset of each object type on `tensorLists` and finally concatenate all image tensors into `xs`, and corresponding to them labels into `ys`.

```
def getData path:
    labels = 0.upto (labelsCount - 1)
    labelTensors = labels.each oneHot
    tensorLists = labels.each label: Tensors.fromPngDir (path + "/" + label.toText)
    ys = labelTensors.zip tensorLists . flatMap ((label, tList): (nTimes tList.length label))
    xs = tensorLists.concat
    (xs, ys)

```

![](Screenshots/getData/getData.png)

And last but not least, helper function to prepare optimizer.

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

## Now we can handle building model, training and testing.

Let's focus on details of Luna Tensorflow API.

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

Loading data, preparing training and testing tensors into batches.
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

Connecting models layers in sequential order:
<ul>
<li> input layer, feeded with 28x28 pixels pictures, </li>
<li> reshape layer, flattening to 1D, </li>
<li> hidden fully connected layer with 128 neurons, </li>
<li> output fully connected layer with 10 neurons as label predictions. </li>
</ul>

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

Building model with its parameters: 
<ul>
<li> input and output layers, </li>
<li> optimizer, </li>
<li> loss function. </li>
</ul>

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

Training model, and calculating its accuracy on test dataset before and after whole process.
![](Screenshots/main/test.png)

</td></tr> 

</table>


Evaluated model let us observe the accuracy ratio after training process, in the node named `trainedAccuracy`, with comparision to accuracy ratio before it, in the node named `untrainedAccuracy`.

<center>

![](Screenshots/main/comparision.png)

</center>

In Node editor we can look at `main` function in full effect.

<center>

![](Screenshots/main/main2.png)

</center>

