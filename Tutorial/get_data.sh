BASE_URL=https://storage.googleapis.com/tensorflow/tf-keras-datasets/
FILES=(train-labels-idx1-ubyte.gz
train-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz)


for f in ${FILES[@]}; do
    wget "${BASE_URL}${f}" -P data
    gunzip "data/$f"
done

python3 -m venv venv
source venv/bin/activate
pip install pypng
pip install numpy
python3 mnist_to_png.py
