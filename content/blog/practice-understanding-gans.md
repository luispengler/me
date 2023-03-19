---
title: "Practice: Understanding GANs"
date: 2023-03-15T23:38:34-04:00
draft: false
showToc: true
---
## Introduction

In this blog post I will walk you through creating your first GAN for MNIST dataset image generation. We will be running our code on Google Colab, but if you have access to a GPU or any other server feel free to do it there, the code will be pretty much the same.

If you are new to the MNIST dataset, I can sum it up saying it is a dataset containing 70k handwritten numbers ranging from 0 to 9, as seen in the picture below.

![MNIST Dataset](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/example-mnist.jpg?raw=true)

Our GAN model will be used for creating images that ressamble those from the MNIST dataset. Which in turn means that by the end of this practice we will have created a GAN that can write down some numbers :)

## Explaining the model

Before going through the code it is important we understand better what we will be creating.

As you may know GANs are composed by two neural networks: Generator and Discriminator.

The Discriminator's goal is to be able to tell if a given input is deemed real rather than fake, providing a probability of the input being either real or fake.

The Generator's goal is to create images that will fool the Discriminator by the latter saying the image has a high probability of being real.

In the training process, we only update the weights of one of the models. If we are training the Discriminator, we will not touch on the settings of our Generator network. The same is valid for the Generator. In more details that's the training process:

The Discriminator takes examples of images from a real dataset (X), and from a fake dataset (X*) that we are generating. Then it computes how much it has mistaken the classifications and updates its weights to minimize what we call loss. Here, the Generator remains still.

The Generator takes input from a random noise source (z), and it produces a fake image as output (X*). Then it computes how much the Discriminator has mistaken the classification of the fake images, and updates its weights to maximize the loss. Here, the Discriminator remains still.

![MNIST Dataset](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan-layout.png?raw=true)
Above you can see the GAN structure we will be creating, which is the summary of what I just described. It is a very general GAN model, but it will be sufficient for generating an understanding of this kind of machine learning framework.

## Hands-on!
In this session I will be breaking down some parts of the code that are generally confusing, but if you are ready you can have the full code, without unnecessary parts, in the [next session](#full-code).

### The MNIST dataset images
Let's first get an understanding of the MNIST dataset images.

```python {linenos=true}
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

{{< highlight plaintext >}}
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step
{{< /highlight >}}

We have a very nicely structured numpy array as our training dataset.

```python {linenos=true}
type(X_train)
```
{{< highlight plaintext >}}
numpy.ndarray
{{< /highlight >}}

Having a look into the dimensions. Below 60000 means the amount of images we have. In the case of X_train we have 60k images. 28, 28 correspond to the height and width (AKA dimensions) of each one of these 60k images.

```python {linenos=true}
X_train.shape
```
{{< highlight plaintext >}}
(60000, 28, 28)
{{< /highlight >}}

We can even confirm this by getting the dimensions of one image.
```python {linenos=true}
X_train[0].shape
```
{{< highlight plaintext >}}
(28, 28)
{{< /highlight >}}

Then plotting it so we see it.

```python {linenos=true}
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='gray'
```
![MNIST Dataset](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/x_train[0].png?raw=true)

Now that you understood we are working with 28x28 images, time to define the output dimensions to our generator/input dimensions to our discriminator. That is saying they will be working with 28x28 images. 

Notice we are also defining channels below. It is equal to one because we want to work with only one channel of color. If we wanted RGB (Red, Green, Blue), we would define channels equal to 3.
```python {linenos=true}
## Model input dimensions
img_rows = 28
img_cols = 28
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)
```
Here we will also define the size of the random noise source we will be using. It can also be called noise vector as you see in the comment below.
```python {linenos=true}
# Size of the noise vector, used as input to the Generator
z_dim = 100
```

### Generator
#### Short explanation
Now let's build the generator. In the `Fully connected layer` part, it takes in the noise vector of size 100 we defined earlier, and connects it to 128 units in our first neural network layer. Then it gets connected to our first and only hidden layer, which is using a `Leaky ReLU activation` function. 

An activation function is a function that will put our values in a defined range. The Leaky ReLU doesn't limit positive numbers, meaning if we put in 16 as a number it wouldn't do anything to it and would still print out 16. However, for negative numbers it would make them bigger (that is, make them approach zero), in an order of 100 times. The following image may better clarify what Leaky ReLU does, but if you still didn't understand it, just think of it as a special layer in our neural network that prevents gradients from dying out during training, improving the quality of our Generator.

![Leaky ReLU image](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/leaky-relu.png?raw=true)

The almost-last layer (`Output layer with tanh activation`) of our neural network is the one that gives us images, although in the format of 28x28 (which equals 784) by 1, and uses the tanh activation function. This activation function formats the pixel values of the image to be in -1 to 1 range. This gives our generator the ability to produce crisper images.

Let's have a better understanding of what I meant by this pixel range (-1 to 1). Every image is made up by numbers, even our beautiful MNIST handwritten digits! To see those numbers, let's print out the image instead of showing it...

Previously we displayed the number five, which is the zeroth image of the MNIST dataset, using matplotlib:

```python {linenos=true}
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='gray')
```
![MNIST Dataset](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/x_train[0].png?raw=true)

Seeing the numbers that make up this image is easy:
```python {linenos=true}
print(X_train[0])
```
{{< highlight plaintext >}}
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136
  175  26 166 255 247 127   0   0   0   0]
 [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253
  225 172 253 242 195  64   0   0   0   0]
 [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251
   93  82  82  56  39   0   0   0   0   0]
 [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119
   25   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253
  150  27   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252
  253 187   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249
  253 249  64   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253
  253 207   2   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253
  250 182   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201
   78   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]]
{{< /highlight >}}

We can almost see the number five being formed by the numbers that were not zero... These numbers are in grayscale (0 to 255 range), and when used `cmap='gray'` in matplotlib, it will show a black and white image like the one we got before. Now you understand that changing the range of the pixel values, changes the range of the numbers that make up an image (tanh would transform 0 to 255, to -1 to 1). We will see more of this in the [training subsection](#training).

Now the real-last layer (`Reshape`) of our neural network simply reshapes the images which are in 784 by 1 to our normal size images: 28 by 28. That is, before getting into this last layer, we had some sort of array in the dimensions 784x1. Watch out for the difference:

```python {linenos=true}
X_train[0].shape
```
{{< highlight plaintext >}}
(28, 28)
{{< /highlight >}}

```python {linenos=true}
import numpy as np
c = np.reshape(X_train[0], (28*28, 1))
c.shape
```
{{< highlight plaintext >}}
(784, 1)
{{< /highlight >}}

The reshape layer takes the (784, 1) array and makes it a (28, 28) array that you just saw above!

#### Code

Let us finally get into the code for creating this Generator

```python {linenos=true}
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU

def build_generator(img_shape, z_dim):

    model = Sequential()

    # Fully connected layer
    model.add(Dense(128, input_dim=z_dim))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with tanh activation
    model.add(Dense(28 * 28 * 1, activation='tanh'))

    # Reshape the Generator output to image dimensions
    model.add(Reshape(img_shape))

    return model
    
```

### Discriminator
#### Short explanation

You may notice the Discriminator and the Generator are very similar networks, however in most GANs implementations they often very greatly in both size and complexity. 

The first layer (`Flatten`) takes in the images in the format of 28x28 and reshapes them to 784x1. Now you might ask me why we went through the small hassle of adding another layer in our GAN model for reshaping since our work would be "undone" when it got to the input of the discriminator. The reason we reshaped the output of the Generator network was for its images to blend in with the real images which are already in the 28x28 format, regardless of the Discriminator network taking in 784x1 images as input.

Then, the reshaped images will go into a `Fully connected layer` of 128 units. Then it gets connected to our first and only hidden layer that uses a `Leaky ReLU` activation function. This part is identical to the Generator network.

The last layer will output probabilities, therefore we are using the `Sigmoid` activation function, which maps all the outputs in the range of 0 to 1.

#### Code
```python {linenos=True}
def build_discriminator(img_shape):

    model = Sequential()

    # Flatten the input image
    model.add(Flatten(input_shape=img_shape))

    # Fully connected layer
    model.add(Dense(128))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    return model
```

### Building the Model
#### Short explanation

In `build_gan()` we add Generator and Discriminator together in our `model`.

The Discriminator is compiled alone, taking as input `img_shape` (which we defined earlier) and the loss being computed using `binary_crossentropy'`. Binary cross-entropy is a measure of the difference between computed probabilities and actual probabilities for predictions with only two possible classes. In our case the possible classes are real or fake. 

Therefore, binary cross-entropy is going to tell us how off the prediction of the Discriminator is. Remember the Discriminator's goal is to not be off in the predictions, while the Generator wants the Discriminator to be very off when it comes to fake images predictions. This also means the Discriminator wants to minimize its loss for real and fake images, and the Generator wants to maximize the Discriminator loss for fake images.

The `optimizer` argument is required for compiling the model, although I don't know how the `Adam()` optimizer works, it has become the default in many GAN implementations due to its often superior performance. However, it suffices to say the optimizer is the one who is responsible for updating the weights and learning rates of the neural network. 

The `'accuracy'` metrics is the way the Discriminator, and later us, will know how well it is doing. 

The Generator is built taking as input `img_shape` and the `z_dim` arguments we defined earlier. Then, we only compile the Generator together with the Discriminator in the `gan` argument. For doing this, we first neet to set `discriminator.trainable` to `false`. 

To clarify what I just stated, notice how below we *built* and *compiled* the Discriminator alone. This means that when we use the Discriminator and train it, it will take care of its own weights without interfering with the Generator network. Since it is alone, it can't interfere with anyone else. In turn, the generator is *built* alone, but it is *compiled* only when it is together with the Discriminator. It is only possible to train the Generator without tweaking the Discriminator weights by mistake if we set `discriminator.trainable` to false.

Lastly, we save a file called `generator_model.h5` containing the Generator network model. After trained, we can even use this file in other projects with the same abilities our generative model will have.


#### Code
```python {linenos=true}
def build_gan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model

```

```python {linenos=true}
from keras.optimizers import Adam

# Build and compile the Discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Build the Generator
generator = build_generator(img_shape, z_dim)

# Keep Discriminator’s parameters constant for Generator training
discriminator.trainable = False

# Build and compile GAN model with fixed Discriminator to train the Generator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

generator.save('generator_model.h5')
```

{{< highlight plaintext >}}
WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
{{< /highlight >}}


### Training
#### Short explanation
Next, we will build a training loop for our GAN.

The following command only gives us the `X_train` dataset from MNIST dataset. You can see that all the 60k images in the format of 28x28 are there.
```python {linenos=true}
(X_train, _), (_, _) = mnist.load_data()
X_train.shape
```
{{< highlight plaintext >}}
(60000, 28, 28)
{{< /highlight >}}

However, they are still in the 0 to 255 range we saw earlier.
```python {linenos=true}
print(X_train[0])
```
{{< highlight plaintext >}}
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136
  175  26 166 255 247 127   0   0   0   0]
 [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253
  225 172 253 242 195  64   0   0   0   0]
 [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251
   93  82  82  56  39   0   0   0   0   0]
 [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119
   25   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253
  150  27   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252
  253 187   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249
  253 249  64   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253
  253 207   2   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253
  250 182   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201
   78   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]]
{{< /highlight >}}

The issue is that the real images are very distinguishable from the fake ones, since we used `tanh` activation function in the Generator code. Therefore the generator is working with images that go from -1 to 1. The fake images are in this -1 to 1 range, while the real ones are in 0 to 255. Very easy to spot their differences like this. Let us fix this by setting the range of values of the real images to *also* be in -1 to 1.

```python {linenos=True}
X_train = X_train / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=3)
```
Now we print another 5 from the real dataset to check if they are correct...
```python {linenos=true}
print(X_train[0])
```
{{< highlight plaintext >}}
[[[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]

 [[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]

 [[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]

 [[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]

 [[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]

 [[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-0.97647059]
  [-0.85882353]
  [-0.85882353]
  [-0.85882353]
  [-0.01176471]
  [ 0.06666667]
  [ 0.37254902]
  [-0.79607843]
  [ 0.30196078]
  [ 1.        ]
  [ 0.9372549 ]
  [-0.00392157]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]

 [[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-0.76470588]
  [-0.71764706]
  [-0.2627451 ]
  [ 0.20784314]
  [ 0.33333333]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.76470588]
  [ 0.34901961]
  [ 0.98431373]
  [ 0.89803922]
  [ 0.52941176]
  [-0.49803922]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]

 [[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-0.61568627]
  [ 0.86666667]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.98431373]
  [ 0.96862745]
  [-0.27058824]
  [-0.35686275]
  [-0.35686275]
  [-0.56078431]
  [-0.69411765]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]

  [[...19 OTHERS...]]  
 [[-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]
  [-1.        ]]]
{{< /highlight >}}

Woosh, that was big. I even had to truncate 19 of these so it wouldn't bother you too much. However, notice in the ones that remained how they were never less than -1 or greater than 1. It doesn't look like a five because we lost the spaces when printing it out, but what matters is the numbers and that this image is still 28x28. I removed 19 units from the rows and columns, but if you count the ones that remained (that is 9 units), you realize that they total 28 (19+9). In fact, if I try to render this -1 to 1 image again, you will see it is still a five!

```python {linenos=true}
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='gray')
```
![A reranged 5](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/x_train[0]_ranged.png?raw=true)

Next with `real = np.ones...` and `fake = np.zeros...` we are creating a 1-dimensional numpy array that will be our labels for real and fake images. As the code suggest, we are encoding real as 1 and fake as 0.

#### Training for loop
Our training for loop is very important. It defines the steps that will be followed during training.

First, we are training the discriminator. We get a random batch of real images. Then we get a random batch of fake images produced by our generator. Now the last step in the discriminator training is to make predictions of what classes those two batches belong to, and give it the right answer so it can calculate how much off the predictions are and update its weights.

For the generator training, we ask it to create fake images that we will label as real, and see what the discriminator will think about this lying act of ours. Lastly, we hold the discriminator weights still and update the generator's.

The last step in the Training code is not necessary for training the GANs, but it will be useful for our understanding of them. We will save their progress for later plotting in a graph. We will also take "snapshots" of what the generator is producing so we can see how well it can handwrite...

#### Code
```python {linenos=True}
losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):

    # Load the MNIST dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)

    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Get a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Train Generator
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(generator)
```

```python {linenos=true}
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
```


### Actually training + Inspecting Output
#### Short explanation
We covered pretty much every part of the code. Now we are just defining the hyperparameters. While machine learning models are usually very sensible to hyperparameters, our GAN model is simple and therefore is less sensible to have bad hyperparameters. Of course, your images will become pretty bad if you set bad hyperparameters, but it is not the end of the world. The number of `iterations` defines how many times we will go through the training loop we just saw a code block ago. It took me one hour to run it on colab, it might take you a different time running somewhere else. An ideal iteration number for this GAN would be 100,000. However, I don't want to wait 5h just to get images for a practice tutorial... Maybe you don't want to wait that much time to learn the content either.

Batch size will tell our training loop how much images to get for the Discriminator to predict at. And lastly `sample_interval` defines after how many iterations to print us a 4x4 grid with the progress of the Generator network.

Also, if you get a warning running this part of the code as `Discrepancy between trainable weights and collected trainable`, it is just Keras complaining we held the Discriminator's parameters constant while training the Generator.

The last two blocks of code are simply the ones that will use matplotlib to output us the 4x4 grid every `sample_interval`

#### Code
```python {linenos=true}
# Set hyperparameters
iterations = 20000 # It takes 1h to run because of this high amount of interations
batch_size = 128
sample_interval = 1000

# Train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval
```
{{< highlight plaintext >}}
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 6ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 5ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 3ms/step
... ALSO TRUNCATED HERE SO IT DOESN'T TAKE MUCH SPACE...
{{< /highlight >}}
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_1.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_2.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_3.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_4.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_5.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_6.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_7.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_8.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_9.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_10.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_11.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_12.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_13.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_14.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_15.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_16.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_17.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_18.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_19.png?raw=true)
![GAN generated digit](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/gan_20.png?raw=true)


```python {linenos=true}
losses = np.array(losses)

# Plot training losses for Discriminator and Generator
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
```
![Training loss graph](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/training_loss.png?raw=true)

```python {linenos=true}
accuracies = np.array(accuracies)

# Plot Discriminator accuracy
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")

plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()
```
![Discriminator accuracy graph](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/disciminator_accuracy.png?raw=true)

## Full Code
In case you already understand the whole code structure, feel free to just run the code provided below.

The code is also available [here](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/GAN.ipynb) as a jupyter notebook.

### Imports

```python {linenos=true}
# Import statements
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

```


```python {linenos=true}
### Model input dimensions
img_rows = 28
img_cols = 28
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100
```

### Generator

```python {linenos=true}
def build_generator(img_shape, z_dim):

    model = Sequential()

    # Fully connected layer
    model.add(Dense(128, input_dim=z_dim))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with tanh activation
    model.add(Dense(28 * 28 * 1, activation='tanh'))

    # Reshape the Generator output to image dimensions
    model.add(Reshape(img_shape))

    return model

```

### Discriminator

```python {linenos=true}
def build_discriminator(img_shape):

    model = Sequential()

    # Flatten the input image
    model.add(Flatten(input_shape=img_shape))

    # Fully connected layer
    model.add(Dense(128))

    # Leaky ReLU activation
    model.add(LeakyReLU(alpha=0.01))

    # Output layer with sigmoid activation
    model.add(Dense(1, activation='sigmoid'))

    return model

```


### Building the Model

```python {linenos=true}
def build_gan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model

```

```python {linenos=true}
# Build and compile the Discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Build the Generator
generator = build_generator(img_shape, z_dim)

# Keep Discriminator’s parameters constant for Generator training
discriminator.trainable = False

# Build and compile GAN model with fixed Discriminator to train the Generator
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

generator.save('generator_model.h5')
```

### Training
```python {linenos=true}
losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):

    # Load the MNIST dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale [0, 255] grayscale pixel values to [-1, 1]
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)

    # Labels for real images: all ones
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Get a random batch of real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Train Discriminator
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # Train Generator
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # Save losses and accuracies so they can be plotted after training
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # Output training progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            # Output a sample of generated image
            sample_images(generator)

```

```python {linenos=true}
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):

    # Sample random noise
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    # Rescale image pixel values to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Set image grid
    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output a grid of images
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

```

### Actually training + Inspecting Output
Note that the `'Discrepancy between trainable weights and collected trainable'` warning from Keras is expected. It is by design: The Generator's trainable parameters are intentionally held constant during Discriminator training, and vice versa.

```python {linenos=true}
# Set hyperparameters
iterations = 20000 # It takes 1h to run because of this high amount of interations
batch_size = 128
sample_interval = 1000

# Train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval)

```

```python {linenos=true}
losses = np.array(losses)

# Plot training losses for Discriminator and Generator
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")

plt.xticks(iteration_checkpoints, rotation=90)

plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()

```

```python {linenos=true}
accuracies = np.array(accuracies)

# Plot Discriminator accuracy
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, accuracies, label="Discriminator accuracy")

plt.xticks(iteration_checkpoints, rotation=90)
plt.yticks(range(0, 100, 5))

plt.title("Discriminator Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.legend()

```
