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

The Discriminator's goal is to be able to tell if a given input is deemed fake rather than real, providing a probability of the input being either real or fake.

The Generator's goal is to create images that will fool the Discriminator by the latter saying the image has a high probability of being real.

In the training process, we only update the weights and biases of one of the models. If we are training the Discriminator, we will not touch on the settings of our Generator network. The same is valid for the Generator. In more details that's the training process:

The Discriminator takes examples of images from a real dataset (X), and from a fake dataset (X*) that we are generating. Then it computes how much it has mistaken the classifications and updates its weights and biases to minimize what we call loss. Here, the Generator remains still.

The Generator takes input from a random noise source (z), and it produces a fake image as output (X*). Then it computes how much the Discriminator has mistaken the classification of the fake images, and updates its weights and biases to maximize the loss. Here, the Discriminator remains still.

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

Now let's build the generator. In the Fully connect layer part, it takes in the noise vector of size 100 we defined earlier, and connects it to 128 nodes in our first neural network layer. Then it gets connected to our first and only hidden layer, which is using a Leaky ReLU activation function. 

An activation function is a function that will put our values in a defined range. The Leaky ReLU doesn't limit positive numbers, meaning if we put in 16 as a number it wouldn't do anything to it and would still print out 16. However for negative numbers it would make than bigger (that is approaching to zero), in an order of 100 times. The following image may better clarify what Leaky ReLU does, but if you still didn't understand it, just think of it as a special layer in neural network that prevents gradients from dying out during training, improving the quality of our Generator.

![Leaky ReLU image](https://github.com/luispengler/me/blob/main/static/blog/practice-understanding-gans/leaky-relu.png?raw=true)

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

    return model, model.summary()
    
```

```python {linenos=true}
build_generator((28, 28, 1), 100)
```

{{< highlight plaintext >}}
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 128)               12928     
                                                                 
 leaky_re_lu (LeakyReLU)     (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 784)               101136    
                                                                 
 reshape (Reshape)           (None, 28, 28, 1)         0         
                                                                 
=================================================================
Total params: 114,064
Trainable params: 114,064
Non-trainable params: 0
_________________________________________________________________

(<keras.engine.sequential.Sequential at 0x7f717cff1d90>, None)
{{< /highlight >}}

## Full Code
In case you already understand the whole code structure, feel free to just run the code provided below.

The code is also available here as a jupyter notebook.

### Imports

{{< highlight python >}}
# Import statements
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

{{< /highlight >}}


{{< highlight python >}}
### Model input dimensions
img_rows = 28
img_cols = 28
channels = 1

# Input image dimensions
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100
{{< /highlight >}}

### Generator

{{< highlight python >}}
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

{{< /highlight >}}

{{< highlight python >}}
build_generator((28, 28, 1), 100)

{{< /highlight >}}

{{< highlight plaintext "linenos=false">}}
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_4 (Dense)             (None, 128)               12928     
                                                                 
 leaky_re_lu_2 (LeakyReLU)   (None, 128)               0         
                                                                 
 dense_5 (Dense)             (None, 784)               101136    
                                                                 
 reshape_2 (Reshape)         (None, 28, 28, 1)         0         
                                                                 
=================================================================
Total params: 114,064
Trainable params: 114,064
Non-trainable params: 0
_________________________________________________________________

{{< /highlight >}}

### Discriminator
{{< highlight python >}}
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

{{< /highlight >}}


### Build the Model
{{< highlight python >}}
def build_gan(generator, discriminator):

    model = Sequential()

    # Combined Generator -> Discriminator model
    model.add(generator)
    model.add(discriminator)

    return model

{{< /highlight >}}

{{< highlight python >}}
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
{{< /highlight >}}

### Training
{{< highlight python >}}
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

{{< /highlight >}}

{{< highlight python >}}
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

{{< /highlight >}}

### Train the GAN and Inspect Output
Note that the `'Discrepancy between trainable weights and collected trainable'` warning from Keras is expected. It is by design: The Generator's trainable parameters are intentionally held constant during Discriminator training, and vice versa.

{{< highlight python >}}
# Set hyperparameters
iterations = 20000 # It takes 1h to run because of this high amount of interations
batch_size = 128
sample_interval = 1000

# Train the GAN for the specified number of iterations
train(iterations, batch_size, sample_interval)

{{< /highlight >}}

{{< highlight plaintext "linenos=false">}}
Streaming output truncated to the last 5000 lines.
4/4 [==============================] - 0s 5ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 6ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 4ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 2ms/step
4/4 [==============================] - 0s 3ms/step
4/4 [==============================] - 0s 4ms/step

{{< /highlight >}}

{{< highlight python >}}
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

{{< /highlight >}}

{{< highlight python >}}
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

{{< /highlight >}}
