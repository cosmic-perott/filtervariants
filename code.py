import kagglehub
puneet6060_intel_image_classification_path = kagglehub.dataset_download('puneet6060/intel-image-classification')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def df_maker(path):
    file_paths = []
    labels = []

    folds = os.listdir(path)
    for fold in folds:
        fold_path = os.path.join(path,fold)
        file_list = os.listdir(fold_path)
        for file in file_list:
            file_path = os.path.join(fold_path,file)
            file_paths.append(file_path)
            labels.append(fold)


    file_series = pd.Series(file_paths,name="file_paths")
    label_series = pd.Series(labels,name="labels")

    df = pd.concat([file_series,label_series],axis=1)
    return df

train_df = df_maker(puneet6060_intel_image_classification_path+ '/seg_train/seg_train')
test_df =  df_maker(puneet6060_intel_image_classification_path+ '/seg_test/seg_test')

target_size= (150,150)
batch_size = 32

import tensorflow as tf


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cv2 import imread

dg_args    = dict(fill_mode = 'reflect',
                   data_format = 'channels_last')

valid_args = dict(fill_mode = 'reflect',
                   data_format = 'channels_last')

core_idg = ImageDataGenerator(**dg_args)
valid_idg = ImageDataGenerator(**valid_args)

IMG_SIZE = (150, 150)

def flow_from_dataframe(img_data_gen, raw_df, path_col, y_col, **dflow_args):
    """Keras update makes this much easier"""
    in_df = raw_df.copy()
    in_df[path_col] = in_df[path_col].map(str)
    in_df[y_col] = in_df[y_col].map(lambda x: np.array(x))
    df_gen = img_data_gen.flow_from_dataframe(in_df,
                                              x_col=path_col,
                                              y_col=y_col,
                                              shuffle=False,
                                              class_mode = 'raw',
                                              **dflow_args)
    # posthoc correction
    df_gen._targets = np.stack(df_gen.labels, 0)
    return df_gen

# used a fixed dataset for evaluating the algorithm
X, y = next(flow_from_dataframe(valid_idg,
                               train_df,
                             path_col = 'file_paths',
                            y_col = 'labels',
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = len(train_df))) # one big batch
print(X.shape, y.shape)

X = X.astype('uint8')
print("X shape:", X.shape)
print("y shape:", y.shape)
plt.imshow(X[29])
print("Diagnosis:", y[29])

X_gray = []
import cv2

for i in range(len(train_df)):
  X_gray.append(cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY))

X_gray = np.array(X_gray)

plt.imshow(X_gray[0], cmap='gray')

# 1. standard gaussian blur (explicit version)
def gaussian_blur_standard(img, ksize=5, sigma=0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# 2. separable gaussian filter (faster implementation)
def separable_gaussian_filter(img, ksize=5, sigma=0):
    g1d = cv2.getGaussianKernel(ksize, sigma)
    return cv2.sepFilter2D(img, -1, g1d, g1d)


# 3. anisotropic gaussian filter (different sigma per axis)
def anisotropic_gaussian_filter(img, ksize=9, sigma_x=3, sigma_y=1):
    gx = cv2.getGaussianKernel(ksize, sigma_x)
    gy = cv2.getGaussianKernel(ksize, sigma_y)
    kernel = gx @ gy.T
    return cv2.filter2D(img, -1, kernel)


# 4. difference of gaussians (DoG)
def dog_filter(img, ksize1=5, ksize2=9, sigma1=1, sigma2=2):
    g1 = cv2.GaussianBlur(img, (ksize1, ksize1), sigma1)
    g2 = cv2.GaussianBlur(img, (ksize2, ksize2), sigma2)
    return cv2.subtract(g1, g2)


# 5. laplacian of gaussian (LoG)
def log_filter(img, ksize=5, sigma=0):
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return cv2.Laplacian(blurred, cv2.CV_64F)


# 6. gaussian pyramid denoise (multi-scale smoothing)
def gaussian_pyramid_denoise(img, levels=2):
    temp = img.copy()
    for _ in range(levels):
        temp = cv2.pyrDown(temp)
    for _ in range(levels):
        temp = cv2.pyrUp(temp)
    return cv2.resize(temp, (img.shape[1], img.shape[0]))


# 7. recursive gaussian approximation (fast large-kernel smoothing)
def recursive_gaussian_filter(img, ksize=31, sigma=5):
    # approximated using multiple small gaussian passes
    temp = img.copy()
    for _ in range(3):
        temp = cv2.GaussianBlur(temp, (ksize, ksize), sigma)
    return temp


# 8. gaussian weighted mean filter (manual kernel)
def gaussian_weighted_mean_filter(img, ksize=5, sigma=1):
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    kernel_2d = kernel_1d @ kernel_1d.T
    kernel_2d /= kernel_2d.sum()
    return cv2.filter2D(img, -1, kernel_2d)

# 10. adaptive gaussian filter (variance-based)
def adaptive_gaussian_filter(img, ksize=5, sigma=1):
    mean = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    sqr_mean = cv2.GaussianBlur(img**2, (ksize, ksize), sigma)
    variance = sqr_mean - mean**2

    weight = variance / (variance + sigma**2 + 1e-5)
    return (mean + weight * (img - mean)).astype(np.uint8)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_gray_flat, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Applies Gaussian noise to an image.

    Args:
        image (numpy.ndarray): Input image.
        mean (float): Mean of the Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: Image with Gaussian noise.
    """

    # Convert to float to prevent overflow
    image = image.astype(np.float32)

    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, image.shape)

    # Add noise
    noisy_image = image + noise

    # Clip values to valid range
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8)

prediction_results = []

for i in range(100):
    single_image = X_test[i]

    # Reshape the 1D single_image to its original 2D (IMG_SIZE, IMG_SIZE) before adding noise
    single_image_2d = single_image.reshape(IMG_SIZE[0], IMG_SIZE[1])

    # Apply salt and pepper noise to the 2D image
    noisy_image_2d = add_gaussian_noise(single_image_2d, mean=0, sigma=25)

    # Flatten the noisy 2D image back to 1D for the model prediction
    noisy_image_flat = noisy_image_2d.reshape(1, -1)

    # Get prediction probabilities for the noisy image
    probabilities = rf_model.predict_proba(noisy_image_flat)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_label = rf_model.classes_[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    prediction_results.append({
        "Image Number": i,
        "Predicted Class": predicted_label,
        "Confidence": f"{confidence:.4f}"
    })

results_df = pd.DataFrame(prediction_results)

display(results_df)

print("X shape:", X_test.shape)
print("y shape:", y_test.shape)
plt.imshow(X_test[1].reshape(150, 150))
print("Diagnosis:", y_test[1])

prediction_results = []

for i in range(50):
    single_image = X_test[i]

    # Reshape the 1D single_image to its original 2D (IMG_SIZE, IMG_SIZE) before adding noise
    single_image_2d = single_image.reshape(IMG_SIZE[0], IMG_SIZE[1])

    # Apply salt and pepper noise to the 2D image
    noisy_image_2d = add_gaussian_noise(single_image_2d, amount=0.1)

    # Flatten the noisy 2D image back to 1D for the model prediction
    noisy_image_flat = noisy_image_2d.reshape(1, -1)

    # Get prediction probabilities for the noisy image
    probabilities = rf_model.predict_proba(noisy_image_flat)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_label = rf_model.classes_[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    prediction_results.append({
        "Image Number": i,
        "Predicted Class": predicted_label,
        "Confidence": f"{confidence:.4f}"
    })

results_df = pd.DataFrame(prediction_results)

display(results_df)

prediction_results_noisy_gaussian = []

for i in range(50):
    single_image = X_test[i]

    # Reshape the 1D single_image to its original 2D (IMG_SIZE, IMG_SIZE)
    single_image_2d = single_image.reshape(IMG_SIZE[0], IMG_SIZE[1])

    # Apply salt and pepper noise
    noisy_image_2d = add_gaussian_noise(single_image_2d, mean=0, sigma=25)

    # Apply gaussian filter to the noisy image
    #adaptive_gaussian_filter(img, ksize=5, sigma=1):
    gaussian_filtered_noisy_image_2d = adaptive_gaussian_filter(noisy_image_2d, ksize=5, sigma=1)

    # Flatten the filtered noisy 2D image back to 1D for the model prediction
    filtered_noisy_image_flat = gaussian_filtered_noisy_image_2d.reshape(1, -1)

    # Get prediction probabilities for the filtered noisy image
    probabilities = rf_model.predict_proba(filtered_noisy_image_flat)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_label = rf_model.classes_[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    prediction_results_noisy_gaussian.append({
        "Image Number": i,
        "Predicted Class": predicted_label,
        "Confidence": f"{confidence:.4f}"
    })

results_df_noisy_gaussian = pd.DataFrame(prediction_results_noisy_gaussian)

display(results_df_noisy_gaussian)


prediction_results = []

for i in range(50):
    single_image = X_test[i]
    single_image_correct_label = y_test[i]
    correct = 1
    # Reshape the 1D single_image to its original 2D (IMG_SIZE, IMG_SIZE) before adding noise
    single_image_2d = single_image.reshape(IMG_SIZE[0], IMG_SIZE[1])

    # Apply salt and pepper noise to the 2D image
    noisy_image_2d = add_gaussian_noise(single_image_2d, mean=0, sigma=25)

    # Flatten the noisy 2D image back to 1D for the model prediction
    noisy_image_flat = noisy_image_2d.reshape(1, -1)

    # Get prediction probabilities for the noisy image
    probabilities = rf_model.predict_proba(noisy_image_flat)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_label = rf_model.classes_[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    if predicted_label != single_image_correct_label:
      correct = 0
    prediction_results.append({
        "Image Number": i,
        "Predicted Class": predicted_label,
        "Correct Class" : single_image_correct_label,
        "Correct?": correct,
        "Confidence": f"{confidence:.4f}"
    })

results_df = pd.DataFrame(prediction_results)

display(results_df)



prediction_results_noisy_gaussian = []

for i in range(50):
    single_image = X_test[i]
    single_image_correct_label = y_test[i]
    # Reshape the 1D single_image to its original 2D (IMG_SIZE, IMG_SIZE)
    single_image_2d = single_image.reshape(IMG_SIZE[0], IMG_SIZE[1])

    # Apply salt and pepper noise
    noisy_image_2d = add_gaussian_noise(single_image_2d, mean=0, sigma=25)
    correct = 1
    # Apply gaussian filter to the noisy image
    #adaptive_gaussian_filter(img, ksize=5, sigma=1):
    gaussian_filtered_noisy_image_2d = adaptive_gaussian_filter(noisy_image_2d, ksize=5, sigma=1)

    # Flatten the filtered noisy 2D image back to 1D for the model prediction
    filtered_noisy_image_flat = gaussian_filtered_noisy_image_2d.reshape(1, -1)

    # Get prediction probabilities for the filtered noisy image
    probabilities = rf_model.predict_proba(filtered_noisy_image_flat)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_label = rf_model.classes_[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    if predicted_label != single_image_correct_label:
          correct = 0
    prediction_results_noisy_gaussian.append({
        "Image Number": i,
        "Correct?": correct,
        "Predicted Class": predicted_label,
        "Confidence": f"{confidence:.4f}"
    })

results_df_noisy_gaussian = pd.DataFrame(prediction_results_noisy_gaussian)

display(results_df_noisy_gaussian)
