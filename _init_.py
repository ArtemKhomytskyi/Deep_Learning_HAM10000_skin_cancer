import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten, SpatialDropout1D, Bidirectional, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import os
from PIL import Image
import imagehash

def remove_duplicate_images(folder_path):
    hashes = {}
    duplicates = []
    
    # Go through the files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Open images and find their hashes
            with Image.open(file_path) as img:
                img_hash = imagehash.average_hash(img)
                
            # Check if there is the similar hashes
            if img_hash in hashes:
                duplicates.append(file_path)
            else:
                hashes[img_hash] = file_path
        except Exception as e:
            print(f"Error in working with {file_path}: {e}")

    print("Finished.")
    return duplicates

def remove_duplicates_from_dataset(dataset_path, duplicates_list, output_path=None):
    dataset = pd.read_csv(dataset_path)

    if "image_id" not in dataset.columns:
        raise ValueError("There is no column 'image_id'")

    updated_dataset = dataset[~dataset["image_id"].isin(duplicates_list)]

    if output_path:
        updated_dataset.to_csv(output_path, index=False)
        print(f"New dataset saved in: {output_path}")

    return updated_dataset

def preprocess(image_path, label):
    image = tf.io.read_file('HAM10000_images/'+image_path+'.jpg')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0
    return image, label

def plot_training_history(history):
    # Extract values from the training history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

#Extract labels and predictions from val_dataset
def auc_class_report(test_dataset, model):
    y_val = []
    y_pred_probs = []

    #Iterate over the validation dataset
    for images, labels in test_dataset:
        y_val.extend(labels.numpy())  # Collect true labels
        y_pred_probs.extend(model.predict(images).flatten())  # Predict probabilities

    #Convert to numpy arrays
    y_val = np.array(y_val)
    y_pred_probs = np.array(y_pred_probs)

    #Convert probabilities to binary predictions
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    auc_score = roc_auc_score(y_val, y_pred_probs)

    #Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    #Generate classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred_binary, target_names=['Class 0', 'Class 1']))  # Replace with class names

    #Calculate AUC
    print(f"AUC Score: {auc_score:.4f}")

def prepare_dataframe(image_paths, labels):
    return pd.DataFrame({
        'filename': image_paths,
        'class': labels.argmax(axis=1).astype(str)  # Convert numeric labels to string
    })

def create_image_generators(dataframes, batch_size, target_size):
    # Define ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,        # Normalize pixel values to [0, 1]
        rotation_range=40,        # Random rotation
        width_shift_range=0.2,    # Horizontal shift
        height_shift_range=0.2,   # Vertical shift
        shear_range=0.2,          # Shear transformation
        zoom_range=0.2,           # Random zoom
        horizontal_flip=True,     # Horizontal flipping
        fill_mode='nearest'       # Fill empty pixels after transformation
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only rescaling for validation
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only rescaling for testing

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=dataframes['train'],
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=dataframes['val'],
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Provide an empty placeholder generator for 'test' if not passed
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=dataframes['test'],
        x_col='filename',
        y_col='class',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def matrix_class_report(model, test_generator):
    # Predict class probabilities
    predictions = model.predict(test_generator, steps=len(test_generator))

    # Get predicted class indices
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true class indices
    true_classes = test_generator.classes

    # Get class labels
    class_labels = list(test_generator.class_indices.keys())

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Print classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print("Classification Report:")
    print(report)

# define autoencoder
def create_autoencoder(input_shape):
    # encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    bottleneck = MaxPooling2D((2, 2), padding='same', name='bottleneck')(x)

    # decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(bottleneck)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

# define classifier on top of bottleneck features
def create_classifier(autoencoder, num_classes):
    bottleneck_output = autoencoder.get_layer('bottleneck').output
    x = Flatten()(bottleneck_output)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    classification_output = Dense(num_classes, activation='softmax')(x)

    classifier = Model(autoencoder.input, classification_output)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier