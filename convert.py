import tensorflow as tf

# Load the Kaggle-trained .h5 model
model = tf.keras.models.load_model(
    "mask_detector.h5",
    compile=False
)

# Save as TensorFlow SavedModel
model.save("mask_detector_tf")

print("Conversion done ✅")
