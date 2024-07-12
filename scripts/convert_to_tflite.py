import tensorflow as tf

# Convert model to be compatible with tensorflow lite

# Load the previously trained and saved model
model = tf.keras.models.load_model('../models/baybayin_to_english_translation_model.keras')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]

# Convert the model
tflite_model = converter.convert()

# Save the converted model to a file - In this case, 'baybayin_to_english_translation_model.tflite'
with open('../models/baybayin_to_english_translation_model.tflite', 'wb') as f:
    f.write(tflite_model)
