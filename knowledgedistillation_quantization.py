import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.datasets import cifar10
import tempfile
import os

IMG_SIZE=224
EPOCHS=10

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3, #temperature default : 1
    ):

        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

#No pre-trained model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import NASNetLarge

teacher = EfficientNetB7(weights=None, input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=True, classes=10) #teacher model
student = EfficientNetB0(weights=None, input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=True, classes=10) #student model

# Clone student for later comparison
student_scratch = keras.models.clone_model(student) #student only model

# Prepare the train and test dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#resize image
x_train = tf.image.resize(x_train,[224,224])
x_train= x_train.numpy()
x_test = tf.image.resize(x_test, [224,224])
x_test = x_test.numpy()
# Normalize data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

"""
## Train the teacher
In knowledge distillation we assume that the teacher is trained and fixed. Thus, we start
by training the teacher model on the training set in the usual way.
"""

# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate teacher on data.
teacher.fit(x_train, y_train, epochs=EPOCHS)
print("evaluate the teacher")
teacher.evaluate(x_test, y_test)
#teacher.save("teacher", save_format="h5")
#teacher.save_weights("teacher")
#tf.keras.models.save_model(teacher, "teacher", save_format="tf")

#model size comparison
float_converter=tf.lite.TFLiteConverter.from_keras_model(teacher)
float_tflite_model=float_converter.convert()

_, float_file = tempfile.mkstemp(".tflite")

with open(float_file,"wb") as f:
    f.write(float_tflite_model)

print("float model in Mb: ", os.path.getsize(float_file)/float(2**20))
"""
## Distill teacher to student
We have already trained the teacher model, and we only need to initialize a
`Distiller(student, teacher)` instance, `compile()` it with the desired losses,
hyperparameters and optimizer, and distill the teacher to the student.
"""

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10, #default : 1
)

# Distill teacher to student
distiller.fit(x_train, y_train, epochs=EPOCHS)
print("evaluate distill")
# Evaluate student on test dataset
distiller.evaluate(x_test, y_test)
#distiller.save("distiller", save_format="h5")
#distiller.save_weights("distiller")
#distiller.save(distiller, "distiller", save_format="tf")

#model size comparison
float_converter=tf.lite.TFLiteConverter.from_keras_model(distiller.student)
float_tflite_model=float_converter.convert()

_, float_file = tempfile.mkstemp(".tflite")

with open(float_file,"wb") as f:
    f.write(float_tflite_model)

print("float model in Mb: ", os.path.getsize(float_file)/float(2**20))

"""
## Train student from scratch for comparison
We can also train an equivalent student model from scratch without the teacher, in order
to evaluate the performance gain obtained by knowledge distillation.
"""

# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, y_train, epochs=EPOCHS)
print("evaluate student scratch ")
student_scratch.evaluate(x_test, y_test)
#student_scratch.save("student_scratch", save_format="h5")
#student_scratch.save_weights("student_scratch")
#tf.keras.models.save_model(student_scratch, "student_scratch", save_format="tf")

float_converter=tf.lite.TFLiteConverter.from_keras_model(student_scratch)
float_tflite_model=float_converter.convert()

_, float_file = tempfile.mkstemp(".tflite")

with open(float_file,"wb") as f:
    f.write(float_tflite_model)

print("float model in Mb: ", os.path.getsize(float_file)/float(2**20))
#QUNTIZATION
converter = tf.lite.TFLiteConverter.from_keras_model(distiller.student) #distillation 수행한 모델 불러옴
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open("distill_quant8.tflite","wb").write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='distill_quant8.tflite')
interpreter.allocate_tensors()

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in x_test:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == y_test[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy

print("evaluate quant")
print(evaluate_model(interpreter))

