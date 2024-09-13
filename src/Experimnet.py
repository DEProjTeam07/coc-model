import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. 실험 관리: 실험을 명시적으로 설정
experiment_name = "my_experiment"
mlflow.set_experiment(experiment_name)

# 데이터셋 생성 (예시용)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 정의
def create_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# 실험 관리 코드 함수
def run_experiment(epochs, batch_size):
    # 2. 실험 시작
    with mlflow.start_run(run_name=f"run_epochs_{epochs}_batch_{batch_size}"):
        # autolog 설정 (tensorflow에 맞게 자동으로 모든 기록)
        mlflow.tensorflow.autolog()

        # 모델 생성 및 학습
        model = create_model()
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_test, y_test))

        # 3. 파라미터, 메트릭, 아티팩트 기록
        # 파라미터 기록
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        # 커스텀 메트릭 기록
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)

        # 4. 모델 저장 (MLflow 텐서플로우 자동 저장)
        # mlflow.tensorflow.log_model(model, "model") -> autolog()에서 자동 기록

        print(f"Finished run: epochs={epochs}, batch_size={batch_size}")

# 5. 실험 실행
if __name__ == "__main__":
    # 여러 실험 실행
    run_experiment(epochs=5, batch_size=32)
    run_experiment(epochs=10, batch_size=64)
