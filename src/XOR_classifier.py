import tensorflow as tf

def xor_classifier_example(): 
    input_data = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32) # XOR 연산 입력 데이터
    xor_labels = tf.constant([0, 1, 1, 0], dtype=tf.float32) # XOR 연산 레이블

    batch_size = 1 # 미니 배치 크기
    epochs = 1000 # 학습 에포크 수

    # 입력 레이어 정의: 2개의 입력 노드를 가진 입력 레이어 정의
    input_layer = tf.keras.Input(shape=[2]) 
    
    # 은닉 레이어 정의:  2개의 유닛과 시그모이드 활성화 함수를 가진 Dense 레이어를 은닉 레이어로 정의
    hidden_layer = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.sigmoid, use_bias=True)(input_layer) 

    # 출력 레이어 정의:  # 1개의 유닛과 시그모이드 활성화 함수를 가진 Dense 레이어를 출력 레이어로 정의
    output = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid, use_bias=True)(hidden_layer) # 2개의 유닛과 시그모이드 활성화 함수를 가진 Dense 

    # 옵티마이저 및 모델 컴파일
    sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=sgd, loss="mse")

    # 모델 학습
    model.fit(x=input_data, y=xor_labels, batch_size=batch_size, epochs=epochs)

    # 예측 결과 출력
    prediction = model.predict(x=input_data, batch_size=batch_size)
    input_and_result = zip(input_data, prediction)
    print("====== XOR classifier result =====")
    for x, y in input_and_result:
        if y > 0.5:
            print("%d XOR %d => %.2f => 1" % (x[0], x[1], y))
        else:
            print("%d XOR %d => %.2f => 0" % (x[0], x[1], y))


if __name__ == '__main__':
    xor_classifier_example()
