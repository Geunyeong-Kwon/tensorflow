import tensorflow as tf

def and_classifier_example():
    input_data = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]]) # tf.constant 함수는 상수 구조체 tensor를 만드는 함수
                                                               # array: 1차원 배열, matrix: 2차원 배열, tensor: 3차원 배열
                                                               # tensor라는 이름의 자료구조 만들어줌(4x2의 matirx)
                                                               # 입력 데이터 버튼 4개
    input_data = tf.cast(input_data, tf.float32) # 데이터 타입 바꿔줌. 여기선 float 32(float64는 메모리를 float32보다 많이 사용, but 정확함)
                                                 # neural netowrk는 저장하려면 float32를 사용하면 데이터 용량 반으로 줄어듦.

    and_labels = tf.constant([0, 0, 0, 1]) # 각각의 정답
    and_labels = tf.cast(and_labels, tf.float32)

    batch_size = 1 # mini batch
    epochs = 1000



    ######### SLP - Begin
    input_layer = tf.keras.Input(shape=[2, ]) # keras.Input은 input layer 정의하는데 사용, input layer의 node가 2개임
    output = tf.keras.layers.Dense(units=1, # Dense 클래스: neurl network data 들어있는 layer 한층 만들어줌
                                   activation=tf.keras.activations.sigmoid, # units: layer 한층에 들어가는 node 개수 설정, activations: sigmoid함수 쓰겠다
                                   use_bias=True) (input_layer) # use_bias: bias까지 쓸거냐 안쓰고 0으로 둘 거냐, use_bias=True까지는 input layer과 output layer 만들어져만 있음
                                                                # input_layer 통해 input layer과 output layer 연결해서 singleregular perceptron 하나 만들어짐
    sgd = tf.keras.optimizers.SGD(learning_rate=0.1) 
    slp_classifier = tf.keras.Model(inputs=input_layer, outputs=output) # Model 클래스: 내가 만든 neural netowrk 등록하고 어떤 알고리즘으로 학습할 지 등록
                                                                        # 파라미터: input이 먼지, 최종적인 network의 형태 output에 등록 
    slp_classifier.compile(optimizer=sgd, loss="mse") #compile 함수: network 학습할 때, loss: object function 등록

    slp_classifier.fit(x=input_data, y=and_labels, batch_size=batch_size, epochs=epochs) # fit: training 함수 --> data 제공하면 학습 가능, 
                                                # x: training data, y: 정답, batch_size: batch학습 하기위한 size, 
                                                # epochs:몇 바퀴 돌릴 것인지
    ######### SLP - End


    ######### MLP - Begin
    input_layer2 = tf.keras.Input(shape=[2, ])
    hidden_layer2 = tf.keras.layers.Dense(units=4,
                                   activation=tf.keras.activations.sigmoid,
                                   use_bias=True)(input_layer2)
    output2 = tf.keras.layers.Dense(units=1,
                                   activation=tf.keras.activations.sigmoid,
                                   use_bias=True)(hidden_layer2)
    mlp_classifier = tf.keras.Model(inputs=input_layer2, outputs=output2)
    sgd2 = tf.keras.optimizers.SGD(learning_rate=0.1)
    mlp_classifier.compile(optimizer=sgd2, loss="mse")
    mlp_classifier.fit(x=input_data, y=and_labels, batch_size=batch_size, epochs=epochs)
    ######### MLP - End


    ######## SLP AND prediciton
    prediction = slp_classifier.predict(x=input_data, batch_size=batch_size)
    input_and_result = zip(input_data, prediction) # 입력 데이터와 예측 결과 함께 출력
    print("====== SLP AND classifier result =====")
    for x, y in input_and_result:
        if y > 0.5:
            print("%d AND %d => %.2f => 1" % (x[0], x[1], y))
        else:
            print("%d AND %d => %.2f => 0" % (x[0], x[1], y))

    ######## MLP AND prediciton
    prediction = mlp_classifier.predict(x=input_data, batch_size=batch_size) # predict 함수: 입력 데이터에 대한 출력 예측
    input_and_result = zip(input_data, prediction)
    print("====== MLP AND classifier result =====")
    for x, y in input_and_result:
        if y > 0.5:
            print("%d AND %d => %.2f => 1" % (x[0], x[1], y))
        else:
            print("%d AND %d => %.2f => 0" % (x[0], x[1], y))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    and_classifier_example()

