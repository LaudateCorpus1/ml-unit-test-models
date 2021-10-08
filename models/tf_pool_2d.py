import numpy as np
import tensorflow as tf

ip_tensor_8x8 =  np.array([[-0.8520309,-0.8490718, 1.129514,-1.3910682,-1.3713549,-1.2450725,-0.64018744,    -0.58683217],
                           [0.61351526,-0.7573147,-1.0944875,-1.3194267,-1.0564152,-1.0523492,-0.8647979,    -0.5105319],
                           [-1.0064573,-0.96857584,-1.049313,-1.1837159,-1.421839,-1.3091717,-1.4398289,     -1.1118798],
                           [-1.2060001,-0.9072193,-0.93204916,-0.9201482,-1.0511543,-0.8700992,-1.0922338,   -1.2065634],
                           [-0.9080574,-1.2397976,-1.0792112,-1.1603603,-1.4759526,-1.6273732,-1.7018712,    -1.2549796],
                           [-0.9400331,-1.2271551,-1.0533371,-1.7574923,-1.4672713,-1.449718,-0.89420605,    -0.7849926],
                           [-0.75937706,-1.0642793,-1.3776617,-1.3407046,-0.70763135,-0.8301665,-0.9863248,  -0.8487809],
                           [-0.45886853,-0.6476302,-0.7841558,-1.0839088,-0.94306296,-0.99122596,-0.7741117, -0.7921684]])

ip_tensor_7x7 =  np.array([[-0.8520309,-0.8490718,1.129514,-1.3910682,-1.3713549,-1.2450725,-0.64018744,    ],
                           [0.61351526,-0.7573147,-1.0944875,-1.3194267,-1.0564152,-1.0523492,-0.8647979,    ],
                           [-1.0064573,-0.96857584,-1.049313,-1.1837159,-1.421839,-1.3091717,-1.4398289,     ],
                           [-1.2060001,-0.9072193,-0.93204916,-0.9201482,-1.0511543,-0.8700992,-1.0922338,   ],
                           [-0.9080574,-1.2397976,-1.0792112,-1.1603603,-1.4759526,-1.6273732,-1.7018712,    ],
                           [-0.9400331,-1.2271551,-1.0533371,-1.7574923,-1.4672713,-1.449718,-0.89420605,    ],
                           [-0.75937706,-1.0642793,-1.3776617,-1.3407046,-0.70763135,-0.8301665,-0.9863248,  ]])

ip_tensor = ip_tensor_7x7
width  = 7
height = 7

ip_tensor = np.expand_dims(np.expand_dims(ip_tensor,0),3)
model_max_pool = tf.keras.Sequential(
    tf.keras.layers.MaxPooling2D(
          pool_size=[3, 3],
          strides=[2, 2],
          padding='SAME',
          data_format='channels_last'))
model_max_pool.build(input_shape=(1,width,height,1))
op_tensor_max_pool = model_max_pool(ip_tensor).numpy()

model_avg_pool = tf.keras.Sequential(tf.keras.layers.AveragePooling2D(
    pool_size=[3, 3],
    strides=[2, 2],
    padding='SAME',
    data_format='channels_last'))
model_avg_pool.build(input_shape=(1,width,height,1))
op_tensor_avg_pool = model_avg_pool(ip_tensor).numpy()
print(np.squeeze(op_tensor_max_pool))
print(np.squeeze(op_tensor_avg_pool))

#tflite_conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model_max_pool)
converter.allow_custom_ops = True
tflite_maxpool_model = converter.convert()
with open("maxpool.tflite", "wb") as f:
    f.write(tflite_maxpool_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model_avg_pool)
converter.allow_custom_ops = True
tflite_avgpool_model = converter.convert()
with open("avgpool.tflite", "wb") as f:
    f.write(tflite_avgpool_model)

interpreter    = tf.lite.Interpreter(model_path="maxpool.tflite")
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], np.float32(ip_tensor))
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print("tflite_output \n", np.squeeze(output))
