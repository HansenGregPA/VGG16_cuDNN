import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

model_path = 'trained_models/vgg_16.ckpt'

#首先，使用tensorflow自带的python打包库读取模型
model_reader = pywrap_tensorflow.NewCheckpointReader(r"trained_models/vgg_16.ckpt")

#然后，使reader变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()

#最后，循环打印输出
out_path = 'vgg16/'
for key in var_dict:
    print("variable name: ", key)
    parameters = model_reader.get_tensor(key)
    print(parameters)

    # parameters = np.asarray(parameters)

    parameter_name = ''
    temp = key.split('/')
    for i in range(len(temp)-1):
        parameter_name += temp[i]
        parameter_name += '-'
    parameter_name += temp[-1]
    file_name = out_path + parameter_name + '.npy'
    # file_handle = open(file_name, mode='w')
    # file_handle.writelines(parameters.astype(str))
    # file_handle.close()

    np.save(file_name,parameters)

print('Done.')