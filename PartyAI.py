import numpy as np

vodka = 1.0
rain = 1.0
friend = 1.0

def activation_funciton(x):
    if x>= 0.5:
        return 1
    else:
        return 0

def predict(vodka, rain, friend):
    inputs = ([vodka, rain, friend])
    weigth_input_to_hidden1 = [0.25, 0.25, 0]
    weigth_input_to_hidden2 = [0.5, -0.4, 0.9]
    weigth_input_to_hidden = np.array([weigth_input_to_hidden1, weigth_input_to_hidden2])

    weight_hidden_to_output = np.array([-1, 1])

    hidden_input = np.dot(weigth_input_to_hidden, inputs)
    hidden_output = np.array([activation_funciton(x) for x in hidden_input])

    output = np.dot(weight_hidden_to_output, hidden_output)
    return activation_funciton(output) >=0.25

print("result: " + str(predict(vodka, rain, friend)))
input()