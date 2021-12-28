import tensorflow as tf
import numpy as np

def predict_with_model(model,img_path):

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image,channels = 3)
    image = tf.image.convert_image_dtype(image,dtype = 'float32')
    image = tf.image.resize(image,[60,60]) #60,60,3
    image = tf.expand_dims(image,axis = 0) #1,60,60,3

    predictions = model.predict(image)
    prediction_dict = {
        0: 0,
        1: 1,
        2: 10,
        3: 11,
        4: 12,
        5: 13,
        6: 14,
        7: 15,
        8: 16,
        9: 17,
        10: 18,
        11: 19,
        12: 2,
        13: 20,
        14: 21,
        15: 22,
        16: 23,
        17: 24,
        18: 25,
        19: 26,
        20: 27,
        21: 28,
        22: 29,
        23: 3,
        24: 30,
        25: 31,
        26: 32,
        27: 33,
        28: 34,
        29: 35,
        30: 36,
        31: 37,
        32: 38,
        33: 39,
        34: 4,
        35: 40,
        36: 41,
        37: 42,
        38: 5,
        39: 6,
        40: 7,
        41: 8,
        42: 9,
    }

    predictions = prediction_dict[np.argmax(predictions)]

    return predictions

if __name__ == '__main__':
    #img_path = r""

    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)

    print(f"prediction = {prediction}")