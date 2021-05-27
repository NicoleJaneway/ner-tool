import pickle
import ktrain
from keras.models import model_from_json

def load_model():
    # Load model json file
    json_file = open('model.json', 'r')

    # Load Ktrain preproc file
    features = pickle.load(open('preproc.sav', 'rb'))

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    print("Model Loaded from disk")

    # compile and evaluate loaded model
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model, features

def get_predictions(loaded_model,features,text):
    predictor = ktrain.get_predictor(loaded_model, features)
    return predictor.predict(text, return_proba=True)

