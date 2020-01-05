import pickle
from processing.processing import PreProcessing
from sklearn.pipeline import Pipeline


def load_trained_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
        return model


if __name__ == "__main__":
    model = load_trained_model("hatespeech_detection.model")
    if model is not None:
        print("*** Welcome to the Hatespeech & Offensive Language Detection Terminal ***")
        print("/!\ To exit the terminal: enter quit() as a message.", end="\n\n")
        while True:
            tweet = input("> Write a Tweet or a Normal Message:\n")
            if tweet.strip() == "quit()":
                break
            tweet = PreProcessing.sanitize_tweet(tweet)
            prediction = model.predict([tweet])
            print("-> Your message has been classified as {}".format(
                prediction[0]), end="\n\n")
    else:
        print("Could not load the trained model. Check if the file exists. If it's not, run the Jupyter notebook.")
