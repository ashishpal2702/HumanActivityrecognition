from src.utils.helper import save_weights, load_weights


class Predict:
    def __init__(self, threshold):
        self.threshold = threshold

    def load_model(self, model_path):
        model = load_weights(model_path)
        return model

    def model_predict(self, model, x):
        predicted_proba = model.predict_proba(x)
        y_pred = (predicted_proba[:, 1] >= self.threshold).astype("int")
        # y_pred = model.predict(x)
        return y_pred
