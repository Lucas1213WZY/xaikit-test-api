class UI:
    def __init__(self):
        """
        Pseudo-UI class that keeps track of what's displayed.
        """
        self.displayed_feature_values = None
        self.displayed_explanation = None
        self.displayed_prediction = None


        self.time = None

    def display(self, feature_values, explanation=None, ai_prediction=None):
        """
        Simulate showing something on the UI.
        """
        self.displayed_feature_values = feature_values
        self.displayed_explanation = explanation
        self.displayed_prediction = ai_prediction

        # may wish to advance time here if we want to incorporate some delays but no need for now

    def get_value(self, key):
        """
        Retrieve a piece of data from the displayed content, e.g. 'prediction'.
        """
        if key == 'ai_prediction':
            return self.displayed_prediction
        elif key == 'features':
            return self.displayed_feature_values
        elif key == 'explanation':
            return self.displayed_explanation
        return None
