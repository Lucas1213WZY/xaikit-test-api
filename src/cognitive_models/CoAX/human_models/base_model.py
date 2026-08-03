class BaseModel:
    def __init__(self):
        """
        A placeholder for the 'human model', which simulates how a user might respond.
        You can expand this with more sophisticated logic if needed.
        """
        pass

    def new_instance(self):
        """
        Called to inform the model that the next presented information
        will be for a new instance.
        """
        pass

    def infer(self, ui):
        """
        Called when the model is expected to make an inference or response 
        based on what's currently displayed in the UI.
        
        Returns:
            - response (dict, int, str, etc.): The response of the 'human'.
            - time (float): How much time is added or used in this step (for the experiment runner).
        """
        # Example dummy response and 0.5 time increment
        return {0: 0.5, 1: 0.5}, 0.5

    def feedback(self, ui):
        """
        Called when additional info (explanations, AI predictions) is shown.
        Returns a time increment or some other measure if needed.
        
        Returns:
            - response (dict, int, str): Possibly an updated response 
            - time (float): Additional time used in the feedback step
        """
        return 0.5



# Need to create some kind of generic AI model that covers non human-specific models to accept the kind of information that is being received as a wrapper
# i.e., when doing the infer and so on, it stores the information and then uses that information later on during the feedback phase.
# also, probably have one model for the w/ XAI part and one for the w/o XAI part