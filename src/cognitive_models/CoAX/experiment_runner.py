from collections import defaultdict
import sys

class Time:
    def __init__(self, initial_time=0.0):
        self.current_time = initial_time

    def add_time(self, amount):
        """Add the specified amount to the current time."""
        self.current_time += amount

    def reset(self):
        """Reset the current time to zero."""
        self.current_time = 0.0

    def get_time(self):
        """Return the current accumulated time."""
        return self.current_time

class GenericExperimentRunner:
    """
    A generic runner that executes inference (and optional feedback)
    for a set of instance IDs, either with or without explanations.

    This class does NOT rely on participant dataset/metadata, so it
    can be used for 'offline' or 'generic' runs for specific instance IDs.
    """

    def __init__(self, human_model, ai_dataset_loader, ui):
        """
        Args:
            human_model: The 'human' logic object (or model).
            ai_dataset_loader: AIDatasetLoader instance for feature/explanation data.
            ui: UI instance that displays the data.
        """
        self.human_model = human_model
        self.ai_dataset_loader = ai_dataset_loader
        self.ui = ui
        
        # A shared time object if needed
        self.time = Time()
        self.human_model.time = self.time

        # For storing results if you want to output them at the end
        self.experiment_log = []

    def run_experiment(self, instance_ids, with_explanation=False, do_feedback=False):
        """
        Runs the experiment for a list of instance IDs. Can optionally display explanations
        and optionally collect feedback from the human_model.

        Args:
            instance_ids (list[int]): The list of instance IDs to run inference on.
            with_explanation (bool): If True, display explanations along with features.
            do_feedback (bool): If True, collect feedback step after inference.
        """

        for instance_id in instance_ids:
            # 1) Reset or initialize the human model for a new inference instance
            self.human_model.new_instance()

            # 2) Retrieve the scaled feature values, AI prediction, and explanation
            feature_values, ai_prediction, explanation_row = self._get_ai_data(instance_id)

            # 3) Display to the UI (with or without explanation, depending on the flag)
            if with_explanation:
                print("Displaying with explanation")
                self.ui.display(
                    feature_values=feature_values,
                    explanation=explanation_row,
                    ai_prediction=ai_prediction
                )
            else:
                self.ui.display(
                    feature_values=feature_values,
                    ai_prediction=ai_prediction
                )

            # 4) Let the 'human model' infer
            response, time_used = self.human_model.infer(ui=self.ui)


            # 5) Log the inference event
            log_entry = {
                "instance_id": instance_id,
                "step": "infer",
                "feature_values": feature_values,
                "explanation": explanation_row if with_explanation else None,
                "response": response,
                "ai_prediction": ai_prediction,
                "time_used": time_used,
                "accumulated_time": self.time.get_time()
            }
            self._log_event(log_entry)

            # 6) Optionally do a feedback step (e.g., in a 'Training' context)
            if do_feedback:
                # Possibly re-display the same info or only the needed info
                if with_explanation:
                    self.ui.display(
                        feature_values=feature_values,
                        explanation=explanation_row,
                        ai_prediction=ai_prediction
                    )
                else:
                    self.ui.display(
                        feature_values=feature_values,
                        ai_prediction=ai_prediction
                    )

                time_used_feedback = self.human_model.feedback(ui=self.ui)

                log_entry_feedback = {
                    "instance_id": instance_id,
                    "step": "feedback",
                    "feature_values": feature_values,
                    "explanation": explanation_row if with_explanation else None,
                    "ai_prediction": ai_prediction,
                    "time_used": time_used_feedback,
                    "accumulated_time": self.time.get_time()
                }
                self._log_event(log_entry_feedback)

        return self.experiment_log


    def _get_ai_data(self, instance_id):
        """
        Retrieves scaled features, AI prediction, and explanation row
        from AIDatasetLoader.
        """
        scaled_features_list, ai_predictions_list, explanation_rows_list = \
            self.ai_dataset_loader.load_instances([instance_id])

        # The returned items should be lists of length 1
        feature_values = scaled_features_list[0]
        ai_prediction = ai_predictions_list[0]
        explanation_row = explanation_rows_list[0]
        return feature_values, ai_prediction, explanation_row

    def _log_event(self, log_entry):
        """
        Logs (or stores) an event in the experiment.

        Args:
            log_entry (dict): A dictionary containing all details of the event to log.
        """
        self.experiment_log.append(log_entry)




class ParticipantExperimentRunner(GenericExperimentRunner):
    """
    This class orchestrates the experiment procedure for a single participant (or multiple),
    building upon the generic runner.
    """

    def __init__(self, human_model, ai_dataset_loader, participant_dataset_loader, ui):
        super().__init__(human_model, ai_dataset_loader, ui)
        self.participant_dataset_loader = participant_dataset_loader

    def run_experiment(self, participant_id):
        """
        Runs the experiment for a given participant, step by step, using
        the participant dataset info.
        """
        responses = self.participant_dataset_loader.get_responses_for_participant(participant_id)
        trials_by_index = self.group_by_trial_index(responses)

        # Iterate in ascending order of trial index
        sorted_trial_indexes = sorted(trials_by_index.keys())

        # print(responses)
        # sys.exit()

        # scam method for now
        # sorted_trial_indexes = sorted_trial_indexes[:56]

        for trial_index in sorted_trial_indexes:
            trial_rows = trials_by_index[trial_index]

            # Separate infer and feedback steps
            infer_rows = [row for row in trial_rows if row["Step"].lower() == "infer"]
            feedback_rows = [row for row in trial_rows if row["Step"].lower() == "feedback"]

            # Process inference step first
            for row in infer_rows:
                instance_id = row["Instance Index"]

                # Retrieve the scaled feature values, AI prediction, and explanation
                feature_values, ai_prediction, explanation_row = self._get_ai_data(instance_id)

                # Display with or without explanation
                if row["Tested w/ XAI"] == "w/ XAI":
                    self.ui.display(
                        feature_values=feature_values,
                        explanation=explanation_row,
                        ai_prediction=ai_prediction
                    )
                else:
                    self.ui.display(
                        feature_values=feature_values,
                        ai_prediction=ai_prediction
                    )

                # Human model inference
                response, time_used = self.human_model.infer(ui=self.ui)

                # Log the inference event
                log_entry = {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "instance_index": instance_id,
                    "Step": "infer",
                    "trialType": row.get("trialType", None),
                    "tested_w_xai": row["Tested w/ XAI"],
                    "feature_values": feature_values,
                    "explanation": explanation_row if row["Tested w/ XAI"] == "w/ Explanation" else None,
                    "response": response,
                    "participant_response": row["Response"],
                    "ai_prediction": ai_prediction,
                    "time_used": time_used,
                    "accumulated_time": self.human_model.time.get_time()
                }
                self._log_event(log_entry)

            # Process feedback step (if available)
            for row in feedback_rows:
                instance_id = row["Instance Index"]
                feature_values, ai_prediction, explanation_row = self._get_ai_data(instance_id)

                # Display feedback
                if explanation_row is not None:
                    self.ui.display(
                        feature_values=feature_values,
                        explanation=explanation_row,
                        ai_prediction=ai_prediction
                    )
                else:
                    self.ui.display(
                        feature_values=feature_values,
                        ai_prediction=ai_prediction
                    )



                time_used = self.human_model.feedback(ui=self.ui)

                # Log the feedback event
                log_entry = {
                    "participant_id": participant_id,
                    "trial_index": trial_index,
                    "instance_index": instance_id,
                    "Step": "feedback",
                    "trialType": row.get("trialType", None),
                    "tested_w_xai": row["Tested w/ XAI"],
                    "feature_values": feature_values,
                    "explanation": explanation_row,
                    "ai_prediction": ai_prediction,
                    "response": None,  # Feedback step does not produce a response
                    "time_used": time_used,
                    "accumulated_time": self.human_model.time.get_time()
                }
                self._log_event(log_entry)

        return self.experiment_log

    def group_by_trial_index(self, responses):
        """
        Utility method to group responses by 'Trial Index'.
        """
        trials = defaultdict(list)
        for resp in responses:
            trials[resp['Trial Index']].append(resp)
        return trials


class StrategyComparisonRunner:
    """
    Runs inference (with/without explanation) and optional feedback 
    based on a split index:
      - For i < split_idx (training portion): 
           1) Inference (no explanation)
           2) Inference (with explanation) if with_explanation=True
           3) Feedback
      - For i >= split_idx (test portion):
           1) Inference (no explanation)
           2) Inference (with explanation) if with_explanation=True
      Logs all events.
    """

    def __init__(self, human_model, ai_loader, ui):
        """
        Args:
            human_model: A 'human model' or strategy object 
                (e.g., TopKExplanationStrategy(...) or AttributionSummingStrategy(...)).
            ai_loader: AIDatasetLoader for retrieving feature values/predictions/explanations.
            ui: UI object for display or user interaction (can be a mock if no real UI).
        """
        self.human_model = human_model
        self.ai_loader = ai_loader
        self.ui = ui

        # For storing logs
        self.experiment_log = []
        # Optional: track time if your strategy uses it
        self.time = Time()
        self.human_model.time = self.time

    def run_experiment(self, instance_ids, split_idx, with_explanation=False):
        """
        Runs inference on the given instance_ids with a 'training' portion 
        (first split_idx) and a 'test' portion (from split_idx onward).

        Args:
            instance_ids (list): The instance IDs to run.
            split_idx (int): The boundary between training and testing portion.
            with_explanation (bool): If True, run an additional inference 
                                     with explanation (and log it).
        """
        for i, instance_id in enumerate(instance_ids):
            # Indicate if we are in training or testing
            is_training = (i < split_idx)

            # Always reset or start a 'new' inference for each instance
            self.human_model.new_instance()

            # Retrieve scaled features, AI prediction, explanation data
            feature_values, ai_prediction, explanation_row = self._get_ai_data(instance_id)

            # --- 1) Inference WITHOUT explanation ---
            response_no_expl, time_used_no_expl = self._infer_no_explanation(
                instance_id, feature_values, ai_prediction
            )
            # Log
            self._log_event(
                instance_id=instance_id,
                step="infer",
                feature_values=feature_values,
                explanation=None,  # no explanation used
                response=response_no_expl,
                ai_prediction=ai_prediction,
                time_used=time_used_no_expl
            )

            # --- 2) Inference WITH explanation (if requested) ---
            if with_explanation:
                response_w_expl, time_used_w_expl = self._infer_with_explanation(
                    instance_id, feature_values, ai_prediction, explanation_row
                )
                # Log
                self._log_event(
                    instance_id=instance_id,
                    step="infer",
                    feature_values=feature_values,
                    explanation=explanation_row,
                    response=response_w_expl,
                    ai_prediction=ai_prediction,
                    time_used=time_used_w_expl
                )

            # --- 3) Feedback (only if training portion) ---
            if is_training:
                # Typically you'd re-display or confirm AI prediction, etc.
                time_used_fb = self._feedback(
                    instance_id, feature_values, ai_prediction, 
                    explanation_row if with_explanation else None
                )
                self._log_event(
                    instance_id=instance_id,
                    step="feedback",
                    feature_values=feature_values,
                    explanation=explanation_row if with_explanation else None,
                    response=None,  # feedback typically doesn't produce a direct "prediction"
                    ai_prediction=ai_prediction,
                    time_used=time_used_fb
                )



        # print("Memory", len(self.human_model.memory.exemplars), self.human_model.memory.exemplars[:3])

        return self.experiment_log


    def generalized_run_experiment(self, trial_sequence):
        """
        Runs a list of trials, where each trial in 'trial_sequence' is a dict with keys:
          - instance_id (int or str)
          - is_training (bool)
          - with_explanation (bool)
        
        The order in which the trials appear in 'trial_sequence' is the order they are run.
        """
        for trial in trial_sequence:
            instance_id = trial["instance_id"]
            is_training = trial["is_training"]
            with_explanation = trial["with_explanation"]


            # Always start fresh for each new instance
            self.human_model.new_instance()

            # Retrieve scaled features, AI prediction, explanation data
            feature_values, ai_prediction, explanation_row = self._get_ai_data(instance_id)

            # --- 1) Inference WITHOUT explanation ---
            if (not with_explanation) or is_training:
                response_no_expl, time_used_no_expl = self._infer_no_explanation(
                    instance_id, feature_values, ai_prediction
                )
                # Log
                self._log_event(
                    instance_id=instance_id,
                    step="infer",
                    feature_values=feature_values,
                    explanation=None,
                    response=response_no_expl,
                    ai_prediction=ai_prediction,
                    time_used=time_used_no_expl
                )

            # --- 2) Inference WITH explanation (if requested) ---
            if with_explanation:
                response_w_expl, time_used_w_expl = self._infer_with_explanation(
                    instance_id, feature_values, ai_prediction, explanation_row
                )
                # Log
                self._log_event(
                    instance_id=instance_id,
                    step="infer",
                    feature_values=feature_values,
                    explanation=explanation_row,
                    response=response_w_expl,
                    ai_prediction=ai_prediction,
                    time_used=time_used_w_expl
                )

            # --- 3) Feedback (only if is_training) ---
            if is_training:
                time_used_fb = self._feedback(
                    instance_id, feature_values, ai_prediction,
                    explanation_row
                )

                self._log_event(
                    instance_id=instance_id,
                    step="feedback",
                    feature_values=feature_values,
                    explanation=explanation_row if with_explanation else None,
                    response=None,
                    ai_prediction=ai_prediction,
                    time_used=time_used_fb
                )

        return self.experiment_log


    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _infer_no_explanation(self, instance_id, feature_values, ai_prediction):
        """Helper to do an inference step without showing explanation."""
        # Display
        self.ui.display(feature_values=feature_values, ai_prediction=ai_prediction)
        response, time_used = self.human_model.infer(ui=self.ui)
        return response, time_used

    def _infer_with_explanation(self, instance_id, feature_values, ai_prediction, explanation_row):
        """Helper to do an inference step with explanation."""
        self.ui.display(feature_values=feature_values, explanation=explanation_row, ai_prediction=ai_prediction)
        response, time_used = self.human_model.infer(ui=self.ui)
        return response, time_used

    def _feedback(self, instance_id, feature_values, ai_prediction, explanation_row):
        """Helper to run a feedback step."""
        # You might show the same or partial info
        if explanation_row is not None:
            self.ui.display(feature_values=feature_values, explanation=explanation_row, ai_prediction=ai_prediction)
        else:
            self.ui.display(feature_values=feature_values, ai_prediction=ai_prediction)
        time_used = self.human_model.feedback(ui=self.ui)
        return time_used

    def _get_ai_data(self, instance_id):
        """
        Retrieves scaled features, AI prediction, and explanation row
        from the AIDatasetLoader.
        """
        scaled_features_list, ai_preds_list, expl_rows_list = \
            self.ai_loader.load_instances([instance_id])

        feature_values = scaled_features_list[0]
        ai_prediction = ai_preds_list[0]
        explanation_row = expl_rows_list[0]  # e.g., list of explanation values
        return feature_values, ai_prediction, explanation_row

    def _log_event(self, instance_id, step, feature_values, explanation, response, ai_prediction, time_used):
        """Stores an event to the experiment_log."""
        log_entry = {
            "strategy_name": type(self.human_model).__name__,  # e.g. "TopKExplanationStrategy"
            "instance_id": instance_id,
            "Step": step,
            "feature_values": feature_values,
            "explanation": explanation,  # could be a list of numbers
            "response": response,
            "ai_prediction": ai_prediction,
            "time_used": time_used,
            "accumulated_time": self.time.get_time()
        }
        self.experiment_log.append(log_entry)
