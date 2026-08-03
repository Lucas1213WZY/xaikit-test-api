from collections import defaultdict
import sys
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