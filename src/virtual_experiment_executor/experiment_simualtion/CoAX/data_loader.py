import random
import pandas as pd
from collections import defaultdict

class AIDatasetLoader:
    def __init__(self, feature_values_df, metadata_df, explanation_values_df=None, explanation_columns=None):
        """
        Loads and handles AI-related data, including feature values, metadata, and (optionally) explanation values.

        Parameters:
            feature_values_df (pd.DataFrame): Feature values DataFrame.
            metadata_df (pd.DataFrame): Metadata DataFrame containing min-max scaling boundaries.
            explanation_values_df (pd.DataFrame): Optional, explanation values DataFrame.
            explanation_columns (list): List of column names from the explanation DataFrame to include.
        """
        self.feature_values_df = feature_values_df
        self.metadata_df = metadata_df
        self.explanation_values_df = explanation_values_df
        self.explanation_columns = explanation_columns or []

    def scale_value(self, value, min_val, max_val):
        if value < min_val:
            return 0
        elif value > max_val:
            return 1
        else:
            return (value - min_val) / (max_val - min_val)

    def scale_feature_values(self, instance_ids):
        """
        Scales feature values for the given instance IDs based on metadata min-max boundaries.

        Parameters:
            instance_ids (list): List of instance IDs to scale feature values for.

        Returns:
            list: List of lists, where each inner list contains the scaled feature values for one instance.
        """
        scaled_features = []

        for instance_id in instance_ids:
            # Fetch the row from feature_values_df for this instance
            feature_row = self.feature_values_df[self.feature_values_df['instanceId'] == instance_id].iloc[0]
            data_id = feature_row['dataId']

            # Retrieve metadata for the corresponding dataset
            data_metadata = self.metadata_df[self.metadata_df['dataId'] == data_id]
            if data_metadata.empty:
                raise ValueError(f"No metadata found for dataId: {data_id}")

            # Prepare to dynamically iterate over all possible features
            scaled_row = []
            i = 0

            while True:
                val_col = f'v{i}'
                min_col = f'v{i}_min'
                max_col = f'v{i}_max'

                # Stop if the feature value column does not exist
                if val_col not in feature_row.index or pd.isna(feature_row[val_col]):
                    break

                # Retrieve min and max values if columns exist, otherwise set to None
                min_val = data_metadata[min_col].values[0] if min_col in data_metadata.columns else None
                max_val = data_metadata[max_col].values[0] if max_col in data_metadata.columns else None

                # Scale the value if min and max are defined; otherwise, use the raw value
                if pd.isna(min_val) and pd.isna(max_val):
                    scaled_row.append(feature_row[val_col])
                else:
                    value = feature_row[val_col]
                    scaled_row.append(self.scale_value(value, min_val, max_val))

                i += 1

            # Append the scaled row to the list of scaled features
            scaled_features.append(scaled_row)

        return scaled_features


    def load_instances(self, selected_ids):
        """
        Load and scale feature values for the given instance IDs, along with AI predictions and explanation values.

        Parameters:
            selected_ids (list): List of instance IDs to load.

        Returns:
            tuple: 
                - List of lists of scaled feature values (one list per instance).
                - List of AI predictions (if available).
                - List of lists of selected explanation values (if available).
        """
        if not selected_ids:
            raise ValueError("selected_ids must be provided and cannot be empty.")

        # Scale feature values for the selected instances
        scaled_features = self.scale_feature_values(selected_ids)

        # Retrieve predictions and explanation rows
        AI_predictions = []
        explanation_values = []
        if self.explanation_values_df is not None:
            for instance_id in selected_ids:
                exp_row = self.explanation_values_df[self.explanation_values_df['instanceId'] == instance_id]
                if not exp_row.empty:
                    AI_predictions.append(exp_row.iloc[0]['pred'])


                    # Normalize by i_max if present
                    i_max = exp_row['i_max'].iloc[0] if 'i_max' in exp_row.columns else 1.0
                    row_vals = []
                    for col in self.explanation_columns:
                        if col in exp_row.columns:
                            val = exp_row[col].iloc[0]
                            row_vals.append(val / i_max if pd.notna(val) and i_max != 0 else None)
                        else:
                            row_vals.append(None)
                    explanation_values.append(row_vals)


                    # Extract only the selected explanation columns
                    # explanation_row = [exp_row.iloc[0][col] for col in self.explanation_columns if col in exp_row.columns]
                    # explanation_values.append(explanation_row)
                    # print("Original:", explanation_row)
                else:
                    print(f"Warning: No prediction found for instanceId {instance_id}")
                    print(self.explanation_values_df)
                    import sys
                    sys.exit()
                    AI_predictions.append(None)
                    explanation_values.append([None] * len(self.explanation_columns))
        else:
            AI_predictions = [None] * len(selected_ids)
            explanation_values = [[None] * len(self.explanation_columns) for _ in selected_ids]

        


        return scaled_features, AI_predictions, explanation_values

    def load_random_instances(self, num_random_ids):
        """
        Select a random set of instance IDs and load their feature values/predictions.

        Parameters:
            num_random_ids (int): Number of random instance IDs to select.

        Returns:
            tuple: (scaled_features, AI_predictions, explanation_values)
        """
        available_instance_ids = self.feature_values_df['instanceId'].unique()
        if len(available_instance_ids) < num_random_ids:
            raise ValueError(f"Not enough instance IDs to select {num_random_ids} random IDs.")

        selected_ids = random.sample(list(available_instance_ids), num_random_ids)
        return self.load_instances(selected_ids)

    def filter_loader(self, condition):
        """
        Filters the dataset based on a condition and returns a new AIDatasetLoader instance.

        Parameters:
            condition (function): A function that takes a DataFrame row and returns True/False.

        Returns:
            AIDatasetLoader: A new loader with filtered data.
        """
        # Filter feature values
        filtered_feature_values_df = self.feature_values_df[condition(self.feature_values_df)]
        
        # Filter metadata based on the filtered dataIds
        relevant_data_ids = filtered_feature_values_df['dataId'].unique()
        filtered_metadata_df = self.metadata_df[self.metadata_df['dataId'].isin(relevant_data_ids)]
        
        # Filter explanation values if available
        if self.explanation_values_df is not None:
            filtered_explanation_values_df = self.explanation_values_df[condition(self.explanation_values_df)]
        else:
            filtered_explanation_values_df = None

        return AIDatasetLoader(
            feature_values_df=filtered_feature_values_df,
            metadata_df=filtered_metadata_df,
            explanation_values_df=filtered_explanation_values_df,
            explanation_columns=self.explanation_columns
        )

class ParticipantDatasetLoader:
    def __init__(self, df):
        """
        Loads participant responses from a DataFrame.
        """
        self.df = df
        self.participant_responses = self._extract_participant_responses()

    def _extract_participant_responses(self):
        """
        Converts the raw DataFrame into a dict of participant -> list of response dicts.
        """
        participant_dict = defaultdict(list)

        for _, row in self.df.iterrows():
            participant_id = row['Participant ID']
            response_data = {
                'dataId': row['dataId'],
                'Trial Index': row['Trial Index'],
                'Instance Index': row['Instance Index'],
                'Step': row['Step'],  # Updated to use "step"
                'Response': row.get('Response', None),  # May be empty for feedback
                'Correct': row.get('Correct', None),
                'Tested w/ XAI': row['Tested w/ XAI'],
                'XAIType': row['XAIType'],
                'Timing': row['Timing'],
                'AI Prediction': row['AI Prediction'],
                'trialType': row.get('trialType', None)

            }
            participant_dict[participant_id].append(response_data)

        return dict(participant_dict)

    def get_responses_for_participant(self, participant_id=None):
        """
        If participant_id is provided, returns that participant's responses,
        otherwise returns all responses from all participants (list of dicts).
        """
        if participant_id:
            return self.participant_responses.get(participant_id, [])
        else:
            all_resps = []
            for _, resp_list in self.participant_responses.items():
                all_resps.extend(resp_list)
            return all_resps

    def list_all_participants(self):
        """
        Retrieve the list of all participant IDs.
        """
        return list(self.participant_responses.keys())

    def get_trial_responses(self, participant_id, trial_index):
        """
        Retrieve responses for a specific participant and trial index.
        """
        responses = self.get_responses_for_participant(participant_id)
        return [resp for resp in responses if resp['Trial Index'] == trial_index]

    def filter_loader(self, condition):
        """
        Returns a new ParticipantDatasetLoader with filtered data based on 'condition',
        which is a function that takes a response dict and returns True/False.
        """
        filtered_data = []
        for participant_id, responses in self.participant_responses.items():
            for response in responses:
                if condition(response):
                    filtered_data.append({
                        'Participant ID': participant_id,
                        **response
                    })

        new_df = pd.DataFrame(filtered_data)

        return ParticipantDatasetLoader(new_df)
