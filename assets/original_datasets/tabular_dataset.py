import copy 
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
import sklearn
import numpy as np
import pandas as pd
import pickle
import os

class TabularDataset:
    def __init__(self, X, y,
            feature_names=None, categorical_feature_options=[], target_name=None, target_options=[],
            ordinal_feature_indices=[], feature_boundaries=[], dataset_name=None):
        
        '''
        Categorical features to refer to both nominal types (i.e., "US", "UK", etc...) as well as discretized variables which are now technically ordinal (e.g., "Age<28", "28<=Age<=45")
        Ordinal Features keeps track of ordinal types (as opposed to nominal types), because those may be interpreted differently by the user
        '''

        self.X = X
        self.y = y

        self.feature_names = feature_names # List containing the name of each attribute

        self.categorical_feature_options = {i:list(v) for i,v in categorical_feature_options.items()} # Dictionary of {index:[]} containing what are the possible values/options that each feature can take

        if ordinal_feature_indices is not None:
            assert(set(ordinal_feature_indices).issubset(set(self.categorical_feature_indices)))
        self.ordinal_feature_indices = ordinal_feature_indices


        # TO-CONFIRM : still open to the idea of having categorical_feature_indices as an individual input
        # TO-CONFIRM : do we want the dataset name to be stored as well?

        self.target_name = target_name
        self.target_options = target_options # List of strings containing the possible outputs

        self.feature_boundaries = feature_boundaries or self.calculate_boundaries()

        self.dataset_name = dataset_name


    def calculate_boundaries(self):
        # Calculate the 5th and 95th percentiles for each feature, assuming continuous unless specified
        boundaries = {}
        for i in range(self.X.shape[1]):
            if i not in self.categorical_feature_indices:
                # Use nanpercentile to ignore NaN values and get the 5th and 95th percentiles
                lower_bound = np.nanpercentile(self.X[:, i], 5)
                upper_bound = np.nanpercentile(self.X[:, i], 95)
                boundaries[i] = (lower_bound, upper_bound)
        return boundaries


    @property
    def categorical_feature_indices(self):
        if self.categorical_feature_options is not None:
            return list(self.categorical_feature_options.keys())
        return []

    def copy_with_modifications(self, **kwargs):
        new_params = {**self.__dict__, **kwargs}
        return TabularDataset(**new_params)

    def discretize(self):
        new_dataset = self.copy_with_modifications()

        # Create the discretizer, specifying the number of bins and strategy
        discretizer = preprocessing.KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')

        # Apply the discretizer only to continuous features.
        numeric_features = [i for i in range(self.X.shape[1]) if i not in self.categorical_feature_indices]
        new_dataset.X[:, numeric_features] = discretizer.fit_transform(self.X[:, numeric_features])


        # Update the categorical_feature_options and ordinal features
        # Generate detailed labels for bins
        ordinal_feature_options = {}
        for i, feature_idx in enumerate(numeric_features):
            edges = discretizer.bin_edges_[i]
            labels = []
            for j in range(len(edges) - 1):
                if j == 0:
                    label = f"{self.feature_names[feature_idx]} < {edges[j+1]:.2f}"
                else:
                    label = f"{edges[j]:.2f} ≤ {self.feature_names[feature_idx]} < {edges[j+1]:.2f}"
                labels.append(label)
            
            # Add label for last bin
            labels[-1] = f"{self.feature_names[feature_idx]} ≥ {edges[-2]:.2f}"
            ordinal_feature_options[feature_idx] = labels

        new_dataset.categorical_feature_options.update(ordinal_feature_options)
        new_dataset.ordinal_feature_indices = list(ordinal_feature_options.keys())
        new_dataset.feature_boundaries = {} # we have discretized everything. There is no necessity to keep track of the high and low 

        return new_dataset

    def balance(self):
        # Calculate the minimum number of instances among all classes
        unique_labels, counts = np.unique(self.y, return_counts=True)
        min_count = np.min(counts)
        
        # Select min_count instances from each class
        indices = np.array([], dtype=int)
        for label in unique_labels:
            label_indices = np.where(self.y == label)[0]
            selected_indices = np.random.choice(label_indices, min_count, replace=False)
            indices = np.hstack((indices, selected_indices))
        
        # Shuffle the indices to mix the classes in the resulting dataset
        np.random.shuffle(indices)
        
        # Create a new balanced dataset
        balanced_X = self.X[indices]
        balanced_y = self.y[indices]
        
        # Return a new instance of TabularDataset with the balanced data
        return self.copy_with_modifications(X=balanced_X, y=balanced_y)

    def split(self, random_state=1):
        # Constants for split sizes
        test_size = 0.4  # 20% of the data goes to the initial test set
        validation_size = 0.7  # 20% of the initial test set goes to validation, rest to test

        # Split the full dataset into training and test subsets
        initial_split = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_indices, test_indices_initial = next(initial_split.split(self.X))

        # Split the initial test subset further into validation and test subsets
        validation_split = ShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
        validation_indices, test_indices_final = next(validation_split.split(test_indices_initial))

        # Adjust indices to refer back to the original dataset
        validation_indices_adjusted = test_indices_initial[validation_indices]
        test_indices_adjusted = test_indices_initial[test_indices_final]

        # Create new TabularDataset instances for each split
        train_dataset = self.copy_with_modifications(X=self.X[train_indices], y=self.y[train_indices])
        

        validation_dataset = self.copy_with_modifications(X=self.X[validation_indices_adjusted], y=self.y[validation_indices_adjusted])

        test_dataset = self.copy_with_modifications(X=self.X[test_indices_adjusted], y=self.y[test_indices_adjusted])

        return train_dataset, validation_dataset, test_dataset


    def use_specific_features(self, feature_names):
        # Determine indices of the specified features from the full feature names list
        for name in feature_names:
            if name not in self.feature_names:
                raise ValueError(f"Feature '{name}' not found in the dataset")
        feature_indices = [self.feature_names.index(name) for name in feature_names if name in self.feature_names]

        # Create new dataset with selected features
        new_X = self.X[:, feature_indices]
        
        # Update feature names
        new_feature_names = [self.feature_names[i] for i in feature_indices]
        
        # Update categorical feature options
        new_categorical_feature_options = {}
        for i, old_index in enumerate(feature_indices):
            if old_index in self.categorical_feature_indices:
                new_categorical_feature_options[i] = self.categorical_feature_options[old_index]

        # Update ordinal feature indices
        new_ordinal_feature_indices = []
        if self.ordinal_feature_indices is not None:
            new_ordinal_feature_indices = [i for i, idx in enumerate(feature_indices) if idx in self.ordinal_feature_indices]


        # TO-DO: feature boundaries have to be updated as well
        new_feature_boundaries = {i: self.feature_boundaries[idx] for i, idx in enumerate(feature_indices) if idx in self.feature_boundaries}

        # Create a new dataset with the updated attributes
        return self.copy_with_modifications(X=new_X, feature_names=new_feature_names, categorical_feature_options=new_categorical_feature_options, 
            ordinal_feature_indices=new_ordinal_feature_indices, feature_boundaries=new_feature_boundaries)


    def remove_specific_features(self, feature_names):
        # Determine indices of the specified features from the full feature names list
        remove_indices = [self.feature_names.index(name) for name in feature_names if name in self.feature_names]

        # Create new dataset without the selected features
        keep_indices = [i for i in range(len(self.feature_names)) if i not in remove_indices]
        new_X = self.X[:, keep_indices]
        
        # Update feature names
        new_feature_names = [self.feature_names[i] for i in keep_indices]
        
        # Update categorical feature options
        new_categorical_feature_options = {}
        for new_i, old_i in enumerate(keep_indices):
            if old_i in self.categorical_feature_indices:
                new_categorical_feature_options[new_i] = self.categorical_feature_options[old_i]

        # Update ordinal feature indices
        new_ordinal_feature_indices = [new_i for new_i, old_i in enumerate(keep_indices) if old_i in self.ordinal_feature_indices]

        # Update feature boundaries
        new_feature_boundaries = {new_i: self.feature_boundaries[old_i] for new_i, old_i in enumerate(keep_indices) if old_i in self.feature_boundaries}

        # Create a new dataset with the updated attributes
        return self.copy_with_modifications(X=new_X, feature_names=new_feature_names, categorical_feature_options=new_categorical_feature_options, 
            ordinal_feature_indices=new_ordinal_feature_indices, feature_boundaries=new_feature_boundaries)


    def prepare_data_for_model(self, one_hot_encode=True):
        return self.prepare_instances_for_model(self.X, one_hot_encode=one_hot_encode), self.y
        
    def prepare_instances_for_model(self, instances, one_hot_encode=True):
        if instances.ndim == 1:
            instances = instances.reshape(1, -1)

        encoded_X = np.copy(instances)  # Copy to avoid modifying original instances

        # Normalize continuous features
        if self.feature_boundaries:
            for i, (min_val, max_val) in self.feature_boundaries.items():
                if i not in self.categorical_feature_indices and min_val!=max_val:
                    encoded_X[:, i] = (encoded_X[:, i] - min_val) / (max_val - min_val)

        if self.categorical_feature_options and one_hot_encode:
            # Extract the number of categories per feature from self.categorical_feature_options
            categories = [list(range(len(self.categorical_feature_options[idx]))) for idx in sorted(self.categorical_feature_indices)]
            
            # Initialize OneHotEncoder with the correct number of categories per feature
            encoder = preprocessing.OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False)
            
            # Extract and transform categorical data
            categorical_data = encoder.fit_transform(instances[:, sorted(self.categorical_feature_indices)])

            # Remove original categorical columns
            non_cat_indices = [i for i in range(encoded_X.shape[1]) if i not in self.categorical_feature_indices]
            encoded_X = encoded_X[:, non_cat_indices]


            # Insert the one-hot encoded columns back in place
            offset = 0
            for i, idx in enumerate(sorted(self.categorical_feature_indices)):
                insert_loc = idx + offset - i
                encoded_X = np.insert(encoded_X, insert_loc, (categorical_data[:, offset:offset + len(categories[i])]).T, axis=1)
                offset += len(categories[i])

        return encoded_X


    def aggregate_importances(self, instances, importances):
        if not self.categorical_feature_options:
            return importances

        # Number of features per the original dataset (before one-hot encoding)
        num_features = len(self.feature_names)
        aggregated_importances = np.zeros((instances.shape[0], num_features))

        # Mapping from one-hot encoded index to original feature index
        feature_map = []
        feature_index = 0

        for i in range(num_features):
            if i in self.categorical_feature_options:
                # Categorical feature: map each category option
                options_count = len(self.categorical_feature_options[i])
                feature_map.extend([feature_index] * options_count)
                feature_index += 1
            else:
                # Continuous feature
                feature_map.append(feature_index)
                feature_index += 1

        # # Aggregate importances
        # for instance_index, instance in enumerate(instances):
        #     for one_hot_index, value in enumerate(instance):
        #         if value == 1 or one_hot_index not in self.categorical_feature_options:
        #             original_index = feature_map[one_hot_index]
        #             aggregated_importances[instance_index, original_index] += importances[instance_index, one_hot_index]

        # Aggregate importances for categorical features
        one_hot_index = 0
        for i in range(num_features):
            if i in self.categorical_feature_options:
                options_count = len(self.categorical_feature_options[i])
                # Sum the importances for all options of this categorical feature
                for instance_index, instance in enumerate(instances):
                    aggregated_importances[instance_index, i] = np.sum(importances[instance_index, one_hot_index:one_hot_index + options_count])
                one_hot_index += options_count
            else:
                # Directly copy the importances for continuous features
                for instance_index, instance in enumerate(instances):
                    aggregated_importances[instance_index, i] = importances[instance_index, one_hot_index]
                one_hot_index += 1


        return aggregated_importances


    def print_instance(self, instance_x, instance_y):
        # Print the instance in a human-readable format
        print("Instance:")
        for i, value in enumerate(instance_x):
            if i in self.categorical_feature_indices:
                print(f"{self.feature_names[i]}: {self.categorical_feature_options[i][int(value)]}")
            else:
                print(f"{self.feature_names[i]}: {value}")
        print(f"{self.target_name}: {self.target_options[int(self.y[int(instance_y)])]}")

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def create_smaller_dataset(self, indices):
        return self.copy_with_modifications(X=self.X[indices], y=self.y[indices])

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            # Handle slicing
            return self.create_smaller_dataset(indices)
        elif isinstance(indices, list) or isinstance(indices, np.ndarray):
            # Handle list or array of indices
            return self.create_smaller_dataset(indices)
        elif isinstance(indices, int):
            # Handle single index
            return self.X[indices], self.y[indices]
        else:
            raise TypeError("Invalid argument type.")

    def dropna(self):

        # Create a mask that will be True for rows without NaNs
        mask = ~np.isnan(self.X).any(axis=1)

        X_filtered = self.X[mask]
        y_filtered = self.y[mask]

        return self.copy_with_modifications(X=X_filtered, y=y_filtered)


    def save(self, file_location):

        script_dir = os.path.dirname(__file__)
        save_path = os.path.join(script_dir, file_location)

        # use pickle to save self
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)


    def __add__(self, other):
        # Concatenate two datasets
        new_X = np.vstack((self.X, other.X))
        new_y = np.concatenate((self.y, other.y))
        return self.copy_with_modifications(X=new_X, y=new_y)
