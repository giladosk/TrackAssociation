import numpy as np
import itertools
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


def calculate_measurement_likelihoods(predicted_features, detected_features):
    """
    Example usage:
    Assuming predicted_features and detected_features are lists of feature vectors
    where each feature vector may contain 3D location (x, y, z), score, and number of sub-objects.
    probability_matrix = calculate_measurement_likelihoods(predicted_features, detected_features)
    """

    num_predicted_objects = len(predicted_features)
    num_detected_objects = len(detected_features)

    # Set up a matrix to store the likelihoods
    likelihood_matrix = np.zeros((num_predicted_objects, num_detected_objects))

    # Example: Normalization functions for 3D location, score, and number of sub-objects
    def normalize_3d_location(location):
        # Placeholder normalization function
        return location / max(location)

    def normalize_score(score):
        # Placeholder normalization function
        return score / max(score)

    def normalize_sub_object_count(sub_object_count):
        # Placeholder normalization function
        return sub_object_count / max(sub_object_count)

    # Calculate likelihoods based on Euclidean distance and other metrics
    for i in range(num_predicted_objects):
        for j in range(num_detected_objects):
            predicted_location = normalize_3d_location(predicted_features[i][:3])
            detected_location = normalize_3d_location(detected_features[j][:3])
            location_distance = np.linalg.norm(predicted_location - detected_location)

            # Example: Assuming predicted_features and detected_features contain score and sub-object count
            predicted_score = normalize_score(predicted_features[i][3])
            detected_score = normalize_score(detected_features[j][3])
            score_distance = abs(predicted_score - detected_score)

            predicted_sub_object_count = normalize_sub_object_count(predicted_features[i][4])
            detected_sub_object_count = normalize_sub_object_count(detected_features[j][4])
            sub_object_count_distance = abs(predicted_sub_object_count - detected_sub_object_count)

            # Combine distances using a weighted sum or other strategy
            total_distance = 0.8 * location_distance + score_distance + sub_object_count_distance

            # Example: Using a Gaussian function to convert distance to likelihood
            likelihood = np.exp(-0.5 * (total_distance ** 2))

            # Store the likelihood in the matrix
            likelihood_matrix[i][j] = likelihood

    return likelihood_matrix


def calculate_joint_probabilities(probability_matrix):
    """
    Example usage:
    Assuming probability_matrix is a 2D array representing measurement likelihoods
    where probability_matrix[i][j] is the likelihood of detecting predicted object i with detected object j.
    joint_probabilities = calculate_joint_probabilities(probability_matrix)
    """
    num_predicted_objects = len(probability_matrix)
    num_detected_objects = len(probability_matrix[0])

    # Generate all possible combinations of associations
    all_associations = list(itertools.product(range(num_detected_objects), repeat=num_predicted_objects))

    joint_probabilities = []

    for association in all_associations:
        joint_probability = 1.0

        for i in range(num_predicted_objects):
            joint_probability *= probability_matrix[i][association[i]]

        joint_probabilities.append(joint_probability)

    return joint_probabilities


class TrackingJPDA:
    # Tracking using Joint Probabilistic Data Association
    def __init__(self):
        self.tracks = {}
        self.vacant_id = 0
        self.mileage = 0
        self.min_y_position = -300
        self.max_y_position = 300
        self.track_timeout = 15  # [sec]

    def clean_old_tracks(self, frame_timestamp):
        # delete tracks that were not seen for too long
        for track_id, track in self.tracks.items():
            if frame_timestamp - track['last_seen'] < self.track_timeout:
                self.remove_track(track_id)

    def associate(self, existing_track_id, new_timestamp, new_track_data):
        # initial implementation: just override existing data for each track.
        # next: use complementary or kalman filter
        # next: also use hidden-but-required features that were detected in previous frames, such as cut position
        self.tracks[existing_track_id]['last_seen'] = new_timestamp
        self.tracks[existing_track_id]['params'] = new_track_data

    def create_new_track(self, frame_timestamp, new_track_data):
        # get the largest track id, and create a new track with all the relevant parameters
        self.tracks[self.vacant_id] = {'last_seen': frame_timestamp, 'params': new_track_data, 'visible': True}
        self.vacant_id += 1

    def remove_track(self, track_id):
        self.tracks.pop(track_id)

    def update_predictions(self, new_mileage):
        # update the expected location, based on the distance driven since the last epoch
        diff_mileage = new_mileage - self.mileage
        for track_id, track in self.tracks.items():
            if self.min_y_position < track['params']['position'][1] + diff_mileage < self.max_y_position:
                track['params']['position'][1] += diff_mileage
            else:
                # outside relevant visible window
                self.remove_track(track_id)

        self.mileage = new_mileage  # update the mileage


def tracking_routine(detections, predictions, threshold):
    # Perform JPDA calculations to obtain joint probabilities
    probability_matrix = calculate_measurement_likelihoods(predictions, detections)
    joint_probabilities = calculate_joint_probabilities(probability_matrix)

    # Iterate through predicted objects
    for i in range(len(detections)):
        # Find the detection with the highest association probability for the current prediction
        best_association_index = np.argmax(joint_probabilities[i, :])

        # Check if the highest probability is above the threshold
        if joint_probabilities[i, best_association_index] > threshold:
            # Associate the prediction with the corresponding detection
            associate(predictions[best_association_index], detections[i])
        else:
            # If probability is below threshold, consider it a new track
            create_new_track(detections[i])

    # Manage tracks (e.g., update existing tracks, handle occlusion, etc.)
    clean_old_tracks()


def load_pickle_file(filename):
    with open(filename, 'rb') as _file:
        file_content = pickle.load(_file)
    return file_content


folder_name = '/home/gilad/work/poc_s/tracking/tracking_dataset/tracking_simple_case/BACK/'
folder_path = Path(folder_name)

file_list = list(folder_path.iterdir())
frames = []

for file in file_list:
    if file.suffix == '.pkl':
        frames.append(load_pickle_file(file))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ['r', 'b', 'm', 'c']
for frame, color in zip(frames, colors):
    print(len(frame))
    for cluster in frame:
        position = cluster['position']
        ax.scatter(position[0], position[1], position[2], c=color)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

last_line = 0
