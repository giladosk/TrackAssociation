import numpy as np
import itertools
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


class TrackingJPDA:
    # Tracking using Joint Probabilistic Data Association
    def __init__(self):
        self.tracks = {}
        self.vacant_id = 0
        self.mileage = 0
        self.min_y_position = -300
        self.max_y_position = 300
        self.location_scale = 1000
        self.tomato_number_scale = 30
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

    # Example: Normalization functions for 3D location, score, and number of sub-objects
    def normalize_3d_location(self, location):
        # Placeholder normalization function
        return location / self.location_scale

    def normalize_sub_object_count(self, tomato_count):
        # Placeholder normalization function
        return tomato_count / self.tomato_number_scale

    def calculate_measurement_likelihoods(self, detected_features):
        """
        Example usage:
        Assuming predicted_features and detected_features are lists of feature vectors
        where each feature vector may contain 3D location (x, y, z), score, and number of sub-objects.
        probability_matrix = calculate_measurement_likelihoods(predicted_features, detected_features)
        """

        num_predicted_objects = len(self.tracks)
        num_detected_objects = len(detected_features)

        if num_predicted_objects == 0:
            return None

        # Set up a matrix to store the likelihoods
        likelihood_matrix = np.zeros((num_predicted_objects, num_detected_objects))

        # Calculate likelihoods based on Euclidean distance and other metrics
        for i, track in enumerate(self.tracks.values()):
            for j in range(num_detected_objects):
                predicted_location = self.normalize_3d_location(track['params']['position'])
                detected_location = self.normalize_3d_location(detected_features[j][:3])
                location_distance = np.linalg.norm(predicted_location - detected_location)

                # Example: Assuming predicted_features and detected_features contain score and sub-object count
                predicted_score = track['params']['confidence']
                detected_score = detected_features[j][3]
                score_distance = abs(predicted_score - detected_score)

                predicted_sub_object_count = self.normalize_sub_object_count(track['params']['tomatoes'])
                detected_sub_object_count = self.normalize_sub_object_count(detected_features[j][4])
                sub_object_count_distance = abs(predicted_sub_object_count - detected_sub_object_count)

                # Combine distances using a weighted sum or other strategy
                total_distance = 0.8 * location_distance + 0.1 * score_distance + 0.1 * sub_object_count_distance

                # Example: Using a Gaussian function to convert distance to likelihood
                likelihood = np.exp(-0.5 * (total_distance ** 2))

                # Store the likelihood in the matrix
                likelihood_matrix[i][j] = likelihood

        # Generate all possible combinations of associations
        """
            Example usage:
            Assuming probability_matrix is a 2D array representing measurement likelihoods
            where probability_matrix[i][j] is the likelihood of detecting predicted object i with detected object j.
            joint_probabilities = calculate_joint_probabilities(probability_matrix)
            """
        all_associations = list(itertools.product(range(num_detected_objects), repeat=num_predicted_objects))

        joint_probabilities = likelihood_matrix / np.sum(likelihood_matrix, axis=1, keepdims=True)

        return joint_probabilities

    def tracking_iteration(self, detections, threshold):
        # Perform JPDA calculations to obtain joint probabilities
        joint_probabilities = self.calculate_measurement_likelihoods(detections)

        # Iterate through predicted objects
        for i in range(len(detections)):
            # Find the detection with the highest association probability for the current prediction
            best_association_index = np.argmax(joint_probabilities[i, :])

            # Check if the highest probability is above the threshold
            if joint_probabilities[i, best_association_index] > threshold:
                # Associate the prediction with the corresponding detection
                self.associate(self.tracks[best_association_index], detections[i])
            else:
                # If probability is below threshold, consider it a new track
                self.create_new_track(detections[i])

        # Manage tracks (e.g., update existing tracks, handle occlusion, etc.)
        self.clean_old_tracks()


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
frames.sort(key=lambda d: d[0]['timestamp'])
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
