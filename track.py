import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


class Tracker:
    # Tracking using Global Nearest Neighbor
    def __init__(self):
        self.tracks = {}
        self.vacant_id = 0
        self.mileage = 0
        self.min_y_position = -300
        self.max_y_position = 300
        self.location_scale = 1000
        self.tomato_number_scale = 30
        self.association_threshold = 0.6
        self.track_timeout = 15  # [sec]
        self.tracks_to_remove = set()  # using a set to avoid duplicates of removal requests

    def create_new_track(self, new_track):
        # get the largest track id, and create a new track with all the relevant parameters
        self.tracks[self.vacant_id] = {'last_seen': new_track['timestamp'], 'visible': True,
                                       'params': {k: new_track[k] for k in ('position', 'tomatoes', 'confidence')}}
        self.vacant_id += 1

    def associate(self, existing_track_id, new_timestamp, new_track_data):
        # initial implementation: just override existing data for each track.
        # next: use complementary or kalman filter
        # next: also use hidden-but-required features that were detected in previous frames, such as cut position
        self.tracks[existing_track_id]['last_seen'] = new_timestamp
        self.tracks[existing_track_id]['params'] = new_track_data

    def update_predictions(self, new_mileage, new_timestamp):
        # update the expected location, based on the distance driven since the last epoch
        diff_mileage = new_mileage - self.mileage
        for track_id, track in self.tracks.items():
            if (self.min_y_position < track['params']['position'][1] + diff_mileage < self.max_y_position and
                    new_timestamp - track['last_seen'] < self.track_timeout):
                track['params']['position'][1] += diff_mileage
            else:
                # outside relevant visible window, or too old
                self.tracks_to_remove.add(track_id)

        self.mileage = new_mileage  # update the mileage

    def clean_old_tracks(self):
        for track_id in self.tracks_to_remove:
            self.tracks.pop(track_id)

    def normalize_3d_location(self, location):
        return location / self.location_scale

    def normalize_sub_object_count(self, tomato_count):
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
        _likelihood_matrix = np.zeros((num_predicted_objects, num_detected_objects))
        _track_idx_to_id = []  # holds the relation between track_id to the positional idx in the matrix

        # Calculate likelihoods based on Euclidean distance and other metrics
        for i, (track_id, track_data) in enumerate(self.tracks.items()):
            _track_idx_to_id.append(track_id)  # saves idx-id relation
            for j in range(num_detected_objects):
                predicted_location = self.normalize_3d_location(track_data['params']['position'])
                detected_location = self.normalize_3d_location(detected_features[j]['position'])
                location_distance = np.linalg.norm(predicted_location - detected_location)

                predicted_score = track_data['params']['confidence']
                detected_score = detected_features[j]['confidence']
                score_distance = abs(predicted_score - detected_score)

                predicted_sub_object_count = self.normalize_sub_object_count(track_data['params']['tomatoes'])
                detected_sub_object_count = self.normalize_sub_object_count(detected_features[j]['tomatoes'])
                sub_object_count_distance = abs(predicted_sub_object_count - detected_sub_object_count)

                # Combine distances using a weighted sum or other strategy
                total_distance = 0.8 * location_distance + 0.1 * score_distance + 0.1 * sub_object_count_distance

                # Using a Gaussian function to convert distance to likelihood (bigger value = more likely)
                _likelihood_matrix[i][j] = np.exp(-0.5 * (total_distance ** 2))

        return _likelihood_matrix, _track_idx_to_id

    def associate_tracks(self, _likelihood_matrix):
        print(f'\ncost_matrix: \n{cost_matrix}\n')

        # rows = bidders (or owners), columns = goods
        num_bidders = cost_matrix.shape[0]
        num_goods = cost_matrix.shape[1]

        owners_allocated_to_goods = dict.fromkeys(range(num_goods))
        association_matrix = np.zeros((num_bidders, num_goods), dtype=int)
        best_prices = [0] * num_goods
        bidders_queue = list(range(num_bidders))
        # epsilon_price = round(np.log10(np.mean(cost_matrix)))
        # epsilon_price = 1 / num_bidders + 1
        # epsilon_price = 0.1
        epsilon_price = np.mean(np.sort(cost_matrix, axis=None)) / 10

        print('start bidding...')
        iterations = 0
        while len(bidders_queue) > 0:
            iterations += 1
            bidder = bidders_queue.pop(0)  # take the first bidder in line
            print(f'{bidder=}')
            desired_good = np.argmax(benefits := (cost_matrix[bidder, :] - best_prices))
            price_rise = benefits[desired_good]
            if not association_matrix[:, desired_good].any():
                association_matrix[bidder, desired_good] = 1  # first time assignment of a good
                best_prices[desired_good] += epsilon_price
            if price_rise > epsilon_price and association_matrix[
                bidder, desired_good] == 0:  # re-assignment for higher bid
                # put previous bidder in end of queue, and set new bidder as the owner
                previous_owner = association_matrix[:, desired_good].argmax()
                bidders_queue.append(previous_owner)
                association_matrix[previous_owner, desired_good] = 0
                association_matrix[bidder, desired_good] = 1
                best_prices[desired_good] += epsilon_price
            print(association_matrix)

        total_price = 0
        print('\nfinal results:')
        print(f'association_matrix=\n{association_matrix}')
        for item in range(num_goods):
            if association_matrix[:, item].sum() > 0:
                owner = association_matrix[:, item].argmax()
                total_price += cost_matrix[owner, item]
                print(f'{item=}: {owner=}')
            else:
                print(f'{item=}: no owner')
        for owner in range(num_bidders):
            if association_matrix[owner, :].sum() == 0:
                print(f'no item for {owner=}')

        print(f'{iterations=}')
        print(f'{total_price=}')

    def tracking_iteration(self, detections):
        # Track objects detected in the new frame probabilities

        # we take the timestamp from one of the detections, but we need to handle a case where there are no detections
        # so just save the frame's timestamp in the pickle
        frame_timestamp = detections[0]['timestamp']
        self.clean_old_tracks()
        self.update_predictions(0, frame_timestamp)

        if len(detections) == 0:  # if there are no detections
            return None

        if len(self.tracks) == 0:  # for a case where there are no existing tracks
            for i in range(len(detections)):
                self.create_new_track(detections[i])
            return None

        # if there are tracks, create the probability matrix for them
        likelihood_matrix, track_idx_to_id = self.calculate_measurement_likelihoods(detections)
        association_matrix = self.associate_tracks(likelihood_matrix)

        for detection_idx in range(len(detections)):
            if association_matrix[:, detection_idx].sum() > 0:
                # Associate the prediction with the corresponding valid detection
                association_idx = association_matrix[:, detection_idx].argmax()
                track_id = track_idx_to_id[association_idx]
                self.associate(self.tracks[track_id], frame_timestamp, detections[detection_idx])
            else:
                # If no track could be related to this new detection,create it as a new track
                self.create_new_track(detections[detection_idx])


def load_pickle_file(filename):
    with open(filename, 'rb') as _file:
        file_content = pickle.load(_file)
    return file_content


tracker = Tracker()
folder_name = '/home/gilad/work/poc_s/tracking/tracking_dataset/tracking_simple_case/BACK/'
folder_path = Path(folder_name)

file_list = list(folder_path.iterdir())
frames = []

for file in file_list:
    if file.suffix == '.pkl':
        frame_detections = load_pickle_file(file)
        tracker.tracking_iteration(frame_detections)
        frames.append(frame_detections)

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
