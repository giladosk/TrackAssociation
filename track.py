import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy

np.set_printoptions(suppress=True)


class Tracker:
    # Tracking using Global Nearest Neighbor
    def __init__(self):
        self.tracks = {}
        self.track_log = []
        self.vacant_id = 0
        self.mileage = 0
        self.min_y_position = -400
        self.max_y_position = 400
        self.location_scale = 100  # normalize to be equivalent to decimeters
        self.tomato_number_scale = 10
        self.association_threshold = 3  # how far in decimeters-equivalent do we think a cluster can be
        self.track_timeout_sec = 60  # [sec]
        self.tracks_to_remove = set()  # using a set to avoid duplicates of removal requests

    def create_new_track(self, new_track):
        # get the largest track id, and create a new track with all the relevant parameters
        self.tracks[self.vacant_id] = {'last_seen': new_track['timestamp'], 'visible': True,
                                       'params': {k: new_track[k] for k in ('position', 'tomatoes', 'confidence')}}
        self.tracks[self.vacant_id]['params']['position'][1] += self.mileage  # add mileage driven so far
        print(f'creating track #{self.vacant_id}')
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
                    new_timestamp - track['last_seen'] < self.track_timeout_sec * 1e9):
                track['params']['position'][1] -= diff_mileage  # subtract the extra mileage driven
            else:
                # outside relevant visible window, or too old
                print(f'{track_id} is set to be removed')
                self.tracks_to_remove.add(track_id)

        self.mileage = new_mileage  # update the mileage

    def clean_old_tracks(self):
        while len(self.tracks_to_remove) > 0:
            track_id = self.tracks_to_remove.pop()
            self.tracks.pop(track_id)
            print(f'removing track #{track_id}')

    def calc_location_distance(self, location_detection, location_prediction):
        return np.linalg.norm(location_detection - location_prediction) / self.location_scale

    def calc_tomato_count_distance(self, tomato_count_detection, tomato_count_prediction):
        return abs(tomato_count_detection - tomato_count_prediction) / self.tomato_number_scale

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
                location_distance = self.calc_location_distance(detected_features[j]['position'], track_data['params']['position'])
                tomato_count_distance = self.calc_tomato_count_distance(detected_features[j]['tomatoes'], track_data['params']['tomatoes'])

                # Combine distances using a weighted sum
                total_distance = 0.8 * location_distance + 0.2 * tomato_count_distance

                # Using a Gaussian function to convert distance to likelihood (bigger value = more likely)
                if total_distance < self.association_threshold:
                    # when it is a plausible association
                    _likelihood_matrix[i][j] = np.exp(-total_distance)
                else:
                    # when the association is so much unlikely we want to forbid it, its profit is negative
                    _likelihood_matrix[i][j] = -1

        return _likelihood_matrix, _track_idx_to_id

    @staticmethod
    def associate_tracks(_likelihood_matrix):
        # the bidders(tracks) try tp the get the goods(detection) they value the most
        print(f'\nlikelihood_matrix=\n{_likelihood_matrix}')

        # rows = bidders (or owners), columns = goods
        num_bidders = _likelihood_matrix.shape[0]
        num_goods = _likelihood_matrix.shape[1]

        association_matrix = np.zeros((num_bidders, num_goods), dtype=int)
        best_prices = [0] * num_goods
        bidders_queue = list(range(num_bidders))
        epsilon_price = np.mean(abs(np.sort(_likelihood_matrix, axis=None))) / 10  # bid step ~(10th * data resolution)

        num_iterations = 0
        while len(bidders_queue) > 0:
            num_iterations += 1
            bidder = bidders_queue.pop(0)  # take the first bidder in queue
            desired_good = np.argmax(benefits := (_likelihood_matrix[bidder, :] - best_prices))
            price_rise = benefits[desired_good]
            if price_rise < epsilon_price:
                # when this bidder has no way to compete on any of the goods
                continue
            if not association_matrix[:, desired_good].any():
                # first time assignment of a good
                association_matrix[bidder, desired_good] = 1
                best_prices[desired_good] += epsilon_price
            elif association_matrix[bidder, desired_good] == 0:
                # re-assignment for higher bid
                # put previous bidder in end of queue, and set new bidder as the owner
                previous_owner = association_matrix[:, desired_good].argmax()
                bidders_queue.append(previous_owner)
                association_matrix[previous_owner, desired_good] = 0
                association_matrix[bidder, desired_good] = 1
                best_prices[desired_good] += epsilon_price

        print(f'association_matrix=\n{association_matrix}')
        total_profit = 0
        for item in range(num_goods):
            if association_matrix[:, item].sum() > 0:
                owner = association_matrix[:, item].argmax()
                total_profit += _likelihood_matrix[owner, item]
                print(f'{item=}: {owner=}')
            else:
                print(f'{item=}: no owner')
        for owner in range(num_bidders):
            if association_matrix[owner, :].sum() == 0:
                print(f'no item for {owner=}')

        print(f'{num_iterations=}')
        print(f'{total_profit=}')

        return association_matrix

    def tracking_iteration(self, detections):
        # Track objects detected in the new frame probabilities

        # we take the timestamp from one of the detections, but we need to handle a case where there are no detections
        # so just save the frame's timestamp in the pickle
        frame_timestamp = detections[0]['timestamp']
        print(f'\n{frame_timestamp=}')
        self.update_predictions(0, frame_timestamp)
        self.clean_old_tracks()

        if len(detections) == 0:  # if there are no detections
            return None

        if len(self.tracks) == 0:  # for a case where there are no existing tracks
            print('no existing track, initializing new tracks')
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
                self.associate(track_id, frame_timestamp, detections[detection_idx])
            else:
                # If no track could be related to this new detection,create it as a new track
                self.create_new_track(detections[detection_idx])

        # (no need for a special action for tracks that had no detections, they'll be handled in update_predictions)
        print(f'current track ids: {list(self.tracks.keys())}')

    def log_detections_and_tracks(self):
        # save the current active tracks to a database
        self.track_log.append(deepcopy(self.tracks))


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
        tracker.log_detections_and_tracks()
        frames.append(frame_detections)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ['r', 'b', 'm', 'c']
frames.sort(key=lambda d: d[0]['timestamp'])
fig_limits = {'xmin': 1000, 'xmax': -1000, 'ymin': 1000, 'ymax': -1000, 'zmin': 1000, 'zmax': -1000}
for frame, color in zip(tracker.track_log, colors):
    for track_id, cluster in frame.items():
        position = cluster['params']['position']
        track_text = f'{track_id}'
        # ax.scatter(position[0], position[1], position[2], c=color)
        ax.text(position[0], position[1], position[2], track_text, size=10, color=color)
        fig_limits['xmin'] = min(fig_limits['xmin'], position[0])
        fig_limits['ymin'] = min(fig_limits['ymin'], position[1])
        fig_limits['zmin'] = min(fig_limits['zmin'], position[2])
        fig_limits['xmax'] = max(fig_limits['xmax'], position[0])
        fig_limits['ymax'] = max(fig_limits['ymax'], position[1])
        fig_limits['zmax'] = max(fig_limits['zmax'], position[2])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(fig_limits['xmin'] - 50, fig_limits['xmax'] + 50)
ax.set_ylim(fig_limits['ymin'] - 50, fig_limits['ymax'] + 50)
ax.set_zlim(fig_limits['zmin'] - 50, fig_limits['zmax'] + 50)
plt.show()

last_line = 0
