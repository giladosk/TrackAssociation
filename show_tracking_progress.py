import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime


# Read the CSV file
with open('tracks.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    data = list(reader)

# Convert data to float
data = [[float(val) if float(val) < 1e8 else int(val) for val in row] for row in data]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the axes limits
x_values = [row[2] for row in data]
y_values = [row[3] for row in data]
z_values = [row[4] for row in data]
ax.set_xlim(min(x_values), max(x_values))
ax.set_ylim(min(y_values), max(y_values))
ax.set_zlim(min(z_values), max(z_values))


# Function to update the plot
def update_plot(val):
    ax.clear()
    timestamp = timestamps[int(val)]
    temp_data = [row for row in data if row[0] == timestamp]
    if temp_data:
        for row in temp_data:
            ax.scatter(row[2], row[3], row[4])  # Plot x, y, z
            ax.text(row[2], row[3], row[4], '%s' % (str(int(row[1]))), size=10, zorder=1, color='k')  # Display id
    timestamp_str = datetime.fromtimestamp(timestamp / 1e9).strftime('%Y-%m-%dT%H:%M:%S.%f')

    ax.set_xlim(min(x_values), max(x_values))
    ax.set_ylim(min(y_values), max(y_values))
    ax.set_zlim(min(z_values), max(z_values))
    ax.set_title('Timestamp: %s' % timestamp_str)  # Display the timestamp
    fig.canvas.draw_idle()


# Create a slider for time
timestamps = sorted(list(set(row[0] for row in data)))
ax_time = plt.axes([0.2, 0.01, 0.65, 0.03])
s_time = Slider(ax_time, 'Time', 0, len(timestamps) - 1, valinit=0, valfmt='%0.0f')
s_time.on_changed(update_plot)

# Initial plot
update_plot(0)

plt.show()
