import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def read_data(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            hour = parts[0]
            timestamp = parts[1]
            owner = parts[2]
            id_num = int(parts[3])
            visibility = parts[4] == 'True'
            list_of_strings = parts[5].strip('[]').split()
            location = [float(num_str) for num_str in list_of_strings]
            if owner not in data:
                data[owner] = {}
            if timestamp not in data[owner]:
                data[owner][timestamp] = []
            data[owner][timestamp].append((id_num, location, visibility))
    return data


def plot_data(data, timestamp, owner):
    ax.clear()
    if owner in data and timestamp in data[owner]:
        all_visible_data = []
        all_invisible_data = []
        for id_num, location, visible in data[owner][timestamp]:
            if visible:
                all_visible_data.append((location, id_num))
            else:
                all_invisible_data.append((location, id_num))

        visible_x = [location[0] for location, id_num in all_visible_data]
        visible_y = [location[1] for location, id_num in all_visible_data]
        visible_z = [location[2] for location, id_num in all_visible_data]
        visible_ids = [str(id_num) for location, id_num in all_visible_data]

        invisible_x = [location[0] for location, id_num in all_invisible_data]
        invisible_y = [location[1] for location, id_num in all_invisible_data]
        invisible_z = [location[2] for location, id_num in all_invisible_data]
        invisible_ids = [str(id_num) for location, id_num in all_invisible_data]

        ax.scatter(visible_x, visible_y, visible_z, c='b', marker='o')
        ax.scatter(invisible_x, invisible_y, invisible_z, c='gray', marker='o')

        for x, y, z, id_num in zip(visible_x, visible_y, visible_z, visible_ids):
            ax.text(x, y, z, id_num, color='b')

        for x, y, z, id_num in zip(invisible_x, invisible_y, invisible_z, invisible_ids):
            ax.text(x, y, z, id_num, color='gray')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    canvas.draw()


def update_timestamps(owner):
    global selected_owner, selected_timestamp_index
    selected_owner = owner
    timestamps_for_owner = timestamps[owner]
    if timestamps_for_owner:
        selected_timestamp_index = 0
        timestamp_var.set(timestamps_for_owner[selected_timestamp_index])
        plot_data(data, timestamps_for_owner[selected_timestamp_index], owner)
    else:
        selected_timestamp_index = -1
        timestamp_var.set('')
        plot_data(data, '', owner)


def prev_timestamp():
    global selected_timestamp_index
    timestamps_for_owner = timestamps[selected_owner]
    if timestamps_for_owner:
        selected_timestamp_index = (selected_timestamp_index - 1) % len(timestamps_for_owner)
        timestamp_var.set(timestamps_for_owner[selected_timestamp_index])
        plot_data(data, timestamps_for_owner[selected_timestamp_index], selected_owner)


def next_timestamp():
    global selected_timestamp_index
    timestamps_for_owner = timestamps[selected_owner]
    if timestamps_for_owner:
        selected_timestamp_index = (selected_timestamp_index + 1) % len(timestamps_for_owner)
        timestamp_var.set(timestamps_for_owner[selected_timestamp_index])
        plot_data(data, timestamps_for_owner[selected_timestamp_index], selected_owner)


# Create the main window
root = tk.Tk()
root.title("3D Plot Visualizer")

# File selection
file_path = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text Files", "*.txt")])
data = read_data(file_path)

# Get the available owners and their associated timestamps
owners = list(data.keys())
timestamps = {owner: sorted(data[owner].keys()) for owner in owners}

# Create the figure and axis
fig = Figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# Create the canvas for the figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Owner selection
owner_var = tk.StringVar(value=owners[0])
owner_frame = tk.Frame(root)
owner_frame.pack(pady=10)
tk.Label(owner_frame, text="Owner:").pack(side=tk.LEFT)
owner_buttons = []
for owner in owners:
    button = tk.Radiobutton(owner_frame, text=owner, variable=owner_var, value=owner, command=lambda o=owner: update_timestamps(o))
    button.pack(side=tk.LEFT)
    owner_buttons.append(button)

# Timestamp selection
timestamp_var = tk.StringVar()
timestamp_frame = tk.Frame(root)
timestamp_frame.pack(pady=10)

timestamp_label = tk.Label(timestamp_frame, textvariable=timestamp_var, font=("Arial", 14))
timestamp_label.pack(side=tk.TOP)

selected_owner = owners[0]
selected_timestamp_index = 0

prev_button = tk.Button(timestamp_frame, text="<", command=prev_timestamp)
prev_button.pack(side=tk.LEFT)

next_button = tk.Button(timestamp_frame, text=">", command=next_timestamp)
next_button.pack(side=tk.LEFT)

# Start the main loop
update_timestamps(owners[0])
root.mainloop()
