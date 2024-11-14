import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = [
    ['Header1', 'Header2'],
    [1, 2],
    [3, 4],
    [5, 6]
]

# Create a blank figure
fig, ax = plt.subplots()

# Hide the axes
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=data, loc='center', cellLoc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 1.2)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        continue
    cell.set_facecolor("#56b5fd")
    cell.set_text_props(color='white', weight='bold')

plt.show()