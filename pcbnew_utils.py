import sys
sys.path.append("C:\\Program Files\\KiCad\\7.0\\bin")
#sys.path.append("C:\\Users\\manue\\AppData\\Roaming\\kicad\\7.0")
#sys.path.append("C:\\Program Files\\KiCad\\7.0\\bin\\Lib\\site-packages")
print(sys.version)
import pcbnew


board = pcbnew.LoadBoard("C:\\Users\\manue\\Desktop\\KiCAD\\test\\test.kicad_pcb")

# footprints = board.GetFootprints()

# find the matching net for the track
m = board.GetPads()

bounding_box = board.GetBoundingBox()

# Calculate the board size
width = bounding_box.GetWidth()
height = bounding_box.GetHeight()

print(f"Board width: {width}")
print(f"Board height: {height}")

#print(len(m))
for x in m:
    print("Pad", x)
    #try:
    #    y = board.GetTraksByPosition(x)
    #    print(y)
    #except Exception as e:
    #    print(e)
l = board.Zones()
print("\nZones", l)


from math import gcd
from functools import reduce

# Function to calculate the greatest common divisor of a list
def find_gcd(list):
    x = reduce(gcd, list)
    return x

# Initialize an array to hold pad information
pads_info = []

# Initialize a list to hold all dimensions for scale factor calculation
all_dimensions = []

# Iterate through all footprints on the board
for footprint in board.GetFootprints():
    # Get the footprint's position relative to the board's origin
    footprint_pos = footprint.GetPosition()
    footprint_x, footprint_y = footprint_pos.x, footprint_pos.y

    # Get the rotation angle of the footprint
    rotation_angle = footprint.GetOrientation() / 10.0  # Orientation is in tenths of degrees

    # Iterate through all pads in the footprint
    for pad in footprint.Pads():
        # Get pad position relative to the footprint's origin
        pad_pos = pad.GetPosition()
        pad_x, pad_y = pad_pos.x - footprint_x, pad_pos.y - footprint_y

        # Get pad size
        pad_size = pad.GetSize()
        width, height = pad_size.x, pad_size.y

        # Add dimensions to the list for scale factor calculation
        all_dimensions.extend([width, height])

        # Add pad information to the array
        pads_info.append({
            'start_x': pad_x,
            'start_y': pad_y,
            'width': width,
            'height': height,
            'rotation': rotation_angle
        })

# Calculate the scale factor
scale_factor = find_gcd(all_dimensions)
print("Scale factor", scale_factor, "\n")

# as putea folosi si scale factor pentru a mai reduce dimensiunile insa ar fi mai bine sa o

# Normalize dimensions using the scale factor
for pad in pads_info:
    pad['start_x'] /= scale_factor
    pad['start_y'] /= scale_factor
    pad['width'] /= scale_factor
    pad['height'] /= scale_factor

# Print the pads information
for pad in pads_info:
    print("Pad_info", pad)

# TODO: Define the routes and bend points based on your algorithm



'''x = board.GetTracks()
for t in x:
    print(t.GetBoundingBox(),"\n",
          t.GetEffectiveShape(), "\n",
          t.GetEndX(), t.GetEndY(), "\n",
          t.GetWidth(), "\n",
          t.GetPosition(), "\n")'''
# optionally you can create the net if it doesn't exist - handy if you are creating a PCB from scratch
#if net is None:
#    net = pcbnew.NETINFO_ITEM(board, "NET NAME")
#    board.Add(net)