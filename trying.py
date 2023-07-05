import random

#Creating hotspot circler regions by defining their center
hotspot_radius = 40 #to avoid overlapping
area_Xaxis_bound = (-1000, 1000) #The area of interest
area_Yaxis_bound = (-1000, 1000) #The area of interest
Num_hotspot = 19 #example 38 hotspots, later it can be random

# Generate a set of all points within 80m of the origin, to be used as offsets later
offset_points = set()
for x in range(-hotspot_radius, hotspot_radius+1):
    for y in range(-hotspot_radius, hotspot_radius+1):
        if x*x + y*y <= hotspot_radius*hotspot_radius:
            offset_points.add((x,y))

hotspot_center_cord = []
excluded = set()
i = 0
while i<Num_hotspot:
    x = random.randrange(*area_Xaxis_bound)
    y = random.randrange(*area_Yaxis_bound)
    if (x,y) in excluded: continue
    hotspot_center_cord.append((x,y))
    i += 1
    excluded.update((x+dx, y+dy) for (dx,dy) in offset_points)
