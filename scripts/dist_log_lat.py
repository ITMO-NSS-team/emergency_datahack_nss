def calc_dist(lat1, lat2, lon1, lon2):
    r = 6371e3
    f1 = lat1 * math.pi/180
    f2 = lat2 * math.pi/180
    del_lat = (lat2 - lat1) * math.pi/180
    del_lon = (lon2 - lon1) * math.pi/180

    a = math.sin(del_lat/2) * math.sin(del_lat/2) + math.cos(f1) * math.cos(f2) * math.sin(del_lon/2) * math.sin(del_lon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = r * c
    
    return d