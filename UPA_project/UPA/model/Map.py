import folium
class Map:
    def __init__(self):
        self.m = folium.Map(location=[50, 15], zoom_start=6)

    def show(self):
        self.m
