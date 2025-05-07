import folium

class TrafficMap:
    def __init__(self, center_location, zoom_start=12):
        self.map = folium.Map(location=center_location, zoom_start=zoom_start)
    
    def add_congestion_markers(self, locations, congestion_levels):
        """Add congestion markers to the map"""
        for loc, level in zip(locations, congestion_levels):
            color = 'green'
            if level > 0.7:
                color = 'red'
            elif level > 0.4:
                color = 'orange'
                
            folium.CircleMarker(
                location=loc,
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Congestion: {level:.2f}"
            ).add_to(self.map)
    
    def get_map(self):
        """Return the map object"""
        return self.map