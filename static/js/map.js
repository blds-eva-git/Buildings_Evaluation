// // Initialize the map
// var map = L.map('map').setView([39.993, 21.99], 5); // Default to a location (latitude, longitude)

// // Set up the OpenStreetMap tile layer
// // L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
// //     attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
// // }).addTo(map);
// L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// // Add a marker for the user's location
// navigator.geolocation.getCurrentPosition(function(position) {
//     var userLocation = [position.coords.latitude, position.coords.longitude];
//     L.marker(userLocation).addTo(map).bindPopup("You are here").openPopup();
//     map.setView(userLocation, 13); // Center the map on user's location

//     // Set hidden input value for form submission
//     document.getElementById("location").value = userLocation.join(",");
// });

// // Allow user to click and select a location
// map.on('click', function(e) {
//     var clickedLocation = e.latlng;
//     L.marker(clickedLocation).addTo(map).bindPopup("Selected Location").openPopup();
//     document.getElementById("location").value = clickedLocation.lat + "," + clickedLocation.lng;
// });

// Initialize the map
var map = L.map('map').setView([39.993, 21.99], 5);

// Set up the OpenStreetMap tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// Variable to store the current marker (either from geolocation, click, or address)
var currentMarker = null;

// Add a marker for the user's location
navigator.geolocation.getCurrentPosition(function(position) {
    var userLocation = [position.coords.latitude, position.coords.longitude];
    currentMarker = L.marker(userLocation).addTo(map).bindPopup("Selected Location").openPopup();
    map.setView(userLocation, 13);

    document.getElementById("location").value = userLocation.join(",");
});

// Allow user to click and select a location
map.on('click', function(e) {
    var clickedLocation = e.latlng;

    // Remove existing marker
    if (currentMarker) {
        map.removeLayer(currentMarker);
    }

    // Add new marker
    currentMarker = L.marker(clickedLocation).addTo(map).bindPopup("Selected Location").openPopup();

    // Update hidden input
    document.getElementById("location").value = clickedLocation.lat + "," + clickedLocation.lng;
});

// Search address and place marker
function searchAddress() {
    var address = document.getElementById('addressInput').value;

    fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}`)
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                var lat = data[0].lat;
                var lon = data[0].lon;

                map.setView([lat, lon], 14);

                if (currentMarker) {
                    map.removeLayer(currentMarker);
                }

                currentMarker = L.marker([lat, lon]).addTo(map)
                    .bindPopup(data[0].display_name)
                    .openPopup();

                document.getElementById("location").value = lat + "," + lon;
            } else {
                alert("Address not found!");
            }
        })
        .catch(error => {
            console.error("Geocoding error:", error);
            alert("Error occurred while searching address.");
        });
}

