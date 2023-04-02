function toggleDown(elem) {
	// function to toggle up and down the parameters div

	let button = document.getElementById('param-button')
	elem = document.getElementById(elem);

	if (elem.style.visibility === 'visible') {
		button.innerHTML = 'Parameters ▼';
		elem.style.visibility = 'hidden';
	} else {
		button.innerHTML = 'Parameters ▲';
		elem.style.visibility = 'visible';
	}

}

function validateArtistForm() {
	// function to check if every element in the form is filled
	
	var artist = document.forms["artist-input-field"]["artist-placeholder"].value;
	var epochs = document.forms["artist-input-field"]["epochs"].value;
	var max_songs = document.forms["artist-input-field"]["max-songs"].value;

	if (artist == "") {
		return false;
	} else if (max_songs == "") {
		return false;
	} else if (epochs == "") {
		return false;
	} 
	return true;

}

// function loadingMessage() {
// 	// function to display a message once the form is validated with parameters chosen by user

// 	var artist = document.forms["artist-input-field"]["artist-placeholder"].value;
// 	var epochs = document.forms["artist-input-field"]["epochs"].value;
// 	var max_songs = document.forms["artist-input-field"]["max-songs"].value;
// 	var message_div = document.getElementById("message-display");

// 	if (validateArtistForm()) {
// 		message_div.innerHTML = '<p>Fetching ' + max_songs + ' songs for ' + artist + '.</p><br><p>Training with ' + epochs + ' epochs.</p>'
// 	}
// }

function hideIfEmpty(elem) {
	// function to hide generation and error message divs while they're empty

	if(elem.innerHTML === "") {
		elem.style.display = "none";
	} else {
		elem.style.display = "block";
	}
}

var gen = document.getElementById("generation-output");
var error = document.getElementById("message-display");

hideIfEmpty(gen)
hideIfEmpty(error)