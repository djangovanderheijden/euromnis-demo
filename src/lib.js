const information = require('../assets/about.json');

module.exports = {
	greet: function(name) {
		console.log(`Hello, ${name}!!`);
	},
	
	about: function(name) {
		console.log(information[name] ?? "No additional information available. :(");
	}
};
