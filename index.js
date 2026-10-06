const { greet, about } = require('./src/lib.js');
const args = process.argv.slice(2);

if (args.length <= 0) {
	console.error("Error: no name provided");
	process.exit(1);
}

const name = args[0];
greet(name);
// about(name);
