var Firebase = require('Firebase');
var topics   = new Firebase("https://breezytalk.firebaseio.com/topics");

topics.on('value', function(snapshot) {
    document.getElementById('topic').innerHTML = snapshot.val();
});
