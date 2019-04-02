var Firebase = require('Firebase');
var ref = new Firebase("https://breezytalk.firebaseio.com");

var submit = document.getElementById("signup-button");
var email  = document.getElementById("myEmail").value;
var pass   = document.getElementById("myPwd").value;

submit.addEventListener('click', function() {
    alert(email);
    ref.createUser({
      email    : email,
      password : pass
    }, function(error, userData) {
      if (error) {
        console.log("Error creating user:", error);
      } else {
        console.log("Successfully created user account with uid:", userData.uid);
		
      }
    });
}, false);

