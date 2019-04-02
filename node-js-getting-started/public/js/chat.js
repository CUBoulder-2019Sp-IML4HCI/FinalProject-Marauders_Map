var Firebase = require('Firebase');
var ref = new Firebase('https://breezytalk.firebaseio.com');

$('document').ready(function(){
	$('.ui.accordion').accordion();
	var rooms = [];
	var myroom="";
	var socket=io();
	var name="";
	$('.tabl').tab();
	name =prompt('enter your name');
	$('#random').click(function(){
		socket.emit('waiting');
	});

        $('#logout').click(function(){
		ref.unauth();
	});

	$('#sender').click(function(){
		sendmess();
	});	
	
	$('#messfield').keydown(function(key)
	{
		if(key.which == 13)
		{
			sendmess();
		}
	});
	
	$('#addfriend').click(function()
	{
		$('#friendadder').modal('toggle');
	});
	
	socket.on('found', function(data){
		socket.emit('stop');
		socket.emit('join',{room: data.room, name: name});
		myroom=data.room;
		$('#ran').append('<p>You Found a Friend!<p>');
	});
	
	socket.on('gotmessage', function(data){
		newmessage(data);
	});
	
	function newmessage(data)
	{
		$('#ran').append('<p>&#60'+data.name+'&#62'+data.message+'<p>');
		$('#ran').scrollTop($('#ran').prop("scrollHeight"));
	}
	
	function sendmess()
	{
		var mess = $('#messfield').val();
		$('#messfield').val("");
		socket.emit('sentmessage', {message:mess, name:name, room:myroom});
	}
});
