$('document').ready(function(){
	var socket=io();
	$('button').click(function(){
		socket.emit('waiting');
	});
	socket.on('found', function(){
		alert('worked');
	});
});