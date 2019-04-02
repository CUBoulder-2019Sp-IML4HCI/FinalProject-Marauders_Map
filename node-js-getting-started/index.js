var express = require('express');
var app = require('express')();
//var uuid= require('tower-uuid');
//var useragent = require('express-useragent');
var server = require('http').createServer(app);
var io = require('socket.io')(server);
var port = process.env.PORT || 3000;
//var key = "trnsl.1.1.20160402T034217Z.7ddc219970f6b6c9.323a12de2c42e5f2bf29acccc5bbd76682a610d2";
//var translate = require('yandex-translate')(key);
app.use(express.static('public'));
var defaultNsps = '/';
//console.log(uuid());
//app.use(useragent.express());
app.get('/', function(req, res){

	res.sendFile(__dirname + '/public/html/home.html');
});

app.get('/lang', function(req, res){
	res.sendFile(__dirname + '/public/html/lang.html');
});

app.get('/signup', function(req, res){
	res.sendFile(__dirname + '/public/html/signup.html');
});

app.get('/test', function(req, res){
	res.sendFile(__dirname + '/public/html/test.html');
});

app.get('/signin', function(req, res){
	res.sendFile(__dirname + '/public/html/signin.html');
});

/*
app.get('/chat', function(req, res){
	if(req.useragent.isMobile)
	{
		res.sendFile(__dirname + '/public/html/mchat.html');
	}
	else
	{
		res.sendFile(__dirname + '/public/html/chat.html');
	}
});
*/
app.get('/aboutus', function(req, res){
	res.sendFile(__dirname + '/public/html/aboutus.html');
});

app.get('/wait', function(req, res){
	res.sendFile(__dirname + '/public/html/wait.html');
});

app.get('/forgotten', function(req, res){
	res.sendFile(__dirname + '/public/html/forgotten.html');
});
/*
io.sockets.on('connection', function(socket){
	console.log('a user connected');
	var inwaiting = false;
	socket.on('waiting', function(data) {
		if(!inwaiting)
		{
			console.log('in here');
			var waiters = io.nsps[defaultNsps].adapter.rooms['waitingroom'];
			if(waiters)
			{
				var roomname = uuid();
				socket.join('waitingroom');
				io.sockets.in('waitingroom').emit('found', {room: roomname});
			}
			else{
				socket.join('waitingroom');
				inwaiting=true;
			}
		}
	});
	socket.on('stop', function(data){
		socket.leave('waitingroom');
		console.log('leaving room');
	})
	socket.on('join', function(data){
		socket.join(data.room);
		console.log('joining room');
	});
	socket.on('sentmessage', function(data){
		console.log(data.room);
		io.sockets.in(data.room).emit('gotmessage', {name:data.name, room:data.room, message:data.message});
	});
	
	socket.on('translate', function(data){
		console.log('trans');
		language = data.lang;
		translate.translate(data.message,{to: language}, function(err, res){
			console.log(err);
			console.log(res.text);
			socket.emit('translated', {name: data.name, message: res.text,room: data.room});
		});
	});	
});
*/
server.listen(port, function(){
  console.log('listening on port: '+ port);
});
