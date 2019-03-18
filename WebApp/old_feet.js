// Where is the circle
let x, y;

function setup() {
  createCanvas(720, 400);
  // Starts in the middle
  x = width / 2;
  y = height;
}

function draw() {
  background(200);
  
  // HEEL OF THE FOOT
  stroke(139,69,19);
  fill(160,82,45);
  arc(x, y-40, 14, 24, 0, PI);
	line(x-7, y-40, x+7, y-40);
	
	// FOOT
  stroke(139,69,19);
  fill(160,82,45);
	bezier(x-7, y-45, x-20, y-90, x+20, y-90, x+7, y-45);
	// arc(x, y-45, 22, 65, PI, 0);
	line(x-7, y-45, x+7, y-45);
  
	// HEEL OF THE FOOT
  stroke(139,69,19);
  fill(160,82,45);
  arc(x+35, y-50, 14, 24, 0, PI);
	line(x+28, y-50, x+42, y-50);
	
	// FOOT
  stroke(139,69,19);
  fill(160,82,45);
	bezier(x+28, y-55, x+15, y-100, x+55, y-100, x+42, y-55);
	// arc(x, y-45, 22, 65, PI, 0);
	line(x+28, y-55, x+42, y-55);
	
//   // Jiggling randomly on the horizontal axis
//   x = x + random(-1, 1);
//   // Moving up at a constant speed
//   y = y - 1;
  
//   // Reset to the bottom
//   if (y < 0) {
//     y = height;
//   }
}