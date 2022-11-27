var -2..3: x;
var 0..5: y;
constraint x * (3*y + x - 4) + 4 * y <= 4;
solve satisfy;