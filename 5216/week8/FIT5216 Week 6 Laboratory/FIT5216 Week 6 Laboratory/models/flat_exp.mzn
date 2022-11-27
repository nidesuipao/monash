var -2..3: x;
var 0..5: y;
var -1..2: u;
var 0..3: v;
constraint (x - u)*(x - u) + (y - v)*(y - v) >= 15;
constraint (x- u)*(y - v) >= 0;
solve :: int_search([x,y,u,v],input_order,indomain_min,complete)
      satisfy;
