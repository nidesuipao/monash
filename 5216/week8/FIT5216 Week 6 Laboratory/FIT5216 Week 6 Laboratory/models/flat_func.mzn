function var int: myabs(var int: x) = 
                  let { var int: y; 
                        constraint y >= 0;
                        constraint y = x \/ y = -x; } in
                  y;

array[1..3] of var -2..2: x;
constraint sum(i in 1..3)(myabs(x[i])) <= 2;

solve :: int_search(x,input_order,indomain_min,complete)
	satisfy;
