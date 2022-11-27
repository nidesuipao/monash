array[-1..1,-2..1] of var -1..1:x;
constraint forall(i in -1..1)(x[i,i] >= 1);
constraint sum(i in -2..1)(x[0,i]) = 0;
constraint x[x[1,1],x[0,0]] = 1;
constraint forall(i in -1..1)(x[i,-2] = 0);
solve :: int_search([x[i,j] | i in -1..1, j in -2..1],
                    input_order, indomain_min, complete)
      satisfy;
