int:  n;
array[1..n] of var 1..n: order;

include "alldifferent.mzn";

constraint alldifferent(order);

constraint forall(i in 1..n-1, j in 1..n-1 where i < j)(abs(order[i+1] - order[i]) != abs(order[j+1] - order[j]));