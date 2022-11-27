% Use this editor as a MiniZinc scratch book
int: n;
set of int: NODE= 1..n;
array[int,1..2] of NODE: edge;
set of int: COLOR = 1..card(NODE);
array[NODE] of var COLOR: x;

constraint forall(i in index_set_1of2(edge))(x[edge[i,1]] != x[edge[i,2]]);

var int: minc = min(x);
solve minimize n + 1 - minc;
output["\(x)\n"];

output["\(n + 1 - minc)"];


