int: n;                % number of points
set of int: P = 1..n;  % points 
array[P,P] of int: d;  % distance matrix
array[P] of int: v;    % value matrix
int: k;                % size limit for chosen set 
int: l;                % average distance lower bound 

set of int: P0 = 0..n;
array[1..k] of var P0: Sx;

var int: len = sum(i in Sx where i != 0)(1);

constraint len <= k;
constraint sum(i, j in Sx where i != 0 /\ j != 0 /\ i > j)(d[i,j]) div (len * (len-1) div 2) >= l;

include "alldifferent.mzn";
constraint (alldifferent(i in Sx where i != 0)(i));

var int: value;
value = sum(i in Sx where i != 0)(v[i]);
solve maximize value;

% output ["\(Sx));\n"];
                         
                                      
