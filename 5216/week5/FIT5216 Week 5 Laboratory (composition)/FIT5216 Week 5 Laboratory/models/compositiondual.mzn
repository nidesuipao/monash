int:  n;
array[1..n] of var 1..n: order;
array[1..n-1] of var 1..n: diffs;
array[1..n] of var 1..n: posn;
array[1..n-1] of var 1..n: dposn;

include "alldifferent.mzn";

constraint alldifferent(order);

% constraint forall(i in 1..n-1, j in 1..n-1 where i < j)(abs(order[i+1] - order[i]) != abs(order[j+1] - order[j]));
constraint forall(i in 1..n-1)(diffs[i] = abs(order[i+1] - order[i]));
constraint alldifferent(diffs);
constraint forall(i in 1..n)(posn[order[i]] = i);
% constraint abs(posn[1] - posn[n]) = 1;
constraint forall(i in 1..n-1)(dposn[diffs[i]] = i);
constraint dposn[n-1] = min(posn[1], posn[n]);