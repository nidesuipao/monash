set of 1..8: num = 1..8;

array[num] of var 1..100:s;

include "alldifferent.mzn";

constraint alldifferent(s);
constraint let {var int: m = max(s)} in  s[1] = m;

solve maximize sum(i in num)(s[i]);