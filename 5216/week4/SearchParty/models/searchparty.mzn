enum DIRN = {n,e,s,w};

var DIRN: A;
var DIRN: B;
var DIRN: C;
var DIRN: D;
var DIRN: E;
var DIRN: F;
array[1..6] of var DIRN: a = [A, B, C, D, E, F];

function var int: countx(array[1..6] of var DIRN: a) 
      = (sum(i in 1..6)(a[i] = n) >=1 /\ sum(i in 1..6)(a[i] = e) >=1 /\ sum(i in 1..6)(a[i] = s) >=1 /\ sum(i in 1..6)(a[i] = w) >=1);


constraint countx(a) = 1;

constraint if(A = n \/ A = s) then B != n /\ B != s endif;

constraint (C = n /\ D = s) \/ (C = s /\ D = n) \/ (C = w /\ D = e) \/ (C = e /\ D = w);

constraint if(E != w \/ F != w) then A != w /\ B != w /\C != w /\D != w endif;

constraint A != E /\ B != F /\ C != E;