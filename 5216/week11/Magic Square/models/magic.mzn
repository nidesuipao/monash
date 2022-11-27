% Use this editor as a MiniZinc scratch book
int: n;
set of int: ROW = 1..n;
set of int: COL = 1..n;
set of int: NUM = 1..n*n;
array[ROW,COL] of var NUM: x;

%% pretty output: comment out for grading
int: d = ceil(log(10,n*n));
output [ show_int(d,x[r,c]) ++ if c = n then "\n" else " " endif 
       | r in ROW, c in COL ];


var n..sum(i in 1..n)(n*n + 1 - i): same;
       
include "alldifferent.mzn";
constraint alldifferent(x);
constraint forall(i in ROW)(sum(j in COL)(x[i,j]) = same);
constraint forall(i in COL)(sum(j in ROW)(x[j,i]) = same);
constraint sum(i in ROW)(x[i,i]) = same;
constraint sum(i in ROW)(x[i,n + 1 - i]) = same;
constraint lex_lesseq([ x[j,i] | i,j in 1..n ], [ x[i,j] | i,j in 1..n ]);
       
%% grader_output 
output ["x = array2d(ROW,COL,\(x));\n"];     
output ["same\(same));\n"];     