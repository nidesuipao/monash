% hidato puzzle

int: n; % no of ROWS
set of int: ROW = 1..n;
int: m; % no of COLS
set of int: COL = 1..m;

set of int: CLUE = 0..m*n;
array[ROW,COL] of CLUE: clue;

set of int: NUM = 1..m*n;
array[ROW,COL] of var NUM: x;
  
%% output for debugging                
% int: d = ceil(log10(m*n+1));
% output [ show_int(d,x[r,c]) ++ " " ++ if c = m then "\n" else "" endif
%        | r in ROW, c in COL ];                 

%% output for grading
output ["x = array2d(ROW,COL,\(x));\n"];
include "alldifferent.mzn";
constraint alldifferent(x);

function var int: adjacent(var int :i1, var int: i2, var int: j1, var int: j2) = if abs(i1 - i2) + abs(j1 - j2) = 1 \/ (abs(i1-i2) = 1 /\ abs(j1-j2) = 1)
                                                                then 1 
                                                             else 0 
                                                             endif;

% constraint forall(num in 1..m*n-1, i1,i2 in ROW, j1,j2 in COL where x[i1,j1] = num /\ x[i2,j2] = num+1)(adjacent(i1,i2,j1,j2) = 1);

array[NUM] of var 1..n: rows;
array[NUM] of var 1..m: cols;

constraint forall(i in ROW, j in COL)(rows[x[i,j]] = i /\ cols[x[i,j]] = j);
constraint forall(num in 1..m*n-1)(adjacent(rows[num], rows[num+1], cols[num], cols[num+1]) = 1);

% constraint forall(i1 in ROW, j1 in COL where x[i1,j1] != m*n)(sum(mx, nx in -1..1 where 1 <=i1 + mx /\
%                                                                                         n >=i1 + mx /\
%                                                                                         1 <=j1 + nx /\
%                                                                                         m >=j1 + nx)(x[i1+mx,j1+nx] = x[i1,j1] + 1) >= 1);
                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                     
constraint forall(i in ROW, j in COL where clue[i,j] != 0)(x[i,j] = clue[i,j]);                                                                                                        
                                                                                                                                                                                                       