% Building a blast plan
int: W; % width of area
set of int: COL = 1..W;
int: H; % height of area
set of int: ROW = 1..H;

array[ROW,COL] of int: cost;   % cost to blast in spot
array[ROW,COL] of int: reward; % reward for mining here
array[ROW,COL] of var 0..1: blast;

int: budget;  
int: limit; % max number of blasts;


array[1..limit] of var 0..H: y;
array[1..limit] of var 0..W: x;


var int: spend;
var int: huibao;

constraint forall(i,j in 1..limit where i < j )(x[i] >= x[j]);
constraint forall(i,j in 1..limit where i < j /\ x[i] <= x[j])(y[i] >= y[j]);
constraint forall(i,j in 1..limit where i < j /\ x[i] != 0)(abs(x[i] - x[j]) + abs(y[i] - y[j]) >= 3 );
constraint spend <= budget;
constraint spend = sum(i in 1..limit where x[i] != 0 )(cost[y[i], x[i]]);

constraint forall(i in 1..limit where x[i] = 0 \/ y[i] = 0)(y[i] = x[i]);
constraint forall(i in 1..limit where x[i] != 0)(reward[y[i], x[i]] >= 0);

% constraint forall(i in 1..limit where x[i] != 0 /\ y[i] != 0)(
%                    forall(j,k in -1..1)(
%                           if x[i] + j >= 1 /\ x[i] + j <= W /\ y[i] + k >= 1 /\ y[i] + k <= H then
%                               blast[y[i] + k, x[i] + j ] = 1
%                           endif
%                           ));

function var int: count(var int: j, var int: i, array[1..limit] of var 0..H: y, array[1..limit] of var 0..W: x) 
                      = sum(k in 1..limit)(j = y[k]-1 /\ i = x[k]-1 /\ x[k] != 0) =1  \/
                        sum(k in 1..limit)(j = y[k]-1 /\ i = x[k] /\ x[k] != 0) = 1 \/
                        sum(k in 1..limit)(j = y[k]-1 /\ i = x[k]+1 /\ x[k] != 0) = 1 \/ 
                        sum(k in 1..limit)(j = y[k] /\ i = x[k]-1/\ x[k] != 0) = 1 \/
                        sum(k in 1..limit)(j = y[k] /\ i = x[k]/\ x[k] != 0) = 1 \/ 
                        sum(k in 1..limit)(j = y[k] /\ i = x[k]+1/\ x[k] != 0) = 1 \/
                        sum(k in 1..limit)(j = y[k]+1 /\ i = x[k]-1/\ x[k] != 0) = 1 \/
                        sum(k in 1..limit)(j = y[k]+1 /\ i = x[k]/\ x[k] != 0) = 1 \/
                        sum(k in 1..limit)(j = y[k]+1 /\ i = x[k]+1/\ x[k] != 0) =1;
                      
constraint forall(i in COL, j in ROW)(
                  if count(j,i,y,x) = 1 then
                        blast[j, i] = 1
                  else
                        blast[j, i] = 0
                  endif
                        );

constraint huibao = sum(i in COL, j in ROW where blast[j,i] = 1)(reward[j,i]);
                  

var int: profit;
constraint profit = huibao - spend;
solve maximize profit;