int: n;                % number of points
set of int: P = 1..n;  % points 
array[P,P] of int: d;  % distance matrix
array[P] of int: v;    % value matrix
int: k;                % size limit for chosen set 
int: l;                % average distance lower bound 

                                  
var set of P: S;       % chosen set S
var int: value;        % objective value of chosen set
var int: len = sum(p in S)(1);
constraint sum(i, j in S where i > j)(d[i,j]) div (card(S) * (card(S)-1) div 2) >= l;


constraint len <= k;

value = sum(p in S)(v[p]);

solve maximize value;    
         
% output ["\(S));\n"];