% poetry
% place k stanzas of E type and l stanzas of S type and m stanzas of O type
% in order such that
%    no two stanzas of E or S are adjacent
% maximize the sum of differences in effectivenesss of 
% adjacent stanzas of types E and S
int: k;
int: l;
int: m;
set of int: ESSTANZA = 1..k+l;
int: other = 0;
set of int: STANZA = 0..k+l;  % other stanzas are not important

enum KIND = { E, S, O };
array[1..k+l+1] of KIND: kind = [O] ++ [E | i in 1..k ] ++ [ S | i in 1..l ];

array[ESSTANZA] of int: effect;

int: n = k+l+m;
set of int: POSN = 1..n;

array[ESSTANZA] of var POSN:  place; % place of each E and S stanza
array[POSN] of var STANZA: what;     % what is in each posn, 0 = other stanza

% symmetry breaking: This constraint is guaranteed to be correct
% Make sure an E or S stanza is in the first position, 
% If there are two other stanzas in a row this has to happen at the end. 
constraint what[1] != other /\ 
           forall(i in 1..n-1)
                 (what[i] = other /\ what[i+1] = other 
                  -> forall(j in i..n)(what[j] = other));

output ["obj = \(obj);\nwhat = \(what);\nplace = \(place);\nkind = \(kind);\n"];% poetry

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ALL MODELLING ERRORS ARE BELOW THIS LINE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

constraint forall(i in ESSTANZA, j in POSN)(place[i] = j <-> what[j] = i);

include "alldifferent.mzn";
constraint alldifferent([p | p in what where p != 0]);
% constraint sum(p in what)(p != 0) = k+l;

constraint forall(p in 1..n-1)(kind[what[p]+1] = E -> kind[what[p+1]+1] != E);
constraint forall(p in 1..n-1)(kind[what[p]+1] = S -> kind[what[p+1]+1] != S);

% constraint forall(p1, p2 in ESSTANZA where p1 < p2 /\ what[p1] != 0 /\ what[p2] != 0)
%                  (kind[p1] = kind[p2] -> abs(place[p1] - place[p2]) > 1);


var int: obj = sum(p in 1..n-1 where what[p]!=0 /\ what[p+1]!=0)(abs(effect[what[p]] - effect[what[p+1]]));

solve maximize obj;
