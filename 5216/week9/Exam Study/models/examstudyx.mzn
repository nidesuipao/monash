enum TOPIC;
array[TOPIC] of int: hours;
int: m; % precedences
set of int: PREC = 1..m;
array[PREC] of TOPIC: before;
array[PREC] of TOPIC: after;
set of TOPIC: morning;
int: D; 
int: H;
set of int: DAY = 1..D;
set of DAY: HALF;

set of int: TIME = 0..D*H;
array[TOPIC] of var TIME: start;

solve maximize min(start);
constraint forall(i in PREC)(start[after[i]] > start[before[i]]);
constraint forall(t in TOPIC)(start[t] + hours[t] <= D*H );

include "disjunctive.mzn";
constraint disjunctive(start, hours);

constraint forall(t in morning)(start[t] mod H = 0);
constraint forall(d in HALF)(forall(t in TOPIC)(start[t] + hours[t] <= H * d - H div 2  \/ start[t] >= H * d));
constraint forall(t in TOPIC)((start[t] + hours[t]) <= start[t] div H * H + H);

/*** debug output ***
array[1..20] of string: key = ["*","+","$","%","@","#","&","=","-",":","a","b","c","d","e","f","g","h","i","j"];
output [ "\(TOPIC[i]) \"" ++ key[i] ++ "\", " | i in 1..n ] ++ ["HALF \".\"\n"];
output [ "| day " ++ show_int(6-H,d) | d in DAY ] ++ ["|\n"]; 
output [ let { int: j = arg_max([ fix(start[t]) <= i /\ fix(start[t]) + hours[t] > i | t in TOPIC ] ++ [ true]); } in
         if j <= card(TOPIC) then key[j] 
         elseif (i div H) + 1 in HALF /\ i mod H >= H div 2 then "."
         else " " endif
       | i in TIME ];  
*** end debug output ***/        
       
/*** grading output ***/    
  
output ["\nstart = \(start);\n"];     