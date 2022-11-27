enum TOPIC;
int: n = card(TOPIC);
array[TOPIC] of int: hours;
int: m; % precedences
set of int: PREC = 1..m;
array[PREC] of TOPIC: before;
array[PREC] of TOPIC: after;
set of TOPIC: morning;
int: D; 
int: H;
constraint assert(H mod 2 = 0,"Hours per day H must be even\n");
set of int: DAY = 1..D;
set of DAY: HALF;

set of int: HOUR = 0..H-1;
array[TOPIC] of var DAY: day;
array[TOPIC] of var HOUR: hour;

solve maximize min([day[t] * H + hour[t]|t in TOPIC]);
constraint forall(i in PREC)(day[after[i]] * H + hour[after[i]] > day[before[i]] * H + hour[before[i]]);
% constraint forall(t1,t2 in TOPIC where t1 < t2)(fuc(day[t1], day[t2], hour[t1], hour[t2], hours[t1], hours[t2]));

% predicate fuc(var int:d1, var int: d2, var int: h1, var int: h2, var int: l1, var int: l2) 
% = d1 * H + h1 >= d2 * H + h2 + l2 \/ d1 * H + h1 + l1 <= d2 * H + h2;

include "diffn.mzn";
constraint diffn(day, hour, [1 | t in TOPIC], hours);

constraint forall(t in TOPIC)(hour[t] + hours[t] <= H);

constraint forall(d in HALF)(
forall(t in TOPIC where day[t] = d)(hour[t] + hours[t] <= H div 2));

constraint forall(t in morning)(hour[t] = 0);

/*** debug output
array[1..20] of string: key = ["*","+","$","%","@","#","&","=","-",":","a","b","c","d","e","f","g","h","i","j"];
output [ "\(TOPIC[i]) \"" ++ key[i] ++ "\", " | i in 1..n ] ++ ["HALF \".\"\n"];
output [ if h = 0 then "day " ++ show_int(2,d) ++ " " else "" endif ++
         let { int: i = (d-1)*H + h; 
               int: j = arg_max([ fix(start[t]) <= i /\ fix(start[t]) + hours[t] > i | t in TOPIC ] ++ [ true]); } in
         if j <= n then key[j] 
         elseif d in HALF /\ h >= H div 2 then "."
         else " " endif ++
         if h = H-1 then "\n" else "" endif 
       | d in DAY, h in HOUR ];   
****/

%% output for grading       
output ["day = \(day);\nhour = \(hour);\n"];       
       
       