enum ACTION = { LAND, INSPECT, FULLSERVICE, RECHARGE, PACK, TAKEOFF };
array[ACTION] of int: resources; % number of machines for each service

int: horizon;  % end time of planning horizon
set of int: TIME = 0..horizon;

int: norders;                    % number of orders;
set of int: ORDER = 1..norders; 
array[ORDER] of int:  dist;      % distance warehouse to delivery         
enum WC = { ULTRA, LIGHT, MEDIUM, HEAVY };         
array[ORDER] of WC:  weight;     % weight catgeory of order
array[ORDER] of TIME: available; % when the order can be packed
array[ORDER] of int:  value;     % value of order
int: ndrones;
set of int: DRONE = 1..ndrones;
array[DRONE] of TIME: arrival;  % when they arrive back
array[DRONE] of int: charge;    % how much charge they have left
array[WC] of int: packtime;
array[1..6] of var ACTION: point={LAND, INSPECT, FULLSERVICE, RECHARGE, PACK, TAKEOFF};

var int: total_value;

/* Decisions */
array[DRONE, ACTION] of var TIME: start;               % start time for each action  
array[DRONE, ACTION] of var 0..max(resources): resource;  % resource used for each action 
array[DRONE] of var 0..norders: order; % which order given to drone
array[DRONE] of var bool: fastcharge;


array[DRONE, ACTION] of var int: lasting;
constraint forall(d in DRONE)(start[d, LAND] >= arrival[d]);
constraint forall(d in DRONE, a in 2..6)(start[d,point[a]] >= start[d, point[a-1]] + lasting[d, point[a-1]]);
constraint forall(d in DRONE where order[d] != 0)(forall(a in 4..6)
(start[d,point[a]] >= start[d,point[a-1]] + lasting[d,point[a-1]]));

%stage A
constraint forall(d in DRONE, a in ACTION where order[d] != 0)(resource[d,a] != 0);
constraint forall(d in DRONE where order[d] = 0)(forall(a in ACTION where a != LAND /\ a != INSPECT)
(resource[d,a] = 0));
constraint forall(d in DRONE where order[d] = 0)(forall(a in ACTION where a = LAND \/ a = INSPECT)
(resource[d,a] != 0));
constraint forall(d in DRONE)(forall(a in ACTION)(resource[d,a]<=resources[a]));

%no overuse
constraint forall(d1, d2 in DRONE, a in ACTION where d1 != d2 /\ resource[d1,a] = resource[d2,a])
((start[d1,a] + lasting[d1,a] <= start[d2,a]) \/ (start[d2,a] + lasting[d2,a] <= start[d1,a]));

constraint forall(d1, d2 in DRONE where order[d1] != 0 /\ order[d2] != 0 /\ d1 < d2)
(order[d1] != order[d2]);
constraint forall(d in DRONE where order[d] != 0)(start[d,PACK] >= available[order[d]]);

% calculate the lasting
constraint forall(d in DRONE,a in ACTION)
( if order[d] != 0 /\ a = FULLSERVICE /\ charge[d] = 0 /\ (weight[order[d]] = MEDIUM \/  
  weight[order[d]] = HEAVY) then lasting[d, a] = 10
  elseif a = FULLSERVICE then lasting[d,a] = 0 
  elseif order[d] != 0 /\ charge[d] >= 10 /\ a = RECHARGE then lasting[d,a] = 0
  elseif order[d] != 0 /\ charge[d] >= 5 /\ dist[order[d]] <=50 /\ a = RECHARGE then lasting[d,a] = 0
  elseif order[d] != 0 /\ a = RECHARGE /\ fastcharge[d] = false then lasting[d,a] = 4
  elseif order[d] != 0 /\ a = PACK then lasting[d, a] = packtime[weight[order[d]]]
  elseif order[d] = 0 /\ a != LAND then lasting[d,a] = 1
  elseif order[d] = 0 /\ a != INSPECT then lasting[d,a] = 1
  elseif order[d] = 0 then lasting[d,a] = 0
  else lasting[d,a] = 1
endif);
%change
constraint forall(d1,d2 in DRONE where d1 != d2)
(resource[d1,RECHARGE] = resource[d2,RECHARGE] \/resource[d1,PACK]!=resource[d2,PACK]\/ start[d2,PACK]-start[d1,PACK]>=5);

% constraint forall(d1 in DRONE, 
% d2 = max(d3 in DRONE where start[d3, PACK] <= start[d1, PACK] /\ resource[d3,PACK] = resource[d1,PACK])(d3))
% (resource[d1,RECHARGE] != resource[d2,RECHARGE] -> start[d1, PACK] >= start[d2,PACK] + 5);

%stage F
include "lex_lesseq.mzn";
% lex_lesseq
constraint forall(t in TIME)
(lex_lesseq([order[d] | d in DRONE where arrival[d] = t /\ charge[d] < 5],[order[d] | d in reverse(DRONE) where arrival[d] = t /\ charge[d] < 5]));
constraint forall(t in TIME)
(lex_lesseq([order[d] | d in DRONE where arrival[d] = t /\ charge[d] >= 5 /\ charge[d] < 10],[order[d] | d in reverse(DRONE) where arrival[d] = t /\ charge[d] >= 5 /\ charge[d] < 10]));
constraint forall(t in TIME)
(lex_lesseq([order[d] | d in DRONE where arrival[d] = t /\ charge[d] = 5],[order[d] | d in reverse(DRONE) where arrival[d] = t /\ charge[d] = 5]));

% constraint forall(o in ORDER)
% (lex_lesseq([order[d] | d in DRONE where arrival[d] <= available[o]-6 /\ charge[d] < 5],[order[d] | d in reverse(DRONE) where arrival[d] <= available[o] /\ charge[d] < 5]));
% constraint forall(o in ORDER)
% (lex_lesseq([order[d] | d in DRONE where arrival[d] <= available[o]-6 /\ charge[d] >= 5 /\ charge[d] < 10],[order[d] | d in reverse(DRONE) where arrival[d] <= available[o] /\ charge[d] >= 5 /\ charge[d] < 10]));
% constraint forall(o in ORDER)
% (lex_lesseq([order[d] | d in DRONE where arrival[d] <= available[o]-6 /\ charge[d] = 5],[order[d] | d in reverse(DRONE) where arrival[d] <= available[o] /\ charge[d] = 5]));
% constraint forall(o in ORDER)
% (lex_lesseq([order[d] | d in DRONE where arrival[d] <= available[o]-6 /\ charge[d] >= 5 /\ dist[o] <= 50],[order[d] | d in reverse(DRONE) where arrival[d] <= available[o] /\ charge[d] >= 5 /\ dist[o] <= 50]));


%Dominance
constraint forall(a in ACTION where resources[a] > 1)
(forall(r1,r2 in 1..resources[a] where r1 < r2)
(sum(d in DRONE)(resource[d,a] = r1) <= 
sum(d in DRONE)(resource[d,a] = r2)));



%calculate the output
var int: cost = sum(d in DRONE where fastcharge[d] = true /\ order[d] != 0)(5);
var int: total_time = sum(d in DRONE where order[d] != 0)(start[d, TAKEOFF] - arrival[d]);
constraint total_value = sum(d in DRONE where order[d] != 0)(value[order[d]]) - cost;
% solve maximize (100*total_value - total_time);
var int: de;
constraint de=100*(total_value)-total_time;
solve maximize(de);


