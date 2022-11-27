int: T; % maximum time allowed;
set of int: TIME = 1..T;
int: G; % number of guards
int: F; 
int: W;

array[TIME] of var -F..F: move_food;      % move food +ve bank to boat, -ve boat to bank
array[TIME] of var -W..W: move_water;     % move water +ve bank to boat, -ve boat to bank
var TIME: end;                            % end time;

constraint end = max(i in TIME where move_food[i] != 0 \/ move_water[i] != 0)(i);

array[0..T] of var 0..F:food_on_boat;
array[0..T] of var 0..W:water_on_boat;

array[0..T] of var 0..F:food_on_bank;
array[0..T] of var 0..W:water_on_bank;
array[0..T] of var 0..1:ele;

constraint food_on_boat[0] = 0;
constraint water_on_boat[0] = 0;
constraint food_on_bank[0] = F;
constraint water_on_bank[0] = W;
constraint ele[0] = 0;

constraint forall(t in 1..T)(food_on_boat[t] = food_on_boat[t-1] + move_food[t]);
constraint forall(t in 1..T)(water_on_boat[t] = water_on_boat[t-1] + move_water[t]);
constraint forall(t in 1..T)(food_on_bank[t] = food_on_bank[t-1] - move_food[t]);
constraint forall(t in 1..T)(water_on_bank[t] = water_on_bank[t-1] - move_water[t]);

constraint food_on_bank[end] = F;
constraint water_on_bank[end] = W;


constraint forall(t in 1..end-1)(ele[t] = if yess(food_on_boat[t],food_on_bank[t],water_on_boat[t],water_on_bank[t]) then 1 
                             elseif nooo(food_on_boat[t],food_on_bank[t],water_on_boat[t],water_on_bank[t]) then 0
                             else ele[t-1] endif);
                             
constraint ele[end-1] = 1;

constraint forall(t in 1..end-1)(if ele[t] = 0 then not noo(food_on_bank[t],water_on_bank[t])
                                 else not noo(food_on_boat[t],water_on_boat[t])endif);
                             
% constraint sum(t in 1..T)(yess(food_on_boat[t],food_on_bank[t],water_on_boat[t],water_on_bank[t])) >= 1;
% constraint sum(t in 1..T)(nooo(food_on_boat[t],food_on_bank[t],water_on_boat[t],water_on_bank[t])) = 0;

function var bool: yess(var 0..F: f1, var 0..F: f2, var 0..W: w1, var 0..W: w2) = (f1 = w1) /\ (f1 > f2) /\ (w1 > w2);
function var bool: nooo(var 0..F: f1, var 0..F: f2, var 0..W: w1, var 0..W: w2) = (f2 = w2) /\ (f1 < f2) /\ (w1 < w2);

predicate noo(var 0..F: f1, var 0..F: w2) = if f1 + w2 > 0 then f1*w2 = 0 else false endif;

constraint forall(t in TIME)(abs(move_food[t]) + abs(move_water[t]) <= G);

solve minimize end;

