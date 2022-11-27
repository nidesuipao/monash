% strains.mzn
enum VACCINE;  % set of VACCINES available
VACCINE: placebo; % placebo vaccine
constraint forall(v in VACCINE)
                 (assert(placebo <= v, "vaccine \(v) appears before placebo\n"));

int: N; % number of test populations
set of int: POP = 1..N;

int: W; % number of weeks
set of int: WEEK = 1..W;

enum TREATMENT = { WAIT, VAX, PCR, RAT, SAT };
array[TREATMENT] of int: cost;
array[TREATMENT] of int: mintreat;
array[TREATMENT] of int: maxtreat;
int: maxvax;
var int: allcost;

array[POP,WEEK] of var TREATMENT: schedule;
array[POP] of var set of VACCINE: vaccinations;


constraint forall(p in POP)(sum(w in WEEK)(schedule[p, w] = VAX) = 2 * card(vaccinations[p]));
constraint forall(p1, p2 in  POP where p1 < p2 )(vaccinations[p1] != vaccinations[p2]);
constraint sum(p in POP)(card(vaccinations[p]) = 0) = 1;
constraint sum(p in POP)(card(vaccinations[p]) = 1 /\ placebo in vaccinations[p]) = 1;
constraint forall(p in POP, w in 1..W-1)(schedule[p, w] = VAX -> schedule[p, w+1] = WAIT);
constraint forall(p in POP, w1 in WEEK)(schedule[p, w1] = PCR -> sum(w2 in 1..w1)(schedule[p, w2] = RAT) >= 1);
constraint forall(p in POP)(schedule[p, 1] != WAIT);
constraint forall(w in WEEK)(sum(p in POP)(schedule[p, w] = VAX) <= maxvax);

include "globals.mzn";
constraint forall(p in POP)(global_cardinality([schedule[p, w] | w in WEEK], [WAIT, VAX, PCR, RAT, SAT], mintreat, maxtreat));


array[POP] of var int: pcost;
constraint forall(p in POP)(pcost[p] = sum(w in WEEK)(cost[schedule[p, w]]));
constraint allcost = sum(p in POP)(pcost[p]);
constraint forall(p1, p2 in POP where p1 < p2)(pcost[p1] >= pcost[p2]);
constraint forall(p1, p2 in POP where p1 < p2)(pcost[p1] <= 2 * pcost[p2] /\ 2 * pcost[p1] >=  pcost[p2]);
% constraint forall(p1, p2 in POP where p1 < p2)(2 * sum(w in WEEK)(cost[schedule[p1, w]]) >= sum(w in WEEK)(cost[schedule[p2, w]]));

enum virus = {ALPHA, BETA, OMICRON, MU};
array[TREATMENT, virus] of float: fp = [|1, 1, 1, 1
                                        |1, 1, 1, 1
                                        |0.02, 0.03, 0.05, 0.3 
                                        |0.1, 0.12, 0.2, 0.16 
                                        |0.03, 0.04, 0.08, 0.1|];

opt bool: stageD;


                                    
% constraint occurs(stageD) -> forall(p in POP, v in virus)(product(w in WEEK)(fp[schedule[p, w],v]) <= 0.01);  
                                   
constraint occurs(stageD) ->
      forall(p in POP, v in virus)(sum(w in WEEK)(if (schedule[p, w] = RAT) then log2(fp[RAT, v]) 
                                      elseif (schedule[p, w] = PCR) then log2(fp[PCR, v])
                                      elseif (schedule[p, w] = SAT) then log2(fp[SAT, v]) else 0 endif) <= log2(0.01));
                                        




     
      
solve minimize allcost;


