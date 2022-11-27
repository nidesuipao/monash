enum PERSON;
enum GOOD;
array[GOOD] of int: available; % number of each good available
array[GOOD] of int: value;
array[PERSON] of int: npref;   % number of preferences for each person
int: tpref = sum(npref);
set of int: PREF = 1..tpref;
array[PREF] of GOOD: good_pref; % good for each preference
array[PREF] of int: req_pref;   % number required for each preference

int: maxreq = max(req_pref);
array[PERSON] of var GOOD: good;          % which good the person gets
array[PERSON] of var 0..maxreq: num;      % how many of good the person gets
array[PERSON] of var 1..max(npref): pref; % which of their preferences the person gets

array[PERSON] of int: prev_pref = [ sum(q in PERSON where q < p)
                                       (npref[q]) | p in PERSON ];
                                       
var int: total_value;
constraint total_value = sum(p in PERSON)(num[p] * value[good[p]]);

% require each person to get one of their preferences
constraint forall(p in PERSON)
                 (pref[p] <= npref[p] /\
                  good[p] = good_pref[prev_pref[p] + pref[p]] /\
                  num[p] = req_pref[prev_pref[p] + pref[p]]);


constraint forall(g in GOOD)(sum(p in PERSON where good[p] = g)(num[p]) <= available[g]);

array[PERSON, GOOD] of 0..max(npref): rank = 
array2d(PERSON, GOOD,[sum( i in 1..npref[a] )( i*bool2int(good_pref[prev_pref[a] + i] = b))|a in PERSON, b in GOOD]);

array[PERSON, GOOD] of 0..max(req_pref): good_req = 
array2d(PERSON, GOOD,[sum( i in 1..npref[a] )( req_pref[prev_pref[a] + i]*bool2int(good_pref[prev_pref[a] + i] = b))|a in PERSON, b in GOOD]);


constraint forall(p1, p2 in PERSON where p1 < p2 /\ rank[p1, good[p2]] != 0 /\ rank[p2, good[p1]] != 0)
(rank[p1, good[p1]] < rank[p1, good[p2]] \/ rank[p2, good[p2]] < rank[p2, good[p1]] \/ stable(p1,p2,good[p1],good[p2]));

predicate stable(var PERSON:p1, var PERSON:p2, var GOOD:g1, var GOOD:g2) =
sum(p in PERSON where good[p] = g1)(num[p]) - good_req[p1,g1] + good_req[p2,g1]  > available[g1] \/
sum(p in PERSON where good[p] = g2)(num[p]) - good_req[p2,g2] + good_req[p1,g2] > available[g2];
                  
solve minimize total_value;

% output["\(rank)\n"];
% output["\(good_req)"];


