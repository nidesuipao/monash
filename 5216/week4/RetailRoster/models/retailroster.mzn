enum SHIFT = { opening, morning, lunch, afternoon, closing };
enum DAY = { MON, TUE, WED, THU, FRI, SAT, SUN };
enum PERSON;
array[PERSON,DAY,SHIFT] of int: pref;
array[PERSON,DAY] of var set of SHIFT: r;

constraint forall(p in PERSON, d in DAY)(card(r[p,d]) <= 2);
constraint forall(p in PERSON, d in DAY)(if(closing in r[p,d]) then not(afternoon in r[p,d]) endif);
constraint forall(p in PERSON, d in DAY)(if(morning in r[p,d]) then not(opening in r[p,d]) endif);
constraint forall(p in PERSON, d in DAY)(if(closing in r[p,d]) then not(opening in r[p,d]) endif);

constraint forall(d in DAY)(sum(p in PERSON)(opening in r[p,d]) = 1);
constraint forall(d in DAY)(sum(p in PERSON)(closing in r[p,d]) = 1);
constraint forall(d in DAY)(sum(p in PERSON)(morning in r[p,d]) >= 2);
constraint forall(d in DAY)(sum(p in PERSON)(lunch in r[p,d]) >= 2);
constraint forall(d in DAY)(sum(p in PERSON)(afternoon in r[p,d]) >= 3);

constraint forall(p in PERSON, d in DAY, s in SHIFT)(if(pref[p,d,s] = 0) then not(s in r[p,d])endif);


constraint forall(d in DAY)((sum(p in PERSON)(opening in r[p,d]) + sum(p in PERSON)(morning in r[p,d])  + sum(p in PERSON)(lunch in r[p,d])) >= 8);

constraint forall(d in DAY)((sum(p in PERSON)(lunch in r[p,d]) + sum(p in PERSON)(afternoon in r[p,d])  + sum(p in PERSON)(closing in r[p,d])) >= 8);


var int: value = sum(p in PERSON, d in DAY, s in SHIFT)(if(s in r[p,d])then pref[p,d,s] else 0 endif);
solve maximize value;


output ["r = array2d(PERSON,DAY,\(r))\n;"];                                             
