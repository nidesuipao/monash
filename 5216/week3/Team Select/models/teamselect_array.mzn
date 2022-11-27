
enum PLAYER = {Ant, Bee, Chu, Deb, Eve, Fin, Ged, Hel, Ila, Jan, Kim};

enum CAPTAIN = {Xavier, Yuri, Zena};

array[CAPTAIN,PLAYER] of int: value = 
   [| 2, 5, 6, 8, 9, 5, 8, 7, 7, 4, 6
    | 9, 8, 4, 7, 6, 4, 5, 3, 5, 5, 7
    | 8, 4, 3, 3, 6, 2, 5, 5, 3, 2, 5 |];

set of int: POS = 1..6;

array[CAPTAIN,POS] of var PLAYER: team;

set of PLAYER: goalies = {Ant, Bee};
set of PLAYER: defence = {Chu, Deb, Eve, Fin};
set of PLAYER: offence = {Ged, Hel, Ila,Jan, Kim};

constraint forall(t in CAPTAIN)(sum(i in POS)(team[t,i] in goalies) >= 1);
constraint forall(t in CAPTAIN)(sum(i in POS)(team[t,i] in defence) >= 2);
constraint forall(t in CAPTAIN)(sum(i in POS)(team[t,i] in offence) >= 2);

constraint sum(i in POS, j in POS)(team[Xavier,i] = team[Yuri,j]) <= 2;
constraint sum(i in POS, j in POS)(team[Xavier,i] = team[Zena,j]) <= 2;

var 0..infinity: values;
solve maximize values;

values = sum(i in POS)(value[Xavier, team[Xavier, i]]) + 
         sum(i in POS)(value[Yuri, team[Yuri, i]]) +
         sum(i in POS)(value[Zena, team[Zena, i]]);

include "alldifferent.mzn";
constraint forall(t in CAPTAIN)(alldifferent(i in  POS)(team[t,i]));
constraint forall(captain in CAPTAIN, i in 1..card(POS)-1)(team[captain, i] >= team[captain, i+1]);
output ["team = array2d(CAPTAIN,POS,\(team));\n"];

