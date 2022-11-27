
enum PLAYER = {Ant, Bee, Chu, Deb, Eve, Fin, Ged, Hel, Ila, Jan, Kim};

enum CAPTAIN = {Xavier, Yuri, Zena};




array[CAPTAIN,PLAYER] of int: value = 
   [| 2, 5, 6, 8, 9, 5, 8, 7, 7, 4, 6
    | 9, 8, 4, 7, 6, 4, 5, 3, 5, 5, 7
    | 8, 4, 3, 3, 6, 2, 5, 5, 3, 2, 5 |];

var set of PLAYER: xavier;
var set of PLAYER: yuri;
var set of PLAYER: zena;
set of PLAYER: goalies = {Ant, Bee};
set of PLAYER: defence = {Chu, Deb, Eve, Fin};
set of PLAYER: offence = {Ged, Hel, Ila,Jan, Kim};
var 0..infinity: values;
solve maximize values;

values = sum(play in xavier)(value[Xavier, play]) + 
         sum(play in yuri)(value[Yuri, play]) + 
         sum(play in zena)(value[Zena, play]);
         
constraint (sum(play in xavier)(1)) = 6;
constraint (sum(play in yuri)(1)) = 6;
constraint (sum(play in zena)(1)) = 6;

constraint forall(team in [xavier, yuri, zena])(sum(player in team)(player in goalies) >= 1);
constraint forall(team in [xavier, yuri, zena])(sum(player in team)(player in defence) >= 2);
constraint forall(team in [xavier, yuri, zena])(sum(player in team)(player in offence) >= 2);

constraint sum(play in xavier)(play in yuri) <= 2;
constraint sum(play in xavier)(play in zena) <= 2;
% constraint sum(play in zena)(play in yuri) <= 2;

output ["xavier = \(xavier);\nyuri = \(yuri);\nzena = \(zena);\n"];

