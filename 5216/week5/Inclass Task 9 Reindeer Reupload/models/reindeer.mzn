set of int: POS = 1..4;
enum REINDEER = { Lancer, Quentin, Ezekiel, Rudy };
array[REINDEER] of var POS: x;

constraint forall(i,j in REINDEER where i < j)(x[i] != x[j]);
constraint abs(x[Lancer] - x[Ezekiel]) != 1;
constraint x[Rudy] < x[Lancer] \/ x[Rudy] < x[Quentin];
output ["x = array1d(REINDEER,\(x));\n"];
